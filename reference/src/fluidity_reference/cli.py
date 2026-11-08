"""Export small manufactured numerical examples."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from .fixtures import atomic_write, canonical_bytes, write_fixture
from .liquid_boundary_3d_fixtures import uniform_mac_liquid_boundary_3d_fixture
from .pressure_3d_fixtures import uniform_mac_pressure_3d_fixture
from .transfer_3d_fixtures import uniform_mac_transfer_3d_fixture


def examples():
    # Values are derived by hand, independently of the reference solvers.
    return {
        'two_cell_pressure': ({'grid': [2, 1], 'spacing': [1.], 'origin': [0., 0.], 'dt': [1.], 'velocity': [0., 2., 0., 0., 0., 0., 0.], 'jump': [0., 3., 0., 0., 0., 0., 0.], 'expected_pressure': [0., 5.], 'expected_velocity': [0.]*7}, {'defaults': {'bulk_modulus': 'infinite', 'gauge_cell': 0}}),
        'compliant_pressure': ({'grid': [1, 1], 'spacing': [1.], 'origin': [0., 0.], 'dt': [1.], 'velocity': [0.]*4, 'bulk_modulus': [2.], 'source': [3.], 'expected_pressure': [6.], 'expected_velocity': [0.]*4}, {}),
        'affine_apic': ({'position': [0., 0.], 'sample_positions': [[-1., 0.], [1., 0.], [0., -1.], [0., 1.]], 'weights': [.25]*4, 'values': [1., 5., 0., 6.], 'expected_value': [3.], 'expected_affine': [2., 3.]}, {}),
        'stiff_viscosity': ({'velocity': [1., -1.], 'mass': [1., 1.], 'strain_operator': [[1., -1.]], 'strain_weight': [10.], 'boundary_strain': [0.], 'dt': [1.], 'expected_velocity': [1./21., -1./21.]}, {'expected_method': 'implicit'}),
    }


def export(directory, overwrite=False):
    directory = Path(directory)
    names = examples()
    names['uniform_mac_transfer_3d'] = uniform_mac_transfer_3d_fixture()
    names['uniform_mac_pressure_3d'] = uniform_mac_pressure_3d_fixture()
    names['uniform_mac_liquid_boundary_3d'] = uniform_mac_liquid_boundary_3d_fixture()
    targets = [directory / (name + '.json') for name in names] + [directory / 'manifest.json']
    if not overwrite and any(p.exists() for p in targets):
        raise FileExistsError('export already exists; use --overwrite')
    if overwrite:
        (directory / 'manifest.json').unlink(missing_ok=True)
    hashes = {}
    provenance = {
        'two_cell_pressure': 'Hand assembly: unit interior flux 2 and jump 3 give gauge-relative pressure 5 and zero corrected flux.',
        'compliant_pressure': 'Hand assembly: source 3 times bulk modulus 2 gives pressure 6 with zero face velocity.',
        'affine_apic': 'Hand evaluation of the affine field 3 + 2x + 3y at four symmetric samples.',
        'stiff_viscosity': 'Hand solution of the two-by-two backward Euler system gives velocities +1/21 and -1/21.',
    }
    policies = {
        'two_cell_pressure': 'PressureInputs defaults: closed boundaries, unit mobility and aperture, infinite bulk modulus; first cell gauge.',
        'compliant_pressure': 'PressureInputs defaults: closed boundaries, unit mobility and aperture; supplied bulk modulus removes gauge.',
        'affine_apic': 'local_apic_fit uses centered full moment and supplied normalized weights.',
        'stiff_viscosity': 'viscous_step auto mode with default safety 0.01.',
    }
    for name, (values, metadata) in names.items():
        arrays = {k: np.asarray(v) for k, v in values.items()}
        if name in provenance:
            units = {k: '1' for k in arrays}
            metadata = {**metadata, 'units': units, 'unit_system': 'dimensionless manufactured example',
                        'provenance': provenance[name], 'default_policy': policies[name],
                        'tolerances': {'rtol': 1e-12, 'atol': 1e-14}}
        path = write_fixture(directory, name, arrays, metadata)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_write(directory / 'manifest.json', canonical_bytes({'schema_version': 1, 'fixtures': hashes}))


def main():
    parser = argparse.ArgumentParser(prog='fluidity-reference')
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('export')
    command.add_argument('directory')
    command.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    try:
        export(args.directory, args.overwrite)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
