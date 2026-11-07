"""Export small manufactured numerical examples."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from .fixtures import atomic_write, canonical_bytes, write_fixture


def examples():
    # Values are derived by hand, independently of the reference solvers.
    return {
        'two_cell_pressure': ({'grid': [2, 1], 'dt': [1.], 'velocity': [0., 2., 0., 0., 0., 0., 0.], 'jump': [0., 3., 0., 0., 0., 0., 0.], 'expected_pressure': [0., 5.], 'expected_velocity': [0.]*7}, {'defaults': {'bulk_modulus': 'infinite', 'gauge_cell': 0}}),
        'compliant_pressure': ({'grid': [1, 1], 'dt': [1.], 'velocity': [0.]*4, 'bulk_modulus': [2.], 'source': [3.], 'expected_pressure': [6.], 'expected_velocity': [0.]*4}, {}),
        'affine_apic': ({'position': [0., 0.], 'sample_positions': [[-1., 0.], [1., 0.], [0., -1.], [0., 1.]], 'weights': [.25]*4, 'values': [1., 5., 0., 6.], 'expected_value': [3.], 'expected_affine': [2., 3.]}, {}),
        'stiff_viscosity': ({'velocity': [1., -1.], 'mass': [1., 1.], 'strain_operator': [[1., -1.]], 'strain_weight': [10.], 'boundary_strain': [0.], 'dt': [1.], 'expected_velocity': [1./21., -1./21.]}, {'expected_method': 'implicit'}),
    }


def export(directory, overwrite=False):
    directory = Path(directory)
    names = examples()
    targets = [directory / (name + '.json') for name in names] + [directory / 'manifest.json']
    if not overwrite and any(p.exists() for p in targets):
        raise FileExistsError('export already exists; use --overwrite')
    if overwrite:
        (directory / 'manifest.json').unlink(missing_ok=True)
    hashes = {}
    for name, (values, metadata) in names.items():
        arrays = {k: np.asarray(v) for k, v in values.items()}
        units = {k: '1' for k in arrays}
        metadata = {**metadata, 'units': units, 'unit_system': 'dimensionless manufactured example'}
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
