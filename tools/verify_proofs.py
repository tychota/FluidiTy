#!/usr/bin/env python3
"""Build, audit, export and independently check the bounded Lean pilots."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PROOFS = ROOT / 'proofs'
PINS = json.loads((ROOT / 'tools/proof-toolchain.json').read_text())
TARGETS = ('Fluidity.pressure_quadratic_form_eq', 'Fluidity.pressure_psd',
           'Fluidity.fixed_point_order_independent', 'Fluidity.fixed_point_partial_bound')
ALLOWED = {'propext', 'Classical.choice', 'Quot.sound'}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require_nonempty(path):
    if not Path(path).is_file() or Path(path).stat().st_size == 0:
        raise ValueError(f'missing or empty export: {path}')


def validate_receipt(receipt, checker):
    for key, expected in [('schema_version', 1), ('nano_lean_commit', PINS['nano_lean_commit']),
                          ('rust_version', PINS['rust_version']), ('checker_sha256', sha256(checker))]:
        if receipt.get(key) != expected:
            raise ValueError(f'checker provenance mismatch: {key}')


def audit_axioms(output):
    found = {}
    for name, axioms in re.findall(r"'([^']+)' depends on axioms: \[([^\]]*)\]", output):
        if name in TARGETS:
            if name in found:
                raise ValueError(f'duplicate axiom target: {name}')
            found[name] = {a.strip() for a in axioms.split(',') if a.strip()}
    if set(found) != set(TARGETS) or any(not a <= ALLOWED for a in found.values()):
        raise ValueError(f'missing target or forbidden axiom: {found}')
    return {name: sorted(axioms) for name, axioms in found.items()}


def checker_command(checker, prelude, proof):
    return [str(checker), '--prelude', str(prelude), '--axioms',
            'propext,Classical.choice,Quot.sound', '--no-cache', str(proof)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('lean-bin', 'exporter', 'checker', 'checker-provenance', 'output'):
        parser.add_argument('--' + option, required=True, type=Path)
    parser.add_argument('--lake-packages', type=Path)
    args = parser.parse_args()
    lean_bin, exporter, checker = (p.resolve() for p in (args.lean_bin, args.exporter, args.checker))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    evidence_file = output / 'evidence.json'
    evidence_file.unlink(missing_ok=True)
    env = dict(os.environ, PATH=str(lean_bin) + os.pathsep + os.environ.get('PATH', ''))
    commands = []

    def run(cmd):
        cmd = list(map(str, cmd))
        result = subprocess.run(cmd, cwd=PROOFS, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        commands.append({'argv': cmd, 'cwd': str(PROOFS), 'exit_code': result.returncode, 'output': result.stdout})
        if result.returncode:
            raise RuntimeError(f'command failed ({result.returncode}): {cmd}\n{result.stdout}')
        return result.stdout

    for exe in (lean_bin / 'lean', lean_bin / 'lake', exporter, checker):
        if not exe.is_file() or not os.access(exe, os.X_OK):
            raise ValueError(f'missing executable: {exe}')
    receipt = json.loads(args.checker_provenance.read_text())
    validate_receipt(receipt, checker)
    lean_version = run([lean_bin / 'lean', '--version'])
    if not re.search(r'Lean \(version 4\.34\.1,.*commit ' + PINS['lean_commit'] + r'\b', lean_version):
        raise ValueError('unexpected Lean version')
    exporter_version = run([exporter, '--version']).strip()
    if exporter_version != 'olean-export ' + PINS['exporter_version']:
        raise ValueError('unexpected exporter version')
    manifest = json.loads((PROOFS / 'lake-manifest.json').read_text())
    if not any(p.get('name') == 'mathlib' and p.get('rev') == PINS['mathlib_commit'] and p.get('url') == 'https://github.com/leanprover-community/mathlib4.git' for p in manifest['packages']):
        raise ValueError('mathlib manifest pin mismatch')
    if (PROOFS / 'lean-toolchain').read_text().strip() != 'leanprover/lean4:' + PINS['lean_tag']:
        raise ValueError('Lean toolchain pin mismatch')
    for source in list((PROOFS / 'Fluidity').glob('*.lean')) + [PROOFS / 'Fluidity.lean']:
        if re.search(r'\b(?:sorry|admit|sorryAx|axiom|native_decide)\b', source.read_text()):
            raise ValueError(f'forbidden admission or axiom token in {source}')
    lake = [lean_bin / 'lake']
    override = None
    if args.lake_packages:
        path = args.lake_packages.resolve()
        override = {'sha256': sha256(path), 'contents': json.loads(path.read_text())}
        lake += ['--packages=' + str(path)]
    run(lake + ['--wfail', 'build'])
    audit = audit_axioms(''.join(run(lake + ['env', 'lean', 'Fluidity/' + file + '.lean']) for file in ('Pressure', 'FixedPoint')))
    with tempfile.TemporaryDirectory(prefix='proof-check-', dir=output) as temp:
        prelude, proof = Path(temp) / 'init.blean', Path(temp) / 'fluidity.blean'
        run(lake + ['env', exporter, 'Init', '-f', 'blean', '-o', prelude])
        require_nonempty(prelude)
        const_args = [item for target in TARGETS for item in ('-c', target)]
        run(lake + ['env', exporter, 'Fluidity', *const_args, '-f', 'blean', '-o', proof])
        require_nonempty(proof)
        run(checker_command(checker, prelude, proof))
        evidence = {'pins': PINS, 'lean_version': lean_version.strip(), 'exporter_version': exporter_version,
                    'checker_provenance': receipt, 'checker_provenance_sha256': sha256(args.checker_provenance),
                    'trust': 'Checker build receipt is an external assertion, not proof of a reproducible build.',
                    'lake_packages_override': override, 'axioms': audit,
                    'sha256': {'lean': sha256(lean_bin / 'lean'), 'lake': sha256(lean_bin / 'lake'),
                               'exporter': sha256(exporter), 'checker': sha256(checker),
                               'prelude': sha256(prelude), 'proof_export': sha256(proof)}, 'commands': commands}
        evidence_file.write_text(json.dumps(evidence, indent=2) + '\n')
    print(f'Proof verification succeeded; evidence: {evidence_file}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as error:
        print(f'Proof verification failed: {error}', file=sys.stderr)
        sys.exit(1)
