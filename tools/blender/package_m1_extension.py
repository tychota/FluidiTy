"""Stage the M1-B local evaluation ZIP with the pinned Blender extension CLI."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import tomllib

BLENDER_BUILDS = {
    ("Blender 5.2.3 LTS", "e13b728c6880"): "5.2.3",
    ("Blender 5.3.0", "02f0899d78e3"): "5.3.0",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    blender = args.blender.resolve()
    build = subprocess.check_output([blender, "--version"], encoding="utf-8")
    banner = build.splitlines()[0]
    match = re.search(r"^\s*build hash: ([0-9a-f]+)\s*$", build, re.MULTILINE)
    build_hash = match[1] if match else ""
    minimum_version = BLENDER_BUILDS.get((banner, build_hash))
    if minimum_version is None:
        parser.error(f"Unsupported package builder: {banner} ({build_hash}). Use a build listed in BLENDER_BUILDS.")
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    root = Path(__file__).resolve().parents[2]
    common = [blender, "--background", "--factory-startup", "--offline-mode", "--disable-autoexec"]
    version = tomllib.loads((root / "blender/fluidity/blender_manifest.toml").read_text())["version"]
    subprocess.run([sys.executable, root / "tools/blender/compile_translations.py"], check=True)
    package = output / f"fluidity-{version}-windows-x64.zip"
    with tempfile.TemporaryDirectory(dir=output) as temp:
        stage = Path(temp) / "fluidity"
        shutil.copytree(root / "blender" / "fluidity", stage, ignore=shutil.ignore_patterns("__pycache__", "bin"))
        manifest = stage / "blender_manifest.toml"
        manifest_text, replacements = re.subn(
            r'^blender_version_min = "[^"]+"$',
            f'blender_version_min = "{minimum_version}"',
            manifest.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
        if replacements != 1:
            raise ValueError("Expected exactly one minimum Blender version in the manifest")
        manifest.write_text(manifest_text, encoding="utf-8")
        binaries = stage / "bin"
        binaries.mkdir()
        for source, name in ((args.solver, "simulate"), (args.bridge, "blender_bridge")):
            shutil.copyfile(source, binaries / f"fluidity_{name}.exe")
        examples = stage / "examples"
        examples.mkdir()
        example = examples / "First Bake.blend"
        subprocess.run(
            [*common, "--python-exit-code", "23", "--python", root / "tools/blender/make_m1_example.py", "--", example],
            check=True,
        )
        domain_example = examples / "Water Tank.blend"
        subprocess.run(
            [
                *common,
                "--python-exit-code",
                "23",
                "--python",
                root / "tools/blender/make_domain_example.py",
                "--",
                domain_example,
            ],
            check=True,
        )
        subprocess.run(
            [*common, "--command", "extension", "build", "--source-dir", stage, "--output-filepath", package],
            check=True,
        )
        subprocess.run([*common, "--command", "extension", "validate", package], check=True)
        shutil.copyfile(example, output / example.name)
        shutil.copyfile(domain_example, output / domain_example.name)
    artifacts = [package, output / "First Bake.blend", output / "Water Tank.blend", args.solver, args.bridge]
    digests = {item.name: hashlib.file_digest(item.open("rb"), "sha256").hexdigest() for item in artifacts}
    (output / "sha256.json").write_text(json.dumps(digests, indent=2) + "\n", encoding="utf-8")
    (output / "blender-build.json").write_text(
        json.dumps({"banner": banner, "build_hash": build_hash, "minimum_version": minimum_version}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(package)


if __name__ == "__main__":
    main()
