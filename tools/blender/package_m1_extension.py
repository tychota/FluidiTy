"""Stage the M1-B local evaluation ZIP with the pinned Blender extension CLI."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    blender = args.blender.resolve()
    build = subprocess.check_output([blender, "--version"], encoding="utf-8")
    if not build.startswith("Blender 5.2.3 LTS") or "build hash: e13b728c6880" not in build:
        parser.error("Blender 5.2.3 LTS e13b728c6880 is required")
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
        subprocess.run(
            [*common, "--command", "extension", "build", "--source-dir", stage, "--output-filepath", package],
            check=True,
        )
        subprocess.run([*common, "--command", "extension", "validate", package], check=True)
        shutil.copyfile(example, output / example.name)
    artifacts = [package, output / "First Bake.blend", args.solver, args.bridge]
    digests = {item.name: hashlib.file_digest(item.open("rb"), "sha256").hexdigest() for item in artifacts}
    (output / "sha256.json").write_text(json.dumps(digests, indent=2) + "\n", encoding="utf-8")
    print(package)


if __name__ == "__main__":
    main()
