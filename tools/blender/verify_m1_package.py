"""Install the ZIP offline, exercise the example, and render with no extension."""

import argparse
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True)
    env = os.environ.copy()
    env["BLENDER_USER_RESOURCES"] = str(root / "installed-blender")
    env["TMP"] = env["TEMP"] = str(root)
    common = [args.blender.resolve(), "--background", "--offline-mode", "--disable-autoexec"]
    subprocess.run(
        [
            *common,
            "--factory-startup",
            "--command",
            "extension",
            "install-file",
            "-r",
            "user_default",
            "-e",
            args.package.resolve(),
        ],
        env=env,
        timeout=30,
        check=True,
    )
    here = Path(__file__).resolve().parent
    project = root / "Projet FluidiTy"
    subprocess.run(
        [
            *common,
            "--python-exit-code",
            "23",
            "--python",
            here / "verify_m1_extension.py",
            "--",
            project,
            "--installed",
        ],
        env=env,
        timeout=120,
        check=True,
    )
    relocated = project.with_name(project.name + " – déplacé")
    env["BLENDER_USER_RESOURCES"] = str(root / "cache-only-blender")
    subprocess.run(
        [
            *common,
            "--factory-startup",
            relocated / "First bake – fluide.blend",
            "--python-exit-code",
            "23",
            "--python",
            here / "verify_m1_render.py",
            "--",
            relocated,
        ],
        env=env,
        timeout=120,
        check=True,
    )


if __name__ == "__main__":
    main()
