"""Exercise three authored liquid sources through Blender and the native solver."""

import json
from pathlib import Path
import subprocess
import sys

import bpy


def box(name, center, dimensions):
    bpy.ops.mesh.primitive_cube_add(size=2, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.scale = tuple(width * 0.5 for width in dimensions)
    return obj


def run(bridge, solver, output):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "blender"))
    import fluidity
    from fluidity.scene_inputs import scene_inputs

    bpy.ops.wm.read_factory_settings(use_empty=True)
    fluidity.register()
    scene = bpy.context.scene
    scene.render.fps = 24
    scene.render.fps_base = 1
    scene.frame_start, scene.frame_end = 0, 2
    scene.use_gravity = False
    settings = scene.fluidity
    settings.boundary_mode = "FREE"
    settings.domain = box("Domain", (3, -3, 3), (5, 5, 5))
    settings.cell_width = 1
    settings.initial_velocity = (0.24, 0, 0)
    settings.phase_tension = 0.031
    definitions = (
        ("Clear water", (2.5, -2.5, 2.5), (0.5, 0.5, 1), 8, 4, 1000, 0.00025),
        ("Dyed water", (2.5, -2.5, 2.5), (0.75, 1, 1), 14, 4, 1020, 0.00075),
        ("Oil", (2.92, -2.57, 2.64), (0.99, 0.99, 0.99), 25, 5, 820, 0.001),
    )
    for name, center, widths, substance, phase, density, emitted_volume in definitions:
        assert bpy.ops.fluidity.liquid_source() == {"FINISHED"}
        source = settings.liquid_sources[-1]
        source.geometry = box(name, center, widths)
        source.substance_id = substance
        source.phase_id = phase
        source.density = density
        source.emission_rate = 25
        source.emitted_volume = emitted_volume

    output.mkdir()
    packet, resolved, initial = (output / name for name in ("scene.txt", "scene.flrs", "initial.flst"))
    packet.write_text(scene_inputs(scene), encoding="ascii")
    assert packet.read_text(encoding="ascii").startswith("FLUIDITY-BLENDER-3\n")
    cache = output / "simulate"

    def execute(log, *args):
        command = subprocess.run(args, capture_output=True, encoding="utf-8", timeout=90)
        (output / log).write_text(command.stdout + command.stderr, encoding="utf-8")
        command.check_returncode()
        return command.stdout

    for label, old, replacement in (
        ("unknown-material", "1 8 25.0", "1 99 25.0"),
        ("reversed-pair", "4 5 ", "5 4 "),
    ):
        authored = packet.read_text(encoding="ascii")
        invalid = output / f"{label}.txt"
        invalid.write_text(authored.replace(old, replacement, 1), encoding="ascii")
        assert invalid.read_text(encoding="ascii") != authored
        bad_scene, bad_state = output / f"{label}.flrs", output / f"{label}.flst"
        refused = subprocess.run([bridge, "resolve", solver, "cpu", invalid, bad_scene, bad_state],
                                 capture_output=True, encoding="utf-8", timeout=30)
        (output / f"{label}.log").write_text(refused.stderr, encoding="utf-8")
        assert refused.returncode != 0 and not bad_scene.exists() and not bad_state.exists()
    execute("resolve.log", bridge, "resolve", solver, "cpu", packet, resolved, initial)
    execute("simulate.log", solver, "run", "--stage", "simulate", "--scene", resolved,
            "--initial-state", initial, "--cache", cache, "--frame-end", "1")
    expected = [(4, 8), (4, 14), (5, 25)]
    for frame in (0, 1):
        inspected = json.loads(execute(f"inspect-{frame}.log", solver, "check", "--cache", cache, "--frame", str(frame)))
        particles = sorted(inspected["particle_composition"], key=lambda particle: particle["particle_id"])
        actual = [(p["phase_id"], p["constituents"][0]["substance_id"]) for p in particles]
        assert actual == expected * (frame + 1), actual
        assert all(len(p["constituents"]) == 1 and p["constituents"][0]["volume_fraction"] == 1 for p in particles)
    print("PASS Blender-authored sources and one accepted child per emitter")


def run_viscous(bridge, solver, output):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "blender"))
    import fluidity
    from fluidity.scene_inputs import scene_inputs

    bpy.ops.wm.read_factory_settings(use_empty=True)
    fluidity.register()
    scene = bpy.context.scene
    scene.render.fps = 24
    scene.render.fps_base = 1
    scene.frame_start, scene.frame_end = 0, 2
    scene.use_gravity = False
    settings = scene.fluidity
    settings.boundary_mode = "TANK"
    settings.domain = box("Tank", (3, -3, 3), (2, 2, 2))
    settings.cell_width = 1
    settings.planar_fill = True
    settings.fill_height = 3
    settings.initial_velocity = (0.125, 0, 0)
    settings.phase_tension = 0.031
    for name, x, substance, phase, density, viscosity in (
        ("Water", 2.5, 8, 4, 1000, 40),
        ("Oil", 3.5, 25, 5, 820, 120),
    ):
        assert bpy.ops.fluidity.liquid_source() == {"FINISHED"}
        source = settings.liquid_sources[-1]
        source.geometry = box(name, (x, -2.5, 2.5), (0.5, 0.5, 0.5))
        source.substance_id = substance
        source.phase_id = phase
        source.density = density
        source.override_viscosity = True
        source.viscosity = viscosity

    output.mkdir()
    packet, resolved, initial = (output / name for name in ("scene.txt", "scene.flrs", "initial.flst"))
    authored = scene_inputs(scene)
    assert authored.startswith("FLUIDITY-BLENDER-4\n")
    assert "\n8 4 1000.0 40.0\n25 5 820.0 120.0\n" in authored
    assert "\n1 3.0\n" in authored
    packet.write_text(authored, encoding="ascii")
    subprocess.run([bridge, "resolve", solver, "cpu", packet, resolved, initial], check=True, timeout=30)
    cache = output / "simulate"
    command = subprocess.run(
        [solver, "run", "--stage", "simulate", "--scene", resolved, "--initial-state", initial,
         "--cache", cache, "--frame-end", "1", "--trace"],
        capture_output=True, encoding="utf-8", timeout=90,
    )
    (output / "simulate.log").write_text(command.stdout + command.stderr, encoding="utf-8")
    command.check_returncode()
    traces = [record["accepted_measurement"] for record in map(json.loads, command.stdout.splitlines())
              if record["event"] == "trace"]
    assert traces and all(
        0 <= trace["viscous_max_momentum_residual_kg_m_per_s"] < 1e-5 for trace in traces
    ), traces
    print("PASS Blender-authored planar tank with distinct immiscible viscosity")


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:]
    if args[0] == "viscous":
        run_viscous(*(Path(path).resolve() for path in args[1:]))
    else:
        run(*(Path(path).resolve() for path in args))
