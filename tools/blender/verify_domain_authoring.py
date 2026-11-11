"""Exercise real domain, material and collider authoring in the pinned Blender."""

import hashlib
import importlib
import sys
import tempfile
from pathlib import Path

import bpy

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "blender"), str(Path(__file__).resolve().parent)]
arguments = sys.argv[sys.argv.index("--") + 1 :]
installed = "--installed" in arguments
cpu = "--cpu" in arguments
arguments = [argument for argument in arguments if argument not in {"--installed", "--cpu"}]
root, binaries = (Path(value).resolve() for value in arguments)
root.mkdir(parents=True, exist_ok=False)
fluidity = importlib.import_module("bl_ext.user_default.fluidity" if installed else "fluidity")
if installed:
    bpy.ops.wm.open_mainfile(filepath=str(Path(fluidity.__file__).parent / "examples/Water Tank.blend"))
else:
    fluidity.register()
    assert bpy.ops.fluidity.create_setup() == {"FINISHED"}
fluidity.jobs.BIN_DIR = binaries
bpy.context.preferences.filepaths.save_version = 0
scene = bpy.context.scene
settings = scene.fluidity
if cpu:
    settings.backend = "cpu"
assert settings.boundary_mode == "TANK"
assert settings.domain and settings.liquid and not settings.paddle
assert settings.liquid != settings.domain
assert settings.density == 1000
assert bpy.ops.fluidity.apply_preset(preset="WATER") == {"FINISHED"}
assert abs(settings.surface_tension - 0.072) < 1e-6
settings.initial_velocity = (0.1, 0.2, 0.3)
scene.frame_end = 3
bpy.ops.wm.save_as_mainfile(filepath=str(root / "Domain authoring.blend"))
packet = fluidity.scene_inputs(scene)
assert packet.startswith("FLUIDITY-BLENDER-2\n")
assert bpy.ops.fluidity.check_scene() == {"FINISHED"}, settings.message
assert bpy.ops.fluidity.bake() == {"FINISHED"}, settings.message
import time

deadline = time.monotonic() + 90
while fluidity.is_running(scene):
    assert time.monotonic() < deadline, settings.message
    fluidity.tick()
    time.sleep(0.005)
assert settings.committed_frame == 3, settings.message
initial = fluidity.playback.inspection(root / "FluidiTy Bake", 0)
assert initial["points"]
assert all(
    max(abs(a - b) for a, b in zip(point["velocity"], settings.initial_velocity, strict=True)) < 1e-6
    for point in initial["points"]
)
assert bpy.ops.fluidity.attach() == {"FINISHED"}, settings.message
settings.density = 950
assert bpy.ops.fluidity.resume() == {"CANCELLED"}
assert "changed" in settings.message
settings.density = 1000
for mode in ("FREE", "PLANE", "SLAB"):
    settings.collider = None
    settings.boundary_mode = mode
    if mode in {"PLANE", "SLAB"}:
        assert bpy.ops.fluidity.add_input(role="collider") == {"FINISHED"}
        assert settings.collider
    assert bpy.ops.fluidity.check_scene() == {"FINISHED"}, settings.message
    settings.bake_cache = "//" + mode + " Bake"
    assert bpy.ops.fluidity.bake() == {"FINISHED"}, settings.message
    deadline = time.monotonic() + 90
    while fluidity.is_running(scene):
        assert time.monotonic() < deadline, settings.message
        fluidity.tick()
        time.sleep(0.005)
    assert settings.committed_frame == 3, (mode, settings.message)
    print("PASS: real", mode, "bake")

retained_collider = settings.collider
settings.boundary_mode = "TANK"
assert bpy.ops.fluidity.check_scene() == {"FINISHED"}, settings.message
assert settings.collider == retained_collider
settings.boundary_mode = "FREE"
assert bpy.ops.fluidity.check_scene() == {"FINISHED"}, settings.message
settings.boundary_mode = "SLAB"
assert settings.collider == retained_collider

# An editable preset changes material alone; it cannot reset authored data.
before = (settings.domain, settings.liquid, tuple(settings.initial_velocity), settings.bake_cache)
assert bpy.ops.fluidity.apply_preset(preset="HIGH_TENSION") == {"FINISHED"}
assert (settings.domain, settings.liquid, tuple(settings.initial_velocity), settings.bake_cache) == before
assert abs(settings.surface_tension - 0.15) < 1e-6

# Every exposed physical quantity participates in canonical resolved settings/state.
with tempfile.TemporaryDirectory(dir=root) as temporary:

    def digest(index):
        directory = Path(temporary) / str(index)
        directory.mkdir()
        resolved, state = fluidity.jobs.prepare(fluidity.scene_inputs(scene), settings.backend, directory)
        return hashlib.sha256(resolved.read_bytes() + state.read_bytes()).hexdigest()

    baseline = digest("base")
    for name, value in (
        ("density", 900),
        ("surface_tension", 0.05),
        ("friction", 0.3),
        ("maximum_step", 0.005),
        ("energy_blend", 0.2),
        ("pressure_iterations", 128),
        ("initial_velocity", (0, 0, 0)),
    ):
        previous = getattr(settings, name)
        if name == "initial_velocity":
            previous = tuple(previous)
        setattr(settings, name, value)
        assert digest(name) != baseline, name
        setattr(settings, name, previous)

print("PASS: real tank domain, water preset, initial velocity, bake, attach and changed-material resume rejection")
