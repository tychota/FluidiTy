# M1-B implementation and qualification record

This is a local evaluation build of the first Blender bake. The adapter and
extension evidence below are independent of M1 solver sign-off.

## M1 acceptance status

GitHub `main` was `b009510852e7b306e8abc8ecab9828c7f97a2d52` when work began.
Its tree matches `3e0526e`. The retained receipts below span revisions and build
artifacts. They are not a matched acceptance matrix for the final piston implementation.

| Frozen acceptance row | Retained CPU evidence | Retained RTX evidence | Status for M1 sign-off |
| --- | --- | --- | --- |
| Free fall and tank layout | PIC, default and retained-mode controls | Corresponding SM120 controls | Repeat on the final pinned pair |
| Hydrostatic column and 10 s material budget | 241 published frames at `e815a2e` | 241 published frames at the same revision | Retained speed at 2 s was 3.956e-7 m/s and peak relative point-volume drift was 1.554e-7; repeat on the final pinned pair |
| Dam break | PIC, default and retained 49-frame receipts | Corresponding 49-frame receipts | Retained material budgets met; repeat on the final pinned pair |
| Static sphere | Checked rest seed, coarse and fine | Corresponding checked rest seed runs | Retained zero reported pressure and speed errors after the repair; repeat on the final pinned pair |
| Oscillating drop | 216-frame public run, 4.90524% frequency error and 0.135843 damping | Corresponding 216-frame run plus the naming-closure run at `3e0526e` | Retained physical limits met; repeat on the final pinned pair |
| Open-tank piston, primary and refined | Frozen receiving error 7.617% and 7.625% against 1% | Retained matched wet-wall trajectories and publication profiles; no closing receiving-volume receipt | **Open** |
| CPU/RTX replay, CUDA reliability, sanitizer and throughput | See native toolchain and optimization receipts | See native toolchain and optimization receipts | Re-pin final artifacts, verify required rosters and repeat the matched campaign |

See [M1 transport evidence](../references/m1-transport-evidence.md#m1-closure-review),
[piston geometry diagnostics](../references/m1-geometry-diagnostics.md#piston-material-mesh),
the [frozen scene plan](../superpowers/plans/2026-11-09-real-3d-liquid-acceptance-and-throughput.md)
and the [native capture procedure](../native/README.md#frozen-tank-capture-provenance).
Good wall-versus-solver flux and particle-volume agreement do not close the receiving
gate. The independent trilinear hexahedral diagnostic still measured +1.0963% for the
primary and +0.9782% for the refined case. The receiving gate needs independently
verified wet-wall geometry, ≤1% receiving mismatch, and the original ≤0.25% spatial
and temporal comparisons. After that correction, pin the final source revision,
solver/fixture SHA-256, resolved inputs, Windows toolchain and SM120 driver/runtime,
and retain full CPU and RTX run, replay, pressure, reliability and performance receipts.

## Adapter and extension evidence

Test target: Windows x64; Blender 5.2.3 LTS build `e13b728c6880`;
MSVC 19.44.35229 and Windows SDK 10.0.26100.0; CUDA 13.4.59;
GeForce RTX 5060 Ti, driver 610.88, compute capability 12.0, executed SM120
image, CUDA driver and runtime APIs 13030. The tested solver SHA-256 was
`ee953ed5f397e0f3418b408197c19267752dd68b3f062dc2cab522fa2f28eab8`.
The bridge links the checked M1 cache reader and Alembic 1.8.8 Ogawa.
The evaluation ZIP SHA-256 is `e1281041fe3e943c6479732ed4a25c702e3939d6416a1e166a48a90e57f2eff5`.
The package builder also records the example and bridge hashes in `sha256.json`.

| Check | Observed in pinned Blender |
| --- | --- |
| Real committed Points | RTX oscillating drop frames 0–3: 552 points each. RTX emission control: 0, 1, 2, 3. Compared each evaluated position and radius to the verified committed frame within 1e-6 m after moving the project. Revisited frames in reverse order. |
| Commit authority | Removing a binary frame commit while keeping its JSON mirror, or corrupting its point blob, stopped export and left no archive. |
| Offline package and job lifecycle | Installed from disk in an isolated profile with online access disabled. The public operators rejected invalid FPS and animated scene settings, ran Preview through frame 3 in a separate cache with a distinct resolved fingerprint, and cancelled and resumed the RTX bake through frame 12. Committed files before cancellation kept their exact bytes. Changing the paddle animation blocked Resume without changing committed bytes. |
| Inspect and binding | Inspected frame 3 and its eight points; verified selected point ID, position and velocity display. Repeat attachment preserved the binding and edited display size. Moved the project and then its cache independently; Relink updated the native point and transform readers. Failed replacement preserved the binding and committed content. |
| Cache-only render | A fresh factory process with no extension and no frame callbacks rendered committed frames 0 and 10 through the saved Points Geometry Nodes group. Both PNGs contain opaque point geometry and transparent background, and their hashes differ. |
| Languages | The pinned GUI captures cover English and French setup, inspection and invalid-FPS recovery at a 280-pixel sidebar width. The headless test also checks the actual Blender translation catalog for Resume and changed-fingerprint guidance. |

The standalone writer readback completed 1/1, the Release native CPU and CLI
rosters completed 469/469, and the Python suite reported 1,188 passed and one
skipped. These runs are regression checks for this delivery, separate from the
frozen M1 physical matrix.

The example uses one closed liquid box, a domain used to validate starting placement,
and a closed paddle box that translates at constant speed along world X. The solver
models the paddle as a slab spanning the domain; there are no finite paddle ends or
domain walls in this example. Bake publishes every frame at 24 fps, starting at zero.
The Points asset exposes size and material in the saved native modifier, with no
rebake needed. Cached velocity and stable ID remain available in Inspect. The
CPU option is available for development, separately from the RTX workflow result.

## Install and reproduce

Install the Windows x64 evaluation ZIP from disk in Blender Preferences → Get
Extensions. Use the pinned Blender, a Windows x64 Visual C++ 2022 runtime and the
pinned NVIDIA driver and GPU for Bake. No Python wheels, Conan packages, CUDA SDK
installation or network access are needed for this ZIP. The package carries the
bridge, solver and example; the project release-license decision remains open.

Copy `First Bake.blend` to an editable project directory and open it. In the 3D
viewport sidebar, choose FluidiTy. Use **Check Scene**, then **Preview** or **Bake**.
Use **Cancel** during Bake and **Resume** to continue. Choose **Attach Points**
after Bake (or **Attach Preview** for the short cache), then save the project.
Use **Inspect** on an integer committed frame. Use **Relink Cache** after moving
the cache independently. Keep 24 fps for playback. The Blender project and its
relative cache tree can be moved together without relinking.

Use the locked CUDA Release setup in the [native guide](../native/README.md) and the
locked Conan Alembic/Imath CMake files generated by the
[standalone writer setup](README.md#reproduce-the-windows-run). In the x64 MSVC/uv
shell, set these paths and run from the repository root:

```powershell
$blender = 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe'
$native = 'out/build/windows-release-cuda'
$abcConan = '<Alembic and Imath CMake package directory>'
cmake -S . -B $native -DFLUIDITY_BUILD_BLENDER_BRIDGE=ON "-DAlembic_DIR=$abcConan" "-DImath_DIR=$abcConan"
cmake --build $native --target fluidity_simulate fluidity_blender_bridge fluidity_simulate_fixture fluidity_liquid_acceptance_fixture
python tools/blender/package_m1_extension.py --blender $blender --solver "$native/fluidity_simulate.exe" --bridge "$native/fluidity_blender_bridge.exe" --output out/m1-b-delivery
python tools/blender/verify_m1_package.py --blender $blender --package out/m1-b-delivery/fluidity-0.1.0-windows-x64.zip --output out/m1-b-offline-receipt
```

For adapter-only verification, use a real committed drop cache through frame 3 and
run `fluidity_simulate_fixture scene EMPTY.flrs SOLVER cuda source`, then
`fluidity_simulate run --stage simulate --scene EMPTY.flrs --cache EMPTY_CACHE --backend cuda --frame-end 3`.
Run `verify_m1_points.py` inside the pinned Blender with
`-- BRIDGE DROP_CACHE EMPTY_CACHE EXISTING_OUTPUT_DIRECTORY`. The last argument
retains the imported scenes and negative controls. For the GUI evidence, open the
relinked project using the isolated installed profile and run
`review_m1_sidebar.py -- OUTPUT_DIRECTORY` via Blender's `--python` option.
