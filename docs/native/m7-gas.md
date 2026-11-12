# M7 low-Mach gas qualification

The local branch starts at upstream main `db4636df15c90e92f109d6b1a7ded65dddad4d65`. The gas runtime reuses the owned MAC pressure topology and projection, APIC transfer operators, the public simulate protocol, and the native immutable file publication primitives.

## Fixed-domain evidence

| Gate | Result |
| --- | --- |
| Open heating | Measured and analytic displacement were both 0.000070494519051143774 m³ in the unit-depth reference. Required relative tolerance is 25%. |
| Closed heating and independent components | The heated pair rose 400 Pa, and the disconnected unheated pair rose 0 Pa. Required pressure-response tolerance is 1%. |
| Conservative transport | Reference and native tests balance fuel, soot, carrier mass, and enthalpy against sources and signed boundary flux as applicable. The coupled test accounts for pressure work and uses the measured projected flux. |
| APIC | The native tests cover affine transfer, a uniform draft through the halo, sealed/open marker boundaries, and replacement of an advected-out marker. |
| Checkpoint and replay | CPU and real CUDA process tests compare continuous and resumed checkpoints byte-for-byte, verify read-only replay, reject tampering, and preserve prior frames when a heat pulse rejects a later step. |
| Blender | Blender 5.2.3 LTS build `e13b728c6880` uses the add-on Check Scene, Bake, Inspect Gas Fields, and Attach Gas Volume controls. The CPU scene publishes 13 frames and five native grids. Frame 12 spans about 293–777 K, with 46 active flame voxels. A fresh-process Cycles render verifies a flame color change while all 30 tracked files retain their hashes and timestamps. |

The authored preset uses fixed boxes, a stationary floor, zero conductive flux, and a uniform acceleration derived from Blender Wind. A native source can supply a prescribed net conductive heat rate. The CPU or CUDA backend performs the pressure projection. Scalar transport and APIC run on the host for both backends. Marker weight is numerical and does not enter the carrier mass or pressure mobility. Optional IPIC and flow maps remain gated.

## Reproduction

From the worktree in a Visual Studio developer PowerShell, use the locked build and environment setup in [Native build and validation](README.md). On the qualification host the standard wrapper required uv 0.13.7, while the shell exposed 0.13.6. The builds used the existing locked Conan toolchains and Python environment. The Release CUDA executable ran on an RTX 5060 Ti with driver 610.88. Run GPU CTest serially on this host. A parallel run exhausted resources in six liquid CUDA cases, and the six passed in a serial retry.

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
python -m pytest -q reference/tests
ctest --test-dir out/build/m7-debug-cpu -j 3 --output-on-failure
ctest --test-dir out/build/m7-release-cuda -j 1 --output-on-failure
```

Use the solver from the selected build. The script requires a fresh output path. The example is saved with project-relative cache and Volume paths. Keep the bake and Volume folders next to the blend file.

```powershell
& $blender -b --factory-startup --python-exit-code 1 --python tools/blender/verify_m7_fire_in_a_draft.py -- --solver out/build/m7-debug-cpu/fluidity.exe --output out/m7-fire-cpu --backend cpu
& $blender -b --factory-startup --python-exit-code 1 --python tools/blender/verify_m7_fire_in_a_draft.py -- --solver out/build/m7-release-cuda/fluidity.exe --output out/m7-fire-cuda --backend cuda
& $blender -b --factory-startup 'out/m7-fire-cpu/Fire in a draft.blend' --python-exit-code 1 --python tools/blender/verify_m7_gas_render.py -- out/m7-fire-cpu/render
```

With the extension installed, select a static domain box, a static emitter box with keyframed `Fuel Rate` and `Heat Rate` custom properties, and an optional Blender Wind field. At 24 fps with frame zero as the start, use Check Scene and Bake, then Inspect Gas Fields or Attach Gas Volume. Render styling is stored in Blender's Volume and material datablocks.
