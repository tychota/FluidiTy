# Graded Simulate runtime

Verified source and replay harness commit `fc01a85986a34c77f5f2e24762e56d3a14268b76`, following baseline `3911b02`. The Windows builds use MSVC 19.44 with Ninja and the locked Conan toolchain. The CUDA Release build uses CUDA 13.4 and `120-real;120-virtual`. The RTX 5060 Ti executed SM120 device code.

## Runtime and accepted state

The scheduler admits an authored stationary open tank whose walls and horizontal free surface align with its graded leaves. It freezes the leaf topology and the global subface roster. P2G deposits component mass and signed normal momentum with normalized weights and centered affine offsets. G2P fits slopes from the eligible, same-side carrier support and uses normalized PIC when the affine fit loses rank or lacks mass. CPU and CUDA share the checked host transfer. CUDA runs the physical pressure solve and particle update on device. See the [M4 graph and ownership inputs](adaptive-pressure-graph-inputs.md).

A trial updates pressure, carriers, particles and material inventories in inactive storage. CUDA stages the candidate topology and inventories with the particle bank, fences the copy, and publishes the accepted device epoch through the scheduler transaction. The host publishes the accepted clock with the same records. A rejected trial keeps both published banks and the previous clock. The frame checkpoint stores the accepted adaptive records and reservation plan through the existing cache publication protocol. Resume reconstructs ownership from the authenticated checkpoint and resolved scene.

## Verification

The CPU rollback case compares the full encoded state before and after rejecting at acceptance, then accepts two shortened retries. The real CUDA case also reads back the published device banks and compares them through rejection and successful publication. The CPU suite checks solid upper-wall ownership and hydrostatic rest across a vertical coarse/fine interface within `5e-8 m/s`. CUDA memcheck reports zero errors for the rollback fixture.

The public CLI fixture runs eleven frames continuously. It kills a second process after the frame 10 publication event and starts a third process to resume to frame 11 without an initial-state argument. The killed process exited with code 1 on both backends. The test verifies checkpoint reservations, adaptive trace output, and matching final state. The checkpoint and frame digests match across CPU and CUDA.

| Accepted state | BLAKE3 |
| --- | --- |
| Frame 10 checkpoint | `a976826cd349fd76338fe906fa20e1cbfc56e8f28f2d0350d1d4fb2f16bd58f5` |
| Frame 11 continuous and resumed | `51ca741fb594df146493d153deb689b3045efe42cdd93cee33bed21dcc7b2579` |

Run from the repository root with the configured Debug CPU and Release CUDA builds and the corresponding tools on PATH.

```powershell
uv run --offline --project reference --locked --group native cmake --build out/build/runtime-debug --parallel 5
uv run --offline --project reference --locked --group native cmake --build out/build/runtime-cuda --parallel 5
./out/build/runtime-debug/fluidity_native_tests.exe '--gtest_filter=LiquidSubstep3D.*Graded*'
./out/build/runtime-cuda/fluidity_cuda_coarse_fine_tests.exe '--gtest_filter=CudaAdaptiveSimulate3D.RejectedGradedTrialPreservesDeviceEpoch'
uv run --offline --project reference --locked --group native python tools/native/tests/test_simulate_replay.py --executable out/build/runtime-debug/fluidity_simulate.exe --fixture out/build/runtime-debug/fluidity_simulate_fixture.exe --case cpu_graded_replay
uv run --offline --project reference --locked --group native python tools/native/tests/test_simulate_replay.py --executable out/build/runtime-cuda/fluidity_simulate.exe --fixture out/build/runtime-cuda/fluidity_simulate_fixture.exe --case cuda_graded_replay --cuda
$env:NV_COMPUTE_SANITIZER_LOCAL_CONNECTION_OVERRIDE = 'named-pipes'
compute-sanitizer --tool memcheck --require-cuda-init yes --error-exitcode 97 --launch-timeout 30 ./out/build/runtime-cuda/fluidity_cuda_coarse_fine_tests.exe '--gtest_filter=CudaAdaptiveSimulate3D.RejectedGradedTrialPreservesDeviceEpoch'
```

Local receipts are `out/graded-rollback-cpu.log`, `out/graded-rollback-cuda.log`, `out/graded-replay-cpu.log`, `out/graded-replay-cuda.log`, and `out/graded-runtime-memcheck.log`.

The complete CPU suite passed 677 CTests. The registered CPU and CUDA replay tests exercise the same hard interruption. Full CPU and serial CUDA suite receipts are in `out/graded-runtime-full-cpu-final.log` and `out/graded-runtime-full-cuda-final.log`. The CUDA suite initially used the earlier hydrostatic assertion tolerance. The rebuilt assertion passed on retest in `out/graded-hydrostatic-cuda-build.log`.

## Independent controls

The independent 3D fixture and its 67 Python controls passed. The manufactured pressure benchmark met the same pressure and divergence gates on both layouts. The measured maximum divergence was `5.96e-6 s^-1` uniform and `8.90e-6 s^-1` adaptive against the common `2e-5 s^-1` limit. Debug CPU median projection time was 4.7751 ms uniform and 1.4664 ms adaptive. This benchmark measures the pressure solve on a sealed cube.

```powershell
uv run --offline --project reference --locked --group native python tools/adaptivity/qualify_coarse_fine_3d.py --output out/adaptivity-m3/qualification.json
uv run --offline --project reference --locked --group native pytest reference/tests/test_coarse_fine_basis.py tools/tests/test_qualify_coarse_fine_basis.py tools/tests/test_qualify_coarse_fine_3d.py -q
./out/build/runtime-debug/fluidity_m3_matched_pressure.exe
```

Local receipts are `out/adaptivity-m3/qualification.json`, `out/graded-independent-controls.log`, and `out/graded-matched-pressure.json`.

## M3 promotion boundary

The runtime keeps the authored topology frozen through the run. It requires material owned by particles and retries when the proposed inventory exceeds the destination leaf volume. The integration still needs live refine/coarsen, conservative family remap and split/merge, physical moving and cut-boundary inputs, and production band handoff with authoritative Eulerian occupancy. The manufactured remap, band and F3/F4 receipts cover their independent contracts. Production Mesh still needs M2 accepted surface and motion data for false inner surfaces, cavity lineage and moving correspondence.

The bounded graph limits are 4,096 leaves and 24,576 subfaces. Host transfer visits the face roster for particle support, and graph staging and transient CUDA solve storage are outside the persistent arena budget. The fixture provides neither a matched-error wave comparison nor a simultaneous peak-memory measurement. Promotion requires the 240-frame wave at equal error in at most 10,800 s, with GPU peak at most `min(0.70 × total, 0.80 × free_at_job_start)`. The temporary uniform-sparse narrow-band fallback has a 21,600 s limit under the same cap and leaves adaptivity open.
