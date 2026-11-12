# Fixed-topology graded M4 runtime

This slice starts from M3 `fc01a85` and the uniform M4 already on main. The
authored fixture has one 1 m leaf and four 0.5 m leaves in a stationary,
leaf-aligned tank. The scheduler uses the same canonical face ownership for
pressure, the volume change, and the accepted carrier bank.

- Constant compliance and Hencky/Tait EOS use the existing graded pressure
  solve. EOS pressure history is backtraced from the accepted carriers with
  an area-weighted one-ring upwind interpolant. A departure whose donor
  weights exceed one requests a smaller step. The accepted pressure must
  lie in the EOS domain. Particle volume ratio uses the divergence of the
  corrected owned flux and the update backend's existing exponential law.
- One explicit miscible phase can carry multiple substances. Particle-owned
  extensive volumes are reduced into canonical leaf inventories by stable
  substance ID, and the additive-volume mixture sets the face mobility.
  The cell inventory is included in the CUDA bank, semantic comparison and
  checkpoint. Particle composition metadata is retained on the host.
- Planar exterior capillarity uses the existing liquid-air timestep
  formula for each exposed leaf, with that leaf's density and width. The
  minimum controls rejection. The authored plane has zero curvature.

Uniform FLST v1 retains the published cavity field 13. Integrated graded FLST
v3 uses field 14. Regenerate prerelease M3 v2 graded checkpoints from their
fixture inputs before using this build. The public CLI tests start from a
v3 seed and resume from a v3 frame-ten checkpoint in a new process.

## Reproduce

Use clean configured Windows Debug CPU and Windows Release CUDA builds with
the locked tools, qualified toolchain receipt and fixture bundle. The
qualified build directories below use `-B out/build/graded-cpu-final` and
`-B out/build/graded-cuda-final` with the corresponding presets.

```powershell
uv run --project reference --locked --group native cmake --build out/build/graded-cpu-final -j 8
uv run --project reference --locked --group native ctest --test-dir out/build/graded-cpu-final -L '^cpu$' -j 8 --output-on-failure --no-tests=error
uv run --project reference --locked --group native python tools/native/tests/test_simulate_replay.py --executable out/build/graded-cpu-final/fluidity_simulate.exe --fixture out/build/graded-cpu-final/fluidity_simulate_fixture.exe --case cpu_graded_replay
uv run --project reference --locked --group native cmake --build out/build/graded-cuda-final --target fluidity_native_tests fluidity_simulate fluidity_simulate_fixture fluidity_cuda_coarse_fine_tests fluidity_cuda_operator_tests -j 6
& ./out/build/graded-cuda-final/fluidity_cuda_coarse_fine_tests.exe '--gtest_filter=CudaAdaptiveSimulate3D.*'
uv run --project reference --locked --group native python tools/native/tests/test_simulate_replay.py --executable out/build/graded-cuda-final/fluidity_simulate.exe --fixture out/build/graded-cuda-final/fluidity_simulate_fixture.exe --case cuda_graded_replay --cuda
$env:NV_COMPUTE_SANITIZER_LOCAL_CONNECTION_OVERRIDE = 'named-pipes'
compute-sanitizer --tool memcheck --require-cuda-init yes --error-exitcode 97 --launch-timeout 30 ./out/build/graded-cuda-final/fluidity_cuda_coarse_fine_tests.exe '--gtest_filter=CudaAdaptiveSimulate3D.*'
& ./out/build/graded-cuda-final/fluidity_cuda_operator_tests.exe
```

## Qualification receipt

Source commit `01356af` passed 51 focused CPU tests. The RTX 5060 Ti
(driver 610.88, CUDA 13.4, SM120) passed all four graded epoch tests.
Compute Sanitizer reported zero errors across those four tests.

Each replay kills the process after publishing frame 10 and starts a
third process to resume to frame 11. The frame-ten checkpoint and final
state match the uninterrupted run byte for byte on each backend.

| Case | Backend | Frame 10 checkpoint BLAKE3 | Frame 11 state BLAKE3 |
| --- | --- | --- | --- |
| EOS | CPU | `1455a734060c0284ba9f2db6f87ab19ee879e26731e64f62523725ea4f71b60d` | `a365c90e13d6d8bc944d72c538e8d7ea6b9b9e624301f486beee21e82c9d127b` |
| EOS | CUDA | `6016c2566382b54504f0fce3a4e17791a9e3f09443f7c594c0a45453b54d26cb` | `5d82cccf7d4818ed1b92e714ebe137b958064220fcccd035d2a1519c6f99247d` |
| Miscible | CPU and CUDA | `e2960a2310d656af3a6c7fa0b63f4cbc499210fb55a450dc5ed6f14fc67e0648` | `9d701b00555e4e8423015ddc4af3c873d64cd878121e64f5f177336032ca15c3` |

The local replay and GPU receipts are `out/m4-graded-cpu-replay.log`,
`out/m4-graded-cuda-replay.log`, `out/m4-graded-cuda-rollback.log` and
`out/m4-graded-cuda-memcheck.log`.

## Remaining M4 gates

The graded runtime still needs tensor strain quadrature and a coupled
viscous solve with adaptive planar-extension and Couette qualification,
a supported immiscible interface with conservative transport and curved
pressure jump, and trapped-air geometry and pressure. Adaptive Mesh needs
authoritative occupancy, shared-interface geometry, persistent IDs and
composition output. These can be developed on fixed authored topology.
Live refine/coarsen, family remap, moving and cut geometry, band handoff,
and the full fixed-cache wave remain integration and qualification work.
