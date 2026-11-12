# Adaptive pressure graph inputs for M4

The bounded assembler in `native/src/mac_pressure_3d_system.h` takes an immutable `AdaptiveMacTopology3D` and step duration. It reconstructs the global slots from `BuildAdaptiveMacFaceOwnership3D`. M4 should use the same topology and ownership for strain, traction, pressure, and constituent tests.

| Input | Order and meaning |
| --- | --- |
| Leaf keys | Finest-lattice signed lower corner and dyadic width exponent. Canonical leaf order is signed `(z,y,x)`. |
| Pressure cells | One row per canonical leaf. Use the leaf lower corner as `cell_index`, positive liquid volume bounded by the geometric volume, source divergence in s⁻¹, bulk modulus in Pa, and previous accepted pressure in Pa. |
| Pressure subfaces | One physical sample per global slot in `(axis,z,y,x,width exponent)` order, including sealed faces. The graph derives incidence, full area, and two-cell center distance from geometry. |
| Open and blocked parts | Supply physical aperture in m², face mobility in m³/kg, predicted positive-normal velocity in m/s, and signed integrated wall flux in m³/s. The wall flux is allowed only on the blocked part. |
| Exterior faces | The geometry identifies the single incident leaf and its orientation. An open exterior needs finite exterior pressure and positive distance to that pressure. A sealed exterior has zero aperture and zero pressure distance. |

Each oriented subface contributes outward flux with sign +1 at its tail and −1 at its head. The matrix and velocity correction use the same aperture, mobility, and distance. Cell source flux is liquid volume times source divergence. M4 traction should use the geometric face identity, endpoint incidence, and physical aperture from this same frozen roster. For constituent conservation, start from extensive per-leaf inventories in canonical cell order and use one signed constituent flux per global slot. Sum sibling inventories and signed outer subfaces during restriction. An internal sibling flux cancels across the two incident cells. Those conventions provide test inputs for the M4 extension. The pressure fixture has no constituent field.

## Qualified two-leaf control

The shared test fixture uses finest width 0.5 m and leaves `((0,0,0),1)` and `((2,0,0),0)`. One two-sided and thirteen exterior subfaces fill the ownership roster. The two-sided face has full area 0.25 m², aperture 0.125 m², and +0.0625 m³/s prescribed wall flux. The coarse source adds 0.09375 m³/s. The fine outlet has aperture 0.25 m², pressure 1 Pa, distance 0.25 m, and mobility 1 m³/kg. All other exterior faces are sealed. With a 1 s step, the analytic coarse and fine pressures are 1.28125 Pa and 1.09375 Pa. The interior fluid velocity is +0.25 m/s and the exterior fluid velocity is +0.375 m/s. The CPU control also closes the interior aperture and verifies a gauge on the isolated coarse component with compatible signed wall and source flux.

## Runtime boundary

Simulate accepts authored, stationary, conforming planar tanks with particle-owned liquid. It freezes the graded leaf topology and the canonical face roster on admission. Full geometric apertures connect interior leaves. Stationary wall faces have zero aperture and zero wall flux. Open top faces use ambient pressure at half the incident leaf width. The predictor uses the signed body-potential difference between incident leaf centers divided by their normal spacing, including at coarse/fine interfaces. Source divergence is zero. Use the accepted state to reconstruct keys, incidence and carrier slots, and the resolved scene and trial-start state to reconstruct physical apertures, mobility, predicted velocities, boundary values and previous pressures. The committed face velocity is the corrected result.

The cut and source fixtures below qualify inputs independently of the current planar scheduler. M4 should supply measured cut apertures and signed wall/source flux for those cases. Carry constituent inventories and signed fluxes on the same frozen roster. The current scheduler commits particle-owned material, with no Eulerian constituent transport.

The graph admits at most 4,096 pressure cells and 24,576 total subfaces, including sealed entries. The slot limit can bind before the leaf limit. The pressure graph and CUDA solve still allocate per attempt outside persistent arena reservations. See the [graded Simulate runtime receipt](graded-simulate-runtime.md) for rollback, replay and wave-scale limits.

## Verification receipt

The tested source is `0cf1df9cb83fe3bed2dd3b431b53a57fc25d083d` on base `549fb3f844841b79d14b7beb9ecfb6db9727c976`. The Windows runs used MSVC 19.44, CUDA 13.4, and an NVIDIA GeForce RTX 5060 Ti (driver 610.88). The device test reported compiled architecture 1200. The open cut-subface case measured a maximum divergence of `9.53674e-7 s⁻¹` against a `2e-5 s⁻¹` limit.

The local build directories were configured with Ninja and the locked Conan toolchain. `runtime-debug` used the Windows Debug CPU profile. `runtime-cuda` used the Windows Release CUDA profile, the shared CUDA runtime, and `120-real;120-virtual`. From the worktree root with the qualified MSVC and CUDA tools on PATH:

```powershell
uv run --offline --project reference --locked --group native cmake --build out/build/runtime-debug --parallel 8
uv run --offline --project reference --locked --group native ctest --test-dir out/build/runtime-debug --output-on-failure -R '^MacPressure' --no-tests=error
uv run --offline --project reference --locked --group native ctest --test-dir out/build/runtime-debug --output-on-failure -j 8
uv run --offline --project reference --locked --group native cmake --build out/build/runtime-cuda --target fluidity_cuda_operator_tests --parallel 6
uv run --offline --project reference --locked --group native ctest --test-dir out/build/runtime-cuda --output-on-failure -V -R '^CudaMacPressure3D' --no-tests=error
$env:NV_COMPUTE_SANITIZER_LOCAL_CONNECTION_OVERRIDE = 'named-pipes'
compute-sanitizer --tool memcheck --require-cuda-init yes --error-exitcode 97 --launch-timeout 30 ./out/build/runtime-cuda/fluidity_cuda_operator_tests.exe '--gtest_filter=CudaMacPressure3D.ExplicitGraphMatchesEdgeAndCornerReferencePressure'
```

The complete native CPU suite passed 665 cases, including all 43 pressure cases. The CUDA run passed four cases. The adaptive GPU case reported zero Compute Sanitizer errors. Raw receipts are `out/cpu-pressure.log`, `out/cpu-final-tests.log`, `out/cuda-final-tests.log`, and `out/cuda-final-memcheck.log`.

The full CUDA operator executable also passed all 42 tests:

```powershell
& ./out/build/runtime-cuda/fluidity_cuda_operator_tests.exe
```

Its raw receipt is `out/cuda-operator-suite.log`. Executable SHA-256 values are `a7eb01a41675faf00074288c7aa1f963f52914c2d839cc8412c3c1b1d7776ce8` for the CPU native tests and `e8094a2a753bb90cfc5313bb8485787e62ed8ff2a33e428b195aa4c796705f26` for the CUDA operator tests.

## Transfer and checkpoint continuation

The tested continuation source is `4ce045003a8b8f6cbdc27b5a84374c6a3e5e40a5`. The bounded two-level transfer stencil can bind its rows to the canonical global ownership of a graded topology. A geometric face must match in position, axis and width. A subdivided coarse face is unavailable at its former width. The host PIC path excludes missing, ineligible and different-side carriers, then uses the existing normalized fallback and preserves the output on refusal.

Adaptive checkpoints use schema version 2 with canonical leaf keys, authoritative occupancy, extensive inventories, previous pressures and a dense canonical subface carrier roster. Host transaction versions keep these records and the accepted clock together. Wire decoding validates geometry and carrier ownership before allocating records, and both initial and retry reservations are capped at 4,096 cells and 24,576 faces. Uniform checkpoints retain v1 encoding. The validation scans leaf pairs with early separation along the canonical z axis and a bounded stack array for face coverage.

The cut-face pressure fixture runs a rejected projection, accepts a shortened retry, encodes a completed frame, and restores a new host transaction for continuation. Continuous and restored results are byte-identical for each backend. CPU and RTX 5060 Ti SM120 also produced the same final BLAKE3 digest, `042769715eda6dec3fb179d023d40dd65460c59ad13f8953bd176ade6534582d`. The CUDA run performs real pressure solves. Both fixtures use the same authored topology and physical aperture data, so they qualify the pressure stage and host checkpoint boundary. That historical receipt covers the pressure stage and host checkpoint only.

With the same configured builds as above:

```powershell
uv run --offline --project reference --locked --group native ctest --test-dir out/build/runtime-debug --output-on-failure -V -R '^(LiquidCache3D|LiquidState3D|LiquidFrameTransaction3D|LiquidCacheSchema)|UniformSchedulerRefusesAdaptiveCheckpointState|CheckpointReplaysAcceptedAdaptivePressure|AdaptiveMacTopology3D.BindsTransferRows' --no-tests=error
uv run --offline --project reference --locked --group native ctest --test-dir out/build/runtime-debug --output-on-failure -j 8
uv run --offline --project reference --locked --group native ctest --test-dir out/build/runtime-cuda --output-on-failure -V -R '^Cuda(CoarseFineTransfer3D|MacPressure3D)' --no-tests=error
$env:NV_COMPUTE_SANITIZER_LOCAL_CONNECTION_OVERRIDE = 'named-pipes'
compute-sanitizer --tool memcheck --require-cuda-init yes --error-exitcode 97 --launch-timeout 30 ./out/build/runtime-cuda/fluidity_cuda_operator_tests.exe '--gtest_filter=CudaMacPressure3D.ExplicitGraphMatchesEdgeAndCornerReferencePressure'
```

All 670 native CPU CTests, 68 focused CPU CTests, seven real CUDA pressure and transfer CTests, and 42 CUDA operator cases passed. Compute Sanitizer reported zero errors on the checkpoint pressure case. Receipts are in `out/adaptive-codec-green.log`, `out/adaptive-runtime-final-cpu.log`, `out/adaptive-runtime-final-cuda.log`, and `out/adaptive-runtime-final-memcheck.log`.
