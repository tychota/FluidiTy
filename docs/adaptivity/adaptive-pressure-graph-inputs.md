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

The scheduler still uses uniform block/face storage, uniform transfer and uniform checkpoint records. The checked adaptive transfer evaluator covers bounded aligned two-level patches. A production caller still needs a general graded-leaf transfer and carrier mapping, certified cut and source samples from the trial geometry, and accepted adaptive topology and inventory storage through device publication and checkpoint. The owning graph assembler allocates when it freezes the graph. Integration into the existing allocation-free attempt path needs reserved staging.

The graph admits at most 4,096 pressure cells and 24,576 total subfaces, including sealed entries. The slot limit can bind before the leaf limit. No adaptive interrupted-replay or rejected-scheduler-trial receipt exists for this slice. No wave-scale runtime or memory claim is made.

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
