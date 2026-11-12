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
- A constant-viscosity tank uses an implicit coupled solve. The normal rows
  are the signed open-area flux derivatives divided by leaf volume, so
  their trace is the same divergence used in pressure. The shear rows
  symmetrize gradients from local affine fits on actual staggered anchors.
  Stationary sealed normal velocities are lifted as zero. The CUDA inverse
  uses the existing reserved pressure ranges on the arena stream.
- Planar exterior capillarity uses the existing liquid-air timestep
  formula for each exposed leaf, with that leaf's density and width. The
  minimum controls rejection. The authored plane has zero curvature.

Uniform FLST v1 retains the published cavity field 13. The draft now uses
the integrated M3 FLST v2 layout on main, with graded records at field 14.
Regenerate the prerelease fixtures from the original M3 branch, which
used field 13, and from the initial M4 draft, which used version 3.

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

Source commit `bc73967`, based on main `b56e06e`, passed all 811 Windows
Debug CPU tests, 23 focused graded and coupled-pressure cases, three CPU
CLI checks, and two real-CUDA CLI checks. The RTX 5060 Ti (driver 610.88,
CUDA 13.4, SM120) passed all seven graded adaptive cases and four
focused uniform viscosity regressions. The viscous retry/readback test
recorded unchanged device allocation and free counters and reserved
device and pinned bytes. Compute Sanitizer reported zero errors on that
case. The earlier runtime source `01356af` passed the 67-case CUDA
operator suite before the signed-jump and graded tensor follow-ups.

Each direct replay kills the process after publishing frame 10 and starts
a third process to resume to frame 11. The frame-ten checkpoint and final
state match the uninterrupted run byte for byte on each backend.

| Case | Backend | Frame 10 checkpoint BLAKE3 | Frame 11 state BLAKE3 |
| --- | --- | --- | --- |
| EOS | CPU | `b342fe7a960b9ffbf3e02aef2939df1f8a716a0207f18f751c85afe29080e347` | `b667831f0d877efd94f2094e9393e80743cb3ebf36ec4744f09befd2aa0d63f4` |
| EOS | CUDA | `fb283d9ebb3861d276a293472f51f99f629ffe4eabc654ae58d4f6dbe01ce125` | `58796c0187b5490a1b0ea9ac65c3a0b95bc7f4fc41cb9614dd4c01119e5fb648` |
| Miscible | CPU and CUDA | `97ad25246808ea1adb255e9fad59196d803cca062a77c0d08e85d699bbd24f53` | `3eae555010ca81525c16318f3ea4db2f800bc59229d78a5df044e00b597d6ae4` |
| Viscous | CPU and CUDA | `2dd95839509f248e3db2b0004e2955e7a066a2651b8bd2a436c2730b9bcb0645` | `7a55f31798622d93c1db1ea1b80450e92e2df6c7db24facba424b51d53754b34` |

The local receipts are `out/m4-graded-tensor-cpu.xml`,
`out/m4-graded-tensor-cpu-focused.log`, `out/m4-graded-tensor-cpu-replay.log`,
`out/m4-graded-tensor-cuda-replay.log`, `out/m4-graded-tensor-cuda-rollback.log`,
`out/m4-graded-tensor-cuda-memcheck.log`, `out/m4-graded-tensor-cuda-uniform.log`,
`out/m4-graded-tensor-cli-cpu.log` and `out/m4-graded-tensor-cli-cuda.log`.

## Coarse/fine interface operator controls

The open interior graph slot accepts a signed equilibrium pressure jump in
Pa. The signed jump is applied once through the existing pressure assembly
and velocity correction. Sealed and exterior slots reject it. The tests
exercise both signs and verify unchanged interior and outlet fluxes.

The separate coupled traction fixture uses manufactured normal rows at a
2:1 viscosity ratio. The CPU and SM120 velocity solves recover −1.5 Pa
normal traction on both sides of one owned cut subface. The three focused
CPU interface tests and all six CUDA adaptive tests passed. Compute
Sanitizer reported zero errors for the two CUDA interface controls.

```powershell
uv run --project reference --locked --group native ctest --test-dir out/build/graded-cpu-final -R 'AdaptiveM4Pressure3D|AdaptiveInterfacePressureJump' --output-on-failure --no-tests=error
uv run --project reference --locked --group native ctest --test-dir out/build/graded-cuda-final -R 'CudaAdaptiveSimulate3D' --output-on-failure --no-tests=error
compute-sanitizer --tool memcheck --require-cuda-init yes --error-exitcode 97 --launch-timeout 30 ./out/build/graded-cuda-final/fluidity_cuda_coarse_fine_tests.exe '--gtest_filter=CudaAdaptiveSimulate3D.CoupledGradedNormalTractionMatchesManufacturedReference:CudaAdaptiveSimulate3D.AdaptiveInterfacePressureJumpPreservesOwnedFlux'
```

The interface receipts are `out/m4-graded-traction-cpu.log`,
`out/m4-graded-traction-cuda.log` and `out/m4-graded-interface-memcheck.log`.

## Follow-up qualification

The merge at `de03198` passed 852 Debug CPU CTests, 28 focused CPU
cases, nine SM120 adaptive cases, four real-device viscosity regressions,
CPU and CUDA checkpoint replay, and the pinned Lean and independent
checker. Compute Sanitizer reported zero errors on graded viscous rollback.

The constituent-viscosity follow-up freezes the existing current-volume
mixture rule per leaf for the duration of an attempt. On `b3d0f53`,
the combined main tree passed 27 focused CPU cases and 16 SM120 adaptive
and CLI cases, including public graded EffectsSource capture and family
replay. The constituent-viscosity fixture reproduces frame 11 after a fresh
process resumes from frame 10. The two backends produced identical
checkpoint and final state digests.

Live family tests cover accepted constant-viscosity refinement and
coarsening with late rollback. The driven EOS case refines, refuses
coarsening when the fine exterior flux sum cannot be exactly represented
on a coarse face, and then resumes on the retained refined topology.
The miscible family case reuses conservative remap and checks extensive
constituent volumes against the actual particle rebin in every leaf. The
uneven sibling control preserves the parent totals and rejects before
publication because the child inventories need certified resampling.

~~~powershell
ctest --test-dir <qualified-build> -R '^Adaptive(Family|Band)Replay' -V --output-on-failure --no-tests=error
~~~

## Remaining M4 gates

The full graded planar-extension and Couette matrix needs independently
fixed traction references at viscosity ratios 1, 10 and 1000, two
resolutions and smaller timesteps. Apply the spec's 5% normalized RMS
threshold, fixture maximum and absolute floor. The current graded tensor
path covers stationary geometry, including constant and miscible
constituent-dependent viscosity. Moving and cut geometry need the
physical wall and interface quadratures. The prepared graph and host
orchestration still allocate.

Graded trapped air needs resolved gas occupancy, liquid–gas apertures on
the canonical subfaces, and checkpointed component lineage. Graded
immiscible transport and curved capillary pressure need one authoritative
phase pair per shared interface, phase-aware conservative transport and
persistent two-sided Mesh composition. The M3 stationary graded Mesh
receipt qualifies the single-phase carrier and ID path.

The family path admits exact particle rebinning. General EOS coarsening
needs conservative storage for exterior flux sums that the current coarse
face cannot represent. General constituent refinement needs certified
resampling or material-aware prolongation. Extended-liquid Eulerian
handoff, automatic topology policy and conservative cross-owner exchange
remain open.

The fresh M2 column-seed wave still refuses Mesh frame two. Reproduction on
`de03198` used 0.125 m Mesh spacing and 0.16 m sigma. Simulate published
frame two. Mesh stopped after frame one because its raw scaffold encloses
1.32813 m³ against 1.76750 m³ of accepted material. The fixed-cache
continuation is a separate control. The full fresh reference wave and
interrupted replay remain open.
