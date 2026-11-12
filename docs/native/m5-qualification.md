# M5 qualification on Windows

Local branch `codex/m5-granular-snow`, based on verified upstream
`549fb3f844841b79d14b7beb9ecfb6db9727c976`. This report covers the bounded
M5 implementations against §§2.3, 5.2, 8.1–8.2 and 15.2 of the
[design](../superpowers/specs/2026-11-07-fluidity-design.md).

| Gate | Status | Evidence |
|---|---|---|
| Granular constitutive invariants | Pass | Elastic compression, translated tensile apex, compressed shear return to the cone at fixed trace, plastic volume transfer, and invalid-state output preservation. |
| Snow constitutive invariants | Pass | Compression and tension clamps, reconstructed total volume, hardened corotational stress, repeated compaction, and invalid-state output preservation. |
| Full 3D material state and wave bound | Pass for the tested domain | Rotated and repeated stretches, reflection and singular rejection, isotropic and hardened wave oracles, and rejection of axial and principal-plane loss of ellipticity. Principal-plane checks do not certify ellipticity in every fully oblique direction. The Hessian row bound covers continuous elastic wave speed. The tested discrete acoustic factor is 0.25 for the bounded scenes. |
| Q-SOLID uncut quadrature | Pass | Analytic affine gradient gives 87.5 W particle power and −87.5 W grid power, with 3.25e-16 scaled residual on both backends versus 1e-6 allowed. This measures uncut stress quadrature and the matching gather derivative. |
| Q-SAND 30° pile | Reference case passes; finer-grid check open | The reference case settles at 30.161149°, inside the required 30–35° interval. The predeclared finer case settles at 27.803822° on both backends. The measured resolution sensitivity is 2.357327°. The finer case remains a failing qualification command. |
| Dry snow compaction | Pass | Two resolutions pass a compress–unload cycle. Mean log plastic Jacobian stays below −0.34 after unload. |
| Dry step acceptance | Pass for the tested domain | Motion and acoustic bounds are checked during the step. Tests reject an inadmissible initial snow state and a move out of allocated support, preserving the particle record. |
| Isolated distinct-body contact | Pass | Closing, separating, sticking, sliding and invalid pairs. Unequal masses give 6 N·s normal and −1.5 N·s sliding impulse on the first body. Momentum is conserved within the fixture tolerance, and pair kinetic energy decreases. |
| Isolated face body capacity | Pass | Four sorted distinct body records, accumulation for an existing body at capacity, fifth-body rejection, and range-failure rollback on CPU and CUDA. The host error is ResourceExhausted / E-CONTACT-BODY-CAPACITY. |
| Production distinct-body water coupling | Open | The production liquid scheduler and resolved scene still lack the solid integration described below. |
| Authored Blender snow–water scene | Gated | The scene requires a real mixed solver, compatible checkpoints and distinct published material channels. |

The material and pair contact fixtures run the same tests in host code and
in an actual SM120 kernel. Scalar analytic comparisons use absolute error
below 1e-11 in their fixture units. The dry CUDA path uses device 0 for
quantized mass and momentum deposition and gathering, with host stress,
constitutive updates, face evolution, swept contact and acceptance. The
scene comparison therefore qualifies these transfers in a hybrid dry step.
It does not qualify a fully device-resident solid scheduler.

The bounded dry solver uses complete quadratic MAC supports, APIC deposits,
PIC advection, uncut reference-volume stress and an analytic halfspace
sweep. Sand retains the incoming normal speed for swept floor friction.
Snow uses grid normal constraints to transfer load through the platen. The
solver uses a force acceleration denominator of at least 1e-4 times the
smallest particle mass. On low-mass faces this reduces the actual stress
impulse. The Q-SOLID receipt covers the unregularized quadrature. The accepted-step fixture accounts for vertical momentum with an omitted
stress impulse bound. A full energy balance remains open.

## Scene measurements

The fixture starts with 0.05 m particle spacing, 300 kg/m³ reference density,
and a 0.7 floor friction coefficient. Sand uses exactly 30° of material
friction and is sampled at 0.70 s and 0.75 s. Its radial envelope estimate
uses atan2(maximum height, maximum horizontal radius from release center).
Angle changes over the final 0.05 s must be at most 2°. The reference
pile also requires radius in (0.10, 0.16) m and kinetic energy below 0.01 J.
The refined case remains open under this same protocol. A floor coefficient
of 1.0 was also tried and changed its angle from 27.803822° to 27.810506°.

Snow advances to 0.45 s with a platen reversal at 0.23 s and removal at
0.35 s. The step splits at those times. The final height must be in
(0.11, 0.15) m and mean log plastic Jacobian in (−0.48, −0.34). The fixture
also checks floor nonpenetration, exact particle mass/reference volume
retention, and finite plastic state. Sand and snow use the same wall and
motion acceptance on both backends.

| Scene | h (m) | nominal dt (s) | height (m) | radius (m) | mean log Jpl | kinetic energy (J) | angle and drift |
|---|---:|---:|---:|---:|---:|---:|---|
| Sand reference | 0.075 | 0.001875 | 0.075670 | 0.130217 | 0.054520 | 0.002264 | 30.161149°, 0.922262° |
| Sand refined | 0.0625 | 0.0015625 | 0.077089 | 0.146188 | 0.089935 | 0.001680 | 27.803822°, 0.677401° |
| Snow | 0.100 | 0.002500 | 0.134685 | 0.070617 | −0.393920 | 0.006660 | |
| Snow | 0.075 | 0.001875 | 0.129934 | 0.070782 | −0.415331 | 0.005504 | |

Native Debug and CUDA Release report the same values at the printed precision.
This is a differential measurement, with no bitwise replay claim.

### Sand timestep sensitivity

Halving only the nominal timestep, with the physical inputs and measurement
times fixed, produced the following results on both backends.

| h (m) | nominal dt (s) | angle | final 0.05 s drift | Q-SAND window |
|---|---:|---:|---:|---|
| 0.075 | 0.0009375 | 37.248475° | 1.036500° | Above |
| 0.0625 | 0.00078125 | 32.699841° | 0.782335° | Within |

The measured shifts are +7.087326° and +4.896019° respectively. No single
nominal timestep factor in this pair of runs qualified both grid widths.
The fixture exposes this as a separate command with exit 1 when either
case misses the existing window. Next isolate the per-step PIC velocity
replacement and the elastic update from the spatial quadrature, before
changing the material parameters or angle estimator.

### Stored velocity experiment

The optional dry-step blend retains 0.594 of the particle velocity
residual against the pre-force grid at energy setting 0.6. The
manufactured test uses opposing velocities with zero grid increment.
It verifies that the residual survives in storage while the PIC advection
and positions stay fixed. Low-mass supports retain PIC storage.

| h (m) | nominal dt (s) | PIC angle | stored blend angle |
|---|---:|---:|---:|
| 0.075 | 0.001875 | 30.161149° | 26.571091° |
| 0.0625 | 0.0015625 | 27.803822° | 25.525547° |
| 0.075 | 0.0009375 | 37.248475° | 32.408302° |
| 0.0625 | 0.00078125 | 32.699841° | 29.525806° |

CPU and CUDA matched at the printed precision. The option changes the
pile response but leaves 5.837211° and 4.000259° timestep shifts at
the respective widths. The existing PIC default stays in place.
Advection and deformation still use the PIC field and its derivative
respectively. Surface separation and an independent deformation
integration experiment remain open.

### Deformation and contact-affine controls

Run these controls independently from commit 6d31b68. Use the ordinary
dry fixture for the reference nominal step, plus the refined and half-step
commands below. Keep the stored-velocity blend at zero.

- Deformation control. In TrialDeformation replace (I + dt L) F with
  (I + dt L + 0.5 dt² L²) F, using the same frozen gathered L, then apply the
  same constitutive return.
- Wall-affine control. Keep the original deformation update. In the final
  dry particle publication retain the fitted x and z slope rows at a floor
  contact and zero the y row. The pile has no platen. This only tests
  sensitivity to the affine reset and does not certify an affine Coulomb
  boundary law.

| h (m) | nominal dt (s) | Original angle | Quadratic update | Retain tangent rows |
|---|---:|---:|---:|---:|
| 0.075 | 0.001875 | 30.161149° | 30.185224° | 23.649397° |
| 0.0625 | 0.0015625 | 27.803822° | 27.832553° | 23.040349° |
| 0.075 | 0.0009375 | 37.248475° | 37.265489° | 27.827683° |
| 0.0625 | 0.00078125 | 32.699841° | 32.722061° | 25.848644° |

Native and CUDA angles matched to six decimal places. The quadratic
update shifted every angle by less than 0.03° and left the gate outcomes
unchanged. Tangent-row retention reduced the reference and refined timestep
shifts to 4.178286° and 2.808295°, with all four angles below 30°. Both
controls were reverted. The next comparison needs accepted-step
wall-contact impulse and omitted low-mass stress impulse over the pile.

### Accepted-step dry momentum budget

For every accepted step the fixture accumulates the absolute proposed and
omitted stress impulses by face, the signed omitted work against the final
face velocity, and grid and particle wall impulses separately. Particle wall
impulse uses mass times the stored-velocity correction. The advecting
velocity correction supplies no second impulse. The wall oracle expects
(−0.5, 1, 0) N·s for a 1 kg point initially moving at (1, −1, 0) m/s
with friction 0.5. A rejected destination preserves the budget sentinel.

For PIC, the fixture compares final vertical particle momentum to gravity
and the two wall impulses. It requires the absolute residual to be within
the accumulated absolute omitted stress impulse plus 1e-6 N·s for
floating-point and CUDA quantization effects. The check runs for both
snow resolutions and all four sand width/timestep combinations. It uses
a signed vertical residual and gives no full-step energy claim.

| Sand h (m) | nominal dt (s) | Proposed stress L1 (N·s) | Omitted stress L1 (N·s) | Particle wall Y (N·s) | Contacts |
|---|---:|---:|---:|---:|---:|
| 0.075 | 0.001875 | 16.433269 | 0.000035746 | 17.306887 | 5760 |
| 0.0625 | 0.0015625 | 17.100097 | 0.000200861 | 17.316426 | 6824 |
| 0.075 | 0.0009375 | 19.511169 | 0.000079376 | 17.296351 | 7763 |
| 0.0625 | 0.00078125 | 20.819777 | 0.000261465 | 17.274316 | 8617 |

Native values shown. CUDA agreed at the printed precision in the table.
Grid wall impulse is zero in the sand fixture. The omitted stress impulse
is at most 0.0013% of the proposed L1 impulse. Halving the timestep raises
the number of contact events at both widths while changing net vertical
wall impulse by less than 0.25%. This checks the aggregate budget and leaves
the spatial and temporal distribution of floor response as an open
explanation for the angle sensitivity.

## Reproduction

Use an x64 Visual Studio developer PowerShell. Set up the pinned Conan
dependencies and toolchain receipts from the [native validation guide](README.md).
The measured Windows host used MSVC 14.44.35207, CUDA 13.4, RTX 5060 Ti
(SM120), driver 610.88, and clang-format 19.1.5. The host's uv shim was
0.13.6 and the lock requires 0.13.7 or newer, so the runs used an existing
Python 3.13 virtual environment, Conan CMake package metadata, and machine
toolchain receipts from sibling worktrees. The M5 sources, generated
fixtures and build directories were local to this worktree.

```powershell
cmake --build --preset windows-debug-cpu --parallel 6
ctest --preset windows-debug-cpu --output-on-failure -j 6
ctest --preset windows-debug-cpu -R '^M5' -V
cmake --build --preset windows-release-cuda --target fluidity_cuda_m5_constitutive_tests fluidity_cuda_m5_dry_scene_tests
ctest --preset windows-release-cuda -R '^M5.*\.cuda$' -V
compute-sanitizer --tool memcheck --error-exitcode 1 ./out/build/windows-release-cuda/fluidity_cuda_m5_constitutive_tests.exe
compute-sanitizer --tool memcheck --error-exitcode 1 ./out/build/windows-release-cuda/fluidity_cuda_m5_dry_scene_tests.exe

# Both commands intentionally retain the failing finer-grid qualification.
& ./out/build/windows-debug-cpu/fluidity_m5_dry_scene_tests.exe --q-sand-refined
& ./out/build/windows-release-cuda/fluidity_cuda_m5_dry_scene_tests.exe --q-sand-refined

# Timestep sensitivity at the same two widths. Exit 1 records the failed reference case.
& ./out/build/windows-debug-cpu/fluidity_m5_dry_scene_tests.exe --q-sand-half-step
& ./out/build/windows-release-cuda/fluidity_cuda_m5_dry_scene_tests.exe --q-sand-half-step

# Stored-velocity diagnostic at both widths and timesteps. Exit 1 retains the misses.
& ./out/build/windows-debug-cpu/fluidity_m5_dry_scene_tests.exe --q-sand-stored-blend
& ./out/build/windows-release-cuda/fluidity_cuda_m5_dry_scene_tests.exe --q-sand-stored-blend

$nativeSources = git ls-files -- 'native/*.cc' 'native/*.h' 'native/*.cu' 'native/*.cuh'
clang-format --dry-run --Werror @nativeSources
git diff --check
```

The final native Debug suite passed 666/666 CTest cases. Both focused CUDA binaries passed memcheck with zero errors. The focused
native and CUDA CTest runs execute two cases each, with multiple physical
checks in each executable.

## Production integration gate

M4's active `codex/m4-current-main` was reviewed through `0426af8`. Its
liquid viscosity, traction and multiphase receipts supply liquid-side
evidence. They provide no shared solid/liquid particle step. M6's active
`codex/m6-solids-cutting-rigids` was reviewed through `669236e`. M6 has
bounded elastic stress, cut, translating blade and rigid interaction
fixtures, with a separate interaction checkpoint. The coordinated interface
uses stable nonzero uint64 body IDs, world SI units, cut-side identity
separate from body identity, and publication on acceptance. The isolated M5
pair impulse API was shared with M6. The production liquid checkpoint
schema was left unchanged.

The mixed gate needs a versioned resolved scene carrying snow material
parameters, liquid pressure and distinct body memberships in the same
accepted substep. The standalone M5 face accumulator keeps separate mass and momentum by body
ID, and rejects a fifth body with `E-CONTACT-BODY-CAPACITY`. The production
step still needs to populate and consume those records transactionally. The coupled
scheduler must resolve contact with signed incoming relative normal speed,
apply each impulse once, preserve nonpenetration over the actual obstacle
sweep, and account for pressure traction and obstacle work without counting
the contact reaction again. Where partitioned pressure feedback is used,
record its lag and added-mass behavior.

Record matched CPU and target CUDA receipts for the authored mound,
obstacle sweep and liquid source. They must include material mass and volume
budgets, pre/post contact momentum and work, compaction, pressure/flux
residuals, capacity and rejected-step tests, plus interrupted/restarted
agreement. A versioned accepted transaction must include liquid pressure
and source history, solid elastic deformation and plastic volume, ordered
body poses and history, cut-side identities where applicable, and consumed
impulses. The Blender run must inspect the committed compaction/contact
frame and render snow with a distinct published substance/material channel.
The isolated pair fixture and the separate M6 checkpoint do not substitute
for these coupled receipts.

## Review

A fresh whole-branch reviewer reproduced two material problems and identified
an invalid initial-state acceptance path. The fixes have regressions for
exactly dependent deformation rows, principal-plane loss of ellipticity
with positive axial modes, and initial overcompression. A separate
destination-support regression protects rejected-step publication. The
platen fixture now splits at reversal and removal. The dry solver reuses
the existing owned stencil and wall sweep, with a local scratch stress
accumulator. M6's separately reviewed single-particle transfer commit was
not cherry-picked into this branch.
