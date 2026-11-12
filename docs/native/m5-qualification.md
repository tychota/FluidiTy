# M5 qualification on Windows

Local branch `codex/m5-granular-snow`, based on verified upstream
`549fb3f844841b79d14b7beb9ecfb6db9727c976`. This report covers the bounded
M5 implementations against §§2.3, 5.2, 8.1–8.2 and 15.2 of the
[design](../superpowers/specs/2026-11-07-fluidity-design.md).

The branch now includes main through 160e2fa. Constitutive models,
face contact, impulse scatter, dry snow, and plastic checkpoint records
landed in GitHub PRs 74, 80, 81, 82, and 83. Draft PR 75 continues the
milestone qualification. The dry snow and plastic checkpoint PRs were
rebuilt and passed native and CUDA tests after integrating main. The
refreshed milestone branch passed all nine M5 CTest targets after a
clean Windows Release CUDA rebuild.

| Gate | Status | Evidence |
|---|---|---|
| Granular constitutive invariants | Pass | Elastic compression, translated tensile apex, compressed shear return to the cone at fixed trace, plastic volume transfer, and invalid-state output preservation. |
| Snow constitutive invariants | Pass | Compression and tension clamps, reconstructed total volume, hardened corotational stress, repeated compaction, and invalid-state output preservation. |
| Full 3D material state and wave bound | Pass for the tested domain | Rotated and repeated stretches, reflection and singular rejection, isotropic and hardened wave oracles, and rejection of axial and principal-plane loss of ellipticity. Principal-plane checks do not certify ellipticity in every fully oblique direction. The Hessian row bound covers continuous elastic wave speed. The tested discrete acoustic factor is 0.25 for the bounded scenes. |
| Q-SOLID uncut quadrature | Pass | Analytic affine gradient gives 87.5 W particle power and −87.5 W grid power, with 3.25e-16 scaled residual on both backends versus 1e-6 allowed. This measures uncut stress quadrature and the matching gather derivative. |
| Q-SAND 30° pile | Open | The nominal reference case reaches 30.161149° at 0.75 s and continues flattening to 20.801830° at 1.5 s. All four grid-width and timestep combinations miss the 30–35° range at the extended endpoint. |
| Dry snow compaction | Pass | Two resolutions pass a compress–unload cycle. Mean log plastic Jacobian stays below −0.34 after unload. |
| Dry step acceptance | Pass for the tested domain | Motion and acoustic bounds are checked during the step. Tests reject an inadmissible initial snow state and a move out of allocated support, preserving the particle record. |
| Isolated distinct-body contact | Pass | Closing, separating, sticking, sliding and invalid pairs. Unequal masses give 6 N·s normal and −1.5 N·s sliding impulse on the first body. Momentum is conserved within the fixture tolerance, and pair kinetic energy decreases. |
| Isolated face body capacity | Pass | Four sorted distinct body records, accumulation for an existing body at capacity, fifth-body rejection, and range-failure rollback on CPU and CUDA. The host error is ResourceExhausted / E-CONTACT-BODY-CAPACITY. |
| Isolated face pair application | Pass | Two bodies receive equal-and-opposite updates to their stored face momenta. The fixture checks normal closure, sliding friction, combined momentum, an unaffected third body, and rejection rollback on native and SM120. |
| Isolated face-to-particle scatter | Pass | A native host fixture deposits an unequal-weight pair, resolves it and recovers both face impulses by summing particle mass times velocity increment. The third body is untouched, and malformed contributor order rejects. The shared per-particle scatter arithmetic also ran on SM120. |
| Isolated multi-face contact staging | Pass | A native host fixture stages two detected faces in slot order, aggregates repeated contributors, and checks combined particle momentum. A missing body on the later face, duplicate slot, and excess total weight each reject the batch. |
| Isolated mixed-contact velocity update | Pass on host | Staged face impulses update the actual liquid and plastic stored velocities atomically. The fixture checks combined momentum, untouched liquid volume and snow history, late rejection, successful retry and final-velocity overflow rollback. |
| Isolated planar mixed contact detection | Pass on host | Certified halfspace sweep admits an analytic interior or endpoint hit, leaves a near miss unchanged, and rejects a later overlapping candidate before publishing any impulse. |
| Isolated common-impact mixed advection | Pass on host | Two translating planar interfaces hit halfway through the attempt and both trial positions reach 0.1875 m. An insufficient path-length budget rejects both velocity and position changes. |
| Particle-derived planar contact fronts | Pass for the bounded axis-aligned model | The fixture reconstructs a 0.25 m gap from liquid and snow particle positions and explicit 0.0625 m offsets, then reaches the analytic impact and post-impact positions. Reversed normal and conflicting tied-front velocity checks pass. |
| Plastic checkpoint handoff | Pass for the typed dry payload and wire record | The snow fixture captures and restores a compressed state with nonzero log plastic Jacobian and APIC fit. After a FlatBuffers record round trip, every particle field matches the uninterrupted next accepted step on native and CUDA. Invalid plastic state, wrong material ID and duplicate particle IDs reject. |
| One-way pressure input to planar mixed contact | Pass for the frozen one-cell fixture | Native and CUDA pressure projection both match the analytic outlet speed and pressure. Host contact matches impact time, impulse, and final positions. A late motion-bound rejection preserves both input particles. |
| Reciprocal pressure for a planar snow interface | Pass for the frozen single-cell fixture | Native and CUDA pressure solvers both recover the analytic speed, pressure, impulse, and work for the two wall orientations. The trial returns the pressure candidate and snow impulse together. |
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

### Extended settling of the original sand column

The extended runs use the original 63 particles, centered grid, drop,
material, and floor. They preserve the split steps at 0.70 and 0.75 s
and reproduce all four original angles at 0.75 s. Continue from that
state to 1.50 s and retain the same 0.05 s drift window and momentum gate.

| h (m) | nominal dt (s) | Angle at 0.75 s | Angle at 1.50 s | Last 0.05 s drift | Final kinetic energy (J) |
|---|---:|---:|---:|---:|---:|
| 0.075 | 0.001875 | 30.161149° | 20.801830° | 0.500147° | 0.000180 |
| 0.0625 | 0.0015625 | 27.803822° | 20.580671° | 0.399749° | 0.000227 |
| 0.075 | 0.0009375 | 37.248475° | 28.040080° | 0.362457° | 0.000208 |
| 0.0625 | 0.00078125 | 32.699841° | 26.585383° | 0.302212° | 0.000300 |

Native and CUDA angles and drift match to six decimal places. Both
diagnostic commands return 1. Each angle falls by 6.11–9.36° between
the two endpoints even though
all cases pass the last-window drift and vertical momentum checks.
The nominal reference angle at 0.75 s is a transient pass. The local
drift and low kinetic energy at 1.50 s coexist with substantial
longer-horizon shape change. Qualification needs evidence that the pile
stays in range over a longer window at both timesteps.

```powershell
.\out\build\windows-release-cuda\fluidity_m5_dry_scene_tests.exe --q-sand-column-long-settle
.\out\build\windows-release-cuda\fluidity_cuda_m5_dry_scene_tests.exe --q-sand-column-long-settle
```

### Elastic stiffness during extended settling

Multiply both granular Lamé moduli by four, from (700, 450) Pa to
(2800, 1800) Pa. Keep the 30° friction angle, cohesion, particle mass,
geometry, gravity, floor friction and timestep schedule fixed. The dry
solver continues to check its normal acoustic and motion limits.

| h (m) | Nominal dt (s) | Baseline angle at 1.50 s | Stiffer angle at 0.75 s | Stiffer angle at 1.50 s |
|---|---:|---:|---:|---:|
| 0.075 | 0.001875 | 20.801830° | 27.169714° | 18.733064° |
| 0.075 | 0.0009375 | 28.040080° | 33.323242° | 26.064333° |
| 0.0625 | 0.0015625 | 20.580671° | 26.186834° | 18.925497° |
| 0.0625 | 0.00078125 | 26.585383° | 30.406358° | 24.891280° |

Native and CUDA angles and drift match to six decimal places. All runs
pass the momentum and late-drift checks. The stiffer pile is 1.65–2.07°
flatter at the extended endpoint.
The mean log plastic volume decreases in every comparison. A stiffer
elastic response alone does not recover the repose angle in this
discretization. All four extended angles miss the gate.

```powershell
.\out\build\windows-release-cuda\fluidity_m5_dry_scene_tests.exe --q-sand-stiffer-column
.\out\build\windows-release-cuda\fluidity_cuda_m5_dry_scene_tests.exe --q-sand-stiffer-column
```

### Extended settling after a velocity reset

At the original 0.75 s endpoint, clear each particle's translational
velocity and APIC slopes. Preserve positions, masses, elastic
deformation and plastic history. Resume immediately under the same
gravity and contact law. Count the reset as an external vertical impulse
in the momentum budget. The listed kinetic energy is translational.

| h (m) | Nominal dt (s) | Removed kinetic energy (J) | Uninterrupted angle at 1.50 s | Reset angle at 1.50 s |
|---|---:|---:|---:|---:|
| 0.075 | 0.001875 | 0.002264 | 20.801830° | 20.859030° |
| 0.0625 | 0.0015625 | 0.001680 | 20.580671° | 20.615958° |
| 0.075 | 0.0009375 | 0.002686 | 28.040080° | 28.138482° |
| 0.0625 | 0.00078125 | 0.004526 | 26.585383° | 26.642217° |

Native and CUDA angles and drift match to six decimal places. All runs
pass the drift and adjusted momentum checks. The reset changes the
endpoint by less than 0.10° in every case.
The resumed piles still flatten by 6.06–9.30° relative to the restart
shape and all four miss the angle gate. The sustained change develops
under the force and deformation update after the reset. Continuing
particle momentum and APIC slopes at 0.75 s account for little of it.

```powershell
.\out\build\windows-release-cuda\fluidity_m5_dry_scene_tests.exe --q-sand-reset-column-velocity
.\out\build\windows-release-cuda\fluidity_cuda_m5_dry_scene_tests.exe --q-sand-reset-column-velocity
```

### Floor-friction sensitivity during extended settling

Use the original column, centered grids, nominal timesteps, and 1.50 s
endpoint. Change only the stationary floor's Coulomb coefficient. The
granular friction angle stays at 30°. Coefficient 0 removes the
tangential reaction. Coefficient 10 increases the allowed impulse
without imposing a separate no-slip boundary condition.

| h (m) | Nominal dt (s) | μ_floor = 0 | μ_floor = 0.7 | μ_floor = 10 |
|---|---:|---:|---:|---:|
| 0.075 | 0.001875 | 11.336790° | 20.801830° | 20.806219° |
| 0.0625 | 0.0015625 | 7.022668° | 20.580671° | 20.587004° |

Native angles are shown. CUDA agrees within 0.000001° at the printed
precision. The drift and momentum checks pass throughout, and both
diagnostic commands return 1 for the angle gate. Removing floor friction
substantially increases flattening. Increasing it from 0.7 to 10 changes
the final angle by less than 0.007° at both widths. The late deficit
persists even with the higher impulse limit.

```powershell
.\out\build\windows-release-cuda\fluidity_m5_dry_scene_tests.exe --q-sand-floor-friction
.\out\build\windows-release-cuda\fluidity_cuda_m5_dry_scene_tests.exe --q-sand-floor-friction
```

### Horizontal grid alignment

Shift only the MAC origin by half a cell in x and then in both x and z.
The initial particles, physical floor, gravity, duration, angle estimator,
and momentum and settling limits are identical. The zero-shift controls
reproduce the nominal reference and finer-grid cases.

| h (m) | nominal dt (s) | (x, z) origin in cell widths | Native and CUDA envelope | Last 0.05 s drift |
|---|---:|---|---:|---:|
| 0.075 | 0.001875 | (0, 0) | 30.161149° | 0.922262° |
| 0.075 | 0.001875 | (0.5, 0) | 30.063385° | 0.957539° |
| 0.075 | 0.001875 | (0.5, 0.5) | 30.022695° | 0.988896° |
| 0.0625 | 0.0015625 | (0, 0) | 27.803822° | 0.677401° |
| 0.0625 | 0.0015625 | (0.5, 0) | 27.009983° | 0.704724° |
| 0.0625 | 0.0015625 | (0.5, 0.5) | 25.829145° | 0.790237° |

Both backends agree at the printed precision and pass all vertical momentum
and drift checks. Both diagnostic commands return 1 for the finer cases.
The measured
alignment spread grows from 0.138454° to 1.974677° at the finer width.
Every finer alignment misses the 30–35° gate. The shift mainly changes
the height in the finer case. Its radius changes by less than 0.4 mm,
while its height decreases by about 6.5 mm. A subsequent transfer or
boundary change should be checked at each of these origins.

```powershell
.\out\build\windows-release-cuda\fluidity_m5_dry_scene_tests.exe --q-sand-grid-phase
.\out\build\windows-release-cuda\fluidity_cuda_m5_dry_scene_tests.exe --q-sand-grid-phase
```

### Stored velocity experiment

In the historical 6d31b68 experiment, the optional dry-step blend retained 0.594 of the particle velocity
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

### Stored velocity retention scaled by physical time

Use the existing momentum-balanced stored residual, but apply a fixed
decay rate across timesteps. For step duration dt, set the retained
fraction to exp(−k dt), with k chosen to reproduce 0.594 at 0.001875 s.
The existing API caps retention at 0.99. Use the actual duration of
each split step. The advection field, deformation gradient, and
low-mass eligibility checks remain those of the ordinary dry solver.

| h (m) | Nominal dt (s) | PIC angle at 1.50 s | Time-scaled angle at 0.75 s | Time-scaled angle at 1.50 s |
|---|---:|---:|---:|---:|
| 0.075 | 0.001875 | 20.801830° | 25.815309° | 16.318601° |
| 0.075 | 0.0009375 | 28.040080° | 28.460215° | 20.885390° |
| 0.0625 | 0.0015625 | 20.580671° | 24.704364° | 17.636402° |
| 0.0625 | 0.00078125 | 26.585383° | 27.751037° | 22.692469° |

Native and CUDA match to six decimal places. Every run passes the
vertical momentum and late-drift checks. At 1.50 s the timestep gap
narrows from 7.24° to 4.57° at the
reference width and from 6.00° to 5.06° at the finer width. Every
endpoint is shallower than PIC and below the Q-SAND range. Scaling
retention by physical time leaves substantial timestep dependence.

```powershell
.\out\build\windows-release-cuda\fluidity_m5_dry_scene_tests.exe --q-sand-time-scaled-retention
.\out\build\windows-release-cuda\fluidity_cuda_m5_dry_scene_tests.exe --q-sand-time-scaled-retention
```

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

### Sand particle sampling

The original column uses 63 cell-centered particles. Doubling the
samples per axis gives 504 particles with 0.025 m spacing, the same
0.15 × 0.35 × 0.15 m initial column, and the same 2.3625 kg total mass.
The release height, grid, nominal timesteps, floor friction, sand law,
and measurement times are unchanged.

| h (m) | nominal dt (s) | 63 particles | 504 particles |
|---|---:|---:|---:|
| 0.075 | 0.001875 | 30.161149° | 26.727381° |
| 0.0625 | 0.0015625 | 27.803822° | 23.585775° |
| 0.075 | 0.0009375 | 37.248475° | 33.443262° |
| 0.0625 | 0.00078125 | 32.699841° | 28.793882° |

Native Release values shown. Native and CUDA differ by at most 1e-6° at the reported precision across
the four dense runs. The vertical momentum budget passed in all four.
Increasing particle sampling lowered the measured angle by 3.43–4.22°
and left a pronounced timestep effect. The reference nominal angle
from the 63-particle fixture therefore does not establish sampling
convergence.

With the same 504 particles, two finer grids were then compared at matched
timesteps. This keeps the physical sampling and material law fixed.

| h (m) | nominal dt (s) | Native angle | CUDA angle |
|---|---:|---:|---:|
| 0.0375 | 0.00078125 | 21.217540° | 21.217540° |
| 0.03125 | 0.00078125 | 20.379811° | 20.379819° |
| 0.0375 | 0.000390625 | 22.110279° | 22.110280° |
| 0.03125 | 0.000390625 | 22.184540° | 22.184549° |

The finer grids reduce the observed timestep shifts to 0.89° and 1.80°,
respectively. All four angles remain below 30°, while final 0.05 s drift is
below 0.49° and the vertical momentum budget still passes. The data show a
settled-angle discrepancy under refinement. They do not establish an
asymptotic slope. The next physical comparison should isolate the floor
transfer and the pile shape measurement on a spatially resolved benchmark
before any material calibration.

### Independent sand surface fit

The refined-grid command also reports a separate free-surface fit. In
each of four horizontal quadrants, divide 15–85% of the maximum radius
into six annuli. Retain the highest particle in each occupied annulus
and regress its height against its actual radius. A quadrant needs at
least four occupied annuli. Report the range of resolved quadrant slopes
as a measure of angular sampling spread. This excludes the apex and toe
from the fit. The manufactured 30° cone recovers 30° in all quadrants
within 1e-11°, even with additional samples below the surface.

| h (m) | nominal dt (s) | Native surface fit | CUDA surface fit |
|---|---:|---:|---:|
| 0.0375 | 0.00078125 | 14.817355° | 14.817355° |
| 0.03125 | 0.00078125 | 8.289243° | 8.289234° |
| 0.0375 | 0.000390625 | 20.016329° | 20.016329° |
| 0.03125 | 0.000390625 | 11.285081° | 11.285063° |

All four quadrants resolved in every run, with zero spread at the printed
precision. The symmetric fixture makes this spread a weak bound on
discretization uncertainty. The surface fit is lower than the original
envelope angle in every refined run. Thus an envelope dominated by the
outermost grains does not explain the Q-SAND discrepancy. The gate still
uses the original protocol and remains open.

### Stationary floor deformation history

Starting from 476ce05, the experimental sand step gathered a second
velocity field for the deformation gradient alone. At the stationary
floor, set the y-normal face on the plane to zero. For faces below the
plane, use the reflected positive-mass donor with odd parity for y
velocity and even parity for x and z. Use the original value if the donor
is unavailable. The cell-centered staggered offset makes the donor
lattice index −y−1 for tangential faces and −y for normal faces.
The deposited force, PIC transport, incoming contact velocities,
stored-velocity reaction and affine fit all use their original fields.

| h (m) | nominal dt (s) | Original envelope | Reflected-gradient envelope | Reflected-gradient surface fit |
|---|---:|---:|---:|---:|
| 0.0375 | 0.00078125 | 21.217540° | 21.452741° | 7.454601° |
| 0.03125 | 0.00078125 | 20.379811° | 17.789592° | 1.773206° |
| 0.0375 | 0.000390625 | 22.110279° | 21.881091° | 11.492093° |
| 0.03125 | 0.000390625 | 22.184540° | 20.701303° | 14.766756° |

Native results shown. CUDA agreed within 1e-5° for the reported angles.
The momentum checks passed and every angle gate still failed. The
prototype changed the gradient alone, so it does not test the full
compatible deposit and ghost gather pair or its work budget. The
experimental source was reverted. The exact local comparison diff is
in the worktree's out/m5-floor-reflected-gradient.patch. Apply it against
476ce05 and run the refined-grid commands below to repeat the control.

### Impact-time normal deformation control

Split the granular trial deformation at the existing swept-floor impact
time. Keep the frozen gathered velocity gradient before impact. For the
remaining time, set the floor-normal velocity derivative row to zero and
keep the two tangential rows. Compose the two forward-Euler increments
and use the same constitutive return. This control isolates normal
kinematics. It leaves out the derivative of the Coulomb tangential
projection. Transport, impulses, APIC slope storage, and snow are unchanged.

| h (m) | nominal dt (s) | Origin (x, z) in cell widths | Baseline envelope | Split envelope |
|---|---:|---|---:|---:|
| 0.075 | 0.001875 | (0, 0) | 30.161149° | 31.234535° |
| 0.075 | 0.001875 | (0.5, 0) | 30.063385° | 31.141341° |
| 0.075 | 0.001875 | (0.5, 0.5) | 30.022695° | 31.102911° |
| 0.0625 | 0.0015625 | (0, 0) | 27.803822° | 28.757479° |
| 0.0625 | 0.0015625 | (0.5, 0) | 27.009983° | 28.127354° |
| 0.0625 | 0.0015625 | (0.5, 0.5) | 25.829145° | 27.222141° |
| 0.075 | 0.0009375 | (0, 0) | 37.248475° | 37.701706° |
| 0.0625 | 0.00078125 | (0, 0) | 32.699841° | 33.075978° |

Native angles are shown. CUDA agrees within 0.000001° at the printed
precision. All sixteen runs pass the vertical momentum and drift checks.
The normal-only split steepens all eight envelopes and leaves all
gate outcomes unchanged. The finer nominal grid still misses the
angle range at every origin. The timestep gap remains above 4° on both
centered grids. The production solver was restored. Apply
out/m5-impact-split-gradient.patch against 820cde2, rebuild both Release
scene targets, and run both the grid-phase and half-step commands to
repeat the control.

### Coulomb-projected post-impact deformation

Extend the normal-only split by differentiating the tangential Coulomb
projection for the stationary floor. Let u be incoming tangential
velocity, n be the signed incoming normal velocity, and r = |u|.
In sliding, the post-impact tangent is (1 + μn/r)u. Differentiate
both the tangent vector and the scale using the frozen gathered gradient.
In sticking, use a zero post-impact gradient. The normal row is zero
in both contacting branches. Twelve directional central differences
away from branch boundaries checked the sliding, sticking, frictionless
and separating formulas to 1e-9. The prototype freezes the local branch
and the impact fraction and retains the existing particle trajectory,
impulses and APIC slope storage.

| h (m) | nominal dt (s) | Origin (x, z) in cell widths | Normal-only envelope | Coulomb-projected envelope |
|---|---:|---|---:|---:|
| 0.075 | 0.001875 | (0, 0) | 31.234535° | 32.825896° |
| 0.075 | 0.001875 | (0.5, 0) | 31.141341° | 32.524599° |
| 0.075 | 0.001875 | (0.5, 0.5) | 31.102911° | 32.251450° |
| 0.0625 | 0.0015625 | (0, 0) | 28.757479° | 30.052977° |
| 0.0625 | 0.0015625 | (0.5, 0) | 28.127354° | 29.383951° |
| 0.0625 | 0.0015625 | (0.5, 0.5) | 27.222141° | 28.351919° |
| 0.075 | 0.0009375 | (0, 0) | 37.701706° | 37.729429° |
| 0.0625 | 0.00078125 | (0, 0) | 33.075978° | 33.013884° |

Native angles are shown. CUDA agrees within 0.000001° at the printed
precision. All sixteen runs pass the vertical momentum and drift checks.
The centered nominal finer case reaches the gate by 0.052977°, while
both shifted cases miss. The centered coarse half-step case remains above
35°. This locates an influence on the nominal angle and leaves the
alignment and timestep sensitivity unresolved. Apply
out/m5-impact-coulomb-gradient.patch against 4c116dd, rebuild both
Release scene targets, and run the grid-phase and half-step commands.
The production solver was restored.

### Floor-supported release and the 30° material oracle

The drop fixture gives the sand column 0.10 m of clearance above the floor.
The floor-supported control removes only that clearance, with the same
504 midpoint samples, grid, timestep, gravity, material, floor friction,
end time, envelope estimator and quadrant fit.

| h (m) | nominal dt (s) | Drop envelope | Floor-supported envelope | Floor-supported surface fit |
|---|---:|---:|---:|---:|
| 0.0375 | 0.00078125 | 21.217540° | 22.561058° | 10.739778° |
| 0.03125 | 0.00078125 | 20.379811° | 21.083522° | 4.301806° |
| 0.0375 | 0.000390625 | 22.110279° | 23.361683° | 15.801772° |
| 0.03125 | 0.000390625 | 22.184540° | 23.062105° | 7.736924° |

Native values shown. CUDA angle differences were at most 0.000011° at
the reported precision. Vertical momentum checks passed on both backends.
Removing the drop lowered mean log plastic Jacobian from 0.199–0.253 to
0.073–0.098, but all four floor-supported angles still missed Q-SAND.
The initial drop contributes to the material history without explaining
the shallow pile by itself.

An independent cohesionless triaxial compression test checks that the
30° material maps to a 3:1 axial-to-radial compressive stress ratio.
A trial with principal Kirchhoff stresses (−5, −1, −1) Pa returns
(−4.2, −1.4, −1.4) Pa with unchanged stress trace and plastic volume.
Native and CUDA meet the 1e-11 fixture tolerance. This verifies the
prescribed friction mapping for triaxial compression. The pile gate
remains a separate spatial benchmark.

### Paired floor reflection and grid-reaction friction

The follow-up used the transpose of the reflection map for stress force.
Each original below-floor face force was added to the selected donor with
the same odd or even sign used by the gradient gather. The y-normal force
on the floor plane was eliminated. The change in the signed y force sum
times dt was accounted for as grid wall impulse. The 0.0375 m runs finished
with envelope angles of 10.600131° and 12.914026°. Both 0.03125 m runs
rejected after reaching the horizontal support boundary, on CPU and CUDA.

Widening only the x and z block ranges to [−4, 4] let the finer runs
finish. In these runs the grid supplied over half of the normal wall
impulse, while the original friction law used only the swept particle
impulse. A further control distributed the signed grid reaction through
the original y-face weights and deposited masses, then used its positive
per-particle portion to cap the pre-sweep tangential speed reduction by
the existing friction coefficient. The sweep then used its own normal
impulse for its own friction correction. Both stored and advecting
velocities received the pre-sweep tangential correction.

| h (m) | nominal dt (s) | Paired reflection envelope | With grid-reaction friction |
|---|---:|---:|---:|
| 0.03125 | 0.00078125 | 7.107824° | 18.438913° |
| 0.03125 | 0.000390625 | 9.504568° | 22.874822° |

Native values shown. Vertical momentum checks passed on both backends
for the completed trials. The grid-reaction friction control restored
some resistance to spreading, but still missed Q-SAND and did not
qualify a full Coulomb work or torque budget. These were temporary
solver controls. Local diffs against e65e078 are
out/m5-paired-floor-original-halo.patch,
out/m5-paired-floor-padded.patch and
out/m5-paired-floor-friction.patch. The first uses the dense refined
grid commands below. The padded variants use the same binaries with
the experimental --q-sand-paired-floor-padded argument.

### 45° cone release

A cone starting above the required repose window tests relaxation without
the high-aspect-ratio column and drop. The 45° cone has radius and height
0.175 m and starts on the floor. A 0.025 m midpoint lattice gives 371
particles and 1.7390625 kg. The granular law, gravity and friction are
unchanged. At each grid width and timestep the envelope criterion remains
30–35°, the last 0.05 s drift limit stays 2°, and kinetic energy must be
below 0.01 J. The quadrant fit remains an independent diagnostic.

| h (m) | nominal dt (s) | Native envelope at 0.75 s | Native quadrant fit |
|---|---:|---:|---:|
| 0.0375 | 0.00078125 | 33.417161° | 36.466922° |
| 0.03125 | 0.00078125 | 32.183522° | 35.923693–36.296073° |
| 0.0375 | 0.000390625 | 36.531421° | 38.533738–38.861962° |
| 0.03125 | 0.000390625 | 35.383665° | 38.436984–38.778075° |

CUDA matched the native envelope to 1e-6° at the printed precision.
The coarser nominal CUDA quadrant minimum was 36.117503° while the other
three quadrants and the native result were 36.466922°. This demonstrates
surface-sample selection sensitivity despite near-identical envelopes.
Both nominal cases meet the envelope criterion. Both half-step cases
miss it, and all quadrant fits exceed 35° at 0.75 s.

With the same start and acceptance criteria, extending the end time to
1.5 s moves the envelope at the nominal timestep below 30°. The half-step
runs are inside the envelope window at this later time.

| h (m) | nominal dt (s) | Native envelope at 1.5 s | Native quadrant fit |
|---|---:|---:|---:|
| 0.0375 | 0.00078125 | 29.594340° | 34.300833–36.120679° |
| 0.03125 | 0.00078125 | 28.245806° | 33.775882–35.571562° |
| 0.0375 | 0.000390625 | 33.715553° | 36.147229–36.490832° |
| 0.03125 | 0.000390625 | 32.577189° | 36.169228–36.544226° |

CUDA envelope angles differed by at most 1e-6° at the reported precision.
The 1.5 s half-step angles are within 0.40° of the 0.75 s nominal-step
angles at equal grid width. The half-step runs use four times as many
nominal steps because the physical time doubles and the timestep halves. At fixed 1.5 s, halving the step shifts the
angle by more than 4° on both grids. All momentum checks pass. This
shows that the cone changes the benchmark result but does not establish
a converged repose angle, even when the last 0.05 s drift meets the
declared bound.

### Conservative stored velocity and the advection control

The original storage-only blend excluded low-mass supports independently.
It could therefore add net momentum even without wall contact. A manufactured
one-body test with four eligible particles and one fallback particle measured
3.621335079 N·s after a step from 3 N·s with no external x impulse. The production
fix subtracts the mass-weighted mean correction over eligible particles.
Fallback particles retain PIC storage. Both native and CUDA now preserve
the manufactured momentum within 1e-9 N·s. The scene applies its existing
vertical momentum gate to the stored-blend cases as well.

An advection prototype also used the existing energy setting 0.6 to scale
the retained correction in advection. It accepted the extra displacement
only when the PIC and residual paths both passed the swept-wall and motion
checks without contact. With the old unbalanced storage it put all cone
envelopes in the window, but the native vertical residual reached
0.522350393 N·s. With balanced storage, all eight momentum checks passed.
The nominal fine-grid envelope was 29.981364° on CPU and 30.027238° on CUDA,
so the prototype did not consistently pass Q-SAND. Independent surface
fits also continued to exceed 35° in several cases. Residual advection was
removed from the production change. The experimental patches against
9562d88 are out/m5-cone-unbalanced-advection.patch and
out/m5-cone-balanced-advection.patch. Their --q-sand-cone-energy command
runs the four 0.75 s cone cases.

The production storage-only comparison at 0.75 s now passes vertical
momentum on all four CPU and all four CUDA runs. Its nominal fine-grid
envelope remains outside the window at 29.903758° on CPU and 29.990626°
on CUDA. The other three envelope cases are inside. This does not close
Q-SAND, and the production default still uses PIC.

### Matched native and CUDA deposition precision

The nominal fine-grid cone at energy setting 0.6 differed between the
native floating-point path and the CUDA quantized path. The quantization
control uses the native quantized MAC deposit with the CUDA scales, then
uses the native gather and the same dry-step acceptance. A second run
decreases both quanta by 256. The default production paths are unchanged.

| Deposit path | Mass quantum (kg) | Momentum quantum (kg m/s) | Envelope |
|---|---:|---:|---:|
| Native floating point | — | — | 29.903758° |
| Native integer | 2^-42 | 2^-38 | 29.990626° |
| CUDA integer | 2^-42 | 2^-38 | 29.990626° |
| Native integer | 2^-50 | 2^-46 | 29.990626° |
| CUDA integer | 2^-50 | 2^-46 | 29.990626° |

Native and CUDA printed identical angle, contact-count, and momentum
diagnostics at each matched scale. The vertical momentum gate passed
in every run. This identifies deposit arithmetic as the source of the
observed native/CUDA split in this comparison. Finer quanta alone do
not move the quantized result into the window. Matching these operators
does not establish general bitwise backend replay or close Q-SAND.

### Low-mass retention taper control

The stored blend normally suppresses its correction when a positive-weight
face falls below the low-mass threshold. The temporary control kept zero
retention at and below that threshold and ramped linearly to full retention
at twice the threshold, taking the minimum across positive-weight faces.
The weighted mean correction uses particle mass times the ramp factor, so
the net retained momentum still cancels.

| Deposit path | Existing envelope | Taper envelope | Taper surface fit |
|---|---:|---:|---:|
| Native floating point | 29.903758° | 29.908319° | 34.495912° |
| Native or CUDA, matched standard quanta | 29.990626° | 29.994948° | 34.111658° |
| Native or CUDA, matched finer quanta | 29.990626° | 29.994948° | 34.111658° |

All three taper runs passed the momentum check. The taper changes the
fitted surface but neither removes the float/quantized difference nor
passes the envelope gate. This control addresses only the transition in
deposited mass for already-active stencil faces. It does not address
changes in stencil membership. The production source was restored.
Apply out/m5-cone-low-mass-taper.patch against 0158e3e and use the cone
quantization commands below, plus --q-sand-cone-float-taper on the native
Release binary, to reproduce the comparison.

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

# Release host and CUDA at matched particle sampling. Both controls report open gates.
cmake --build --preset windows-release-cuda --target fluidity_m5_dry_scene_tests fluidity_cuda_m5_dry_scene_tests
& ./out/build/windows-release-cuda/fluidity_m5_dry_scene_tests.exe --q-sand-dense-particles
& ./out/build/windows-release-cuda/fluidity_cuda_m5_dry_scene_tests.exe --q-sand-dense-particles
& ./out/build/windows-release-cuda/fluidity_m5_dry_scene_tests.exe --q-sand-dense-refined-grid
& ./out/build/windows-release-cuda/fluidity_cuda_m5_dry_scene_tests.exe --q-sand-dense-refined-grid
& ./out/build/windows-release-cuda/fluidity_m5_dry_scene_tests.exe --q-sand-floor-release
& ./out/build/windows-release-cuda/fluidity_cuda_m5_dry_scene_tests.exe --q-sand-floor-release
& ./out/build/windows-release-cuda/fluidity_m5_dry_scene_tests.exe --q-sand-cone-release
& ./out/build/windows-release-cuda/fluidity_cuda_m5_dry_scene_tests.exe --q-sand-cone-release
& ./out/build/windows-release-cuda/fluidity_m5_dry_scene_tests.exe --q-sand-cone-long-settle
& ./out/build/windows-release-cuda/fluidity_cuda_m5_dry_scene_tests.exe --q-sand-cone-long-settle
& ./out/build/windows-release-cuda/fluidity_m5_dry_scene_tests.exe --q-sand-cone-stored-only
& ./out/build/windows-release-cuda/fluidity_cuda_m5_dry_scene_tests.exe --q-sand-cone-stored-only
& ./out/build/windows-release-cuda/fluidity_m5_dry_scene_tests.exe --q-sand-cone-quantization
& ./out/build/windows-release-cuda/fluidity_cuda_m5_dry_scene_tests.exe --q-sand-cone-quantization

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

M6 commit 2294680 adds isolated rigid pressure traction and torque from
ordered, certified wet patches and solved liquid pressures. The mixed
transaction still needs to produce and consume those patches at the
declared pressure time level.

M6 commit 2627937 also publishes the planar wetted piston pressure load
with accepted liquid substeps. This establishes an accepted load receipt
for that geometry. A mixed snow/liquid substep, body-state advance, and
joint checkpoint still require separate evidence.

M6 commit 1f5363e feeds accepted liquid pressure into a pressure-only,
axially guided planar piston. Its fixture is particle-free and exercises
accepted axial pose feedback and rollback. It leaves snow contact, general
body motion, and mixed checkpoint replay open.

M6 commit 4cebb40 adds an accepted-step FLSI envelope for the liquid and
solid interaction histories. The guided-piston restart reproduces the
next accepted envelope byte for byte. This provides a joint codec and
ordinal check for that pressure-only case. Plastic material records and
particle-contact replay remain open for the mixed scene.

M6 commits cd7c50b and e342d94 add the guided CPU piston's swept particle
reaction and byte-identical coupled replay after a rejected attempt.
The fixture prescribes the incoming gather to isolate the impulse.
The CUDA reaction, deformable plastic particle records, and a production
snow–water restart remain open.

M6 commits 5d017de and 8a15777 measure light-body amplification for
partitioned pressure feedback and provide a stable coupled update for
one analytic pressure cell and a guided piston. M6 commit 62cbf0c extends that guided update to multiple pressure cells. The coupled path is particle-free.
It does not provide a coupled plastic-particle pressure or contact step.

M6 commit 11bd9c6 adds an isolated CUDA piston reaction output, reduced
in particle order and published transactionally with the particle batch.
Its SM120 fixture matches the CPU impulse and checks rejection rollback.
Accepted dynamic-body feedback still needs to consume that CUDA output.

M6 commit 9bce891 transacts pressure-row flux, reciprocal traction,
guided floating-box drift, and accepted checkpoint replay in the CPU
liquid scheduler. Its attempt is particle-free. Continuous finite-box
particle contact and the accepted plastic payload remain separate
dependencies for the snow–water scene.

M6 commit 62ff170 runs its coupled guided-box projection with production
liquid particles. The solved wall speed drives flux, drift and the
continuous particle sweep. The CPU scheduler retries when the coupled
body impulse residual exceeds its bound. M6 commit 78cd5ac adds
partitioned translation in all three axes for an aligned rigid box.
These provide production references for accepted pressure and contact
accounting. The deformable scene still needs a snow-derived moving
interface, reciprocal traction on its plastic particles, and an atomic
advance of both material states.

M5 provides a typed plastic checkpoint record with stable nonzero particle,
body and material IDs, the MAC transfer state, reference volume, accepted
elastic map and log plastic Jacobian. The dry adapter restores records in
strict particle ID order against a resolved body/material binding. It
uses the same admissibility and acoustic-state check as the dry step.
The next dry attempt bins from the accepted world position. The focused
test restores a compressed snow particle with nonzero APIC fit, advances
one step, and compares all state fields exactly on each backend.
The replay now passes through the generated PlasticParticle record in a
FlatBuffers byte vector. The owning FLSI codec and authored snow–water
restart still need integration.

The shared payload validator checks globally increasing particle IDs
against body bindings in strictly increasing body ID order. It resolves
each particle's material through its own body and applies the existing
dry acceptance test before the codec can publish a payload. Both M5
scene suites check an interleaved-by-body sand and snow vector, rejection
of a missing body binding, unsorted bindings, duplicate IDs across
bodies, and inadmissible snow in the later record. This host validator
does not write the production FLSI envelope.

The wire decoder accepts the optional vector only after the owning
FlatBuffers reader has verified its structure. It checks the supplied
particle capacity before allocation and returns an owned vector only
after semantic validation of every record. Both scene suites exercise
the absent-vector case, a two-record vector at capacity, a vector
above capacity, and a malformed late snow record. The remaining
codec work is to supply the verified field, enforce the envelope
version and pass scene bindings and the resolved particle limit.

The matching encoder checks capacity and validates the entire payload
before adding any bytes to the owning builder. An empty payload yields
an absent field. The round trip preserves the body-interleaved particle
order and nonzero snow compaction history. A capacity failure or late
material failure preserves the builder's prior size and leaves the
previously staged vector readable. The owner still enforces the
envelope byte limit and schema version.

The shared plastic wire fragment is in
schemas/simulate/plastic_solid_particle.fbs. A scratch FLSI envelope
based on M6 ba644ea included that fragment and added the optional
plastic_particles vector at field ID 6. Pinned flatc 25.12.19 passed
schema conformance, read a legacy size-prefixed payload with the
plastic vector absent, and preserved all selected fields of a
plastic-bearing size-prefixed record. The schemas, inputs and decoded
payloads are under out/m5-flsi-plastic.

The shared wire helper selects schema_version 2 when the plastic vector
contains records and retains version 1 for an absent field. The old
structural reader sees the version but ignores field 6. M6's current
native decoder rejects versions other than 1, preventing silent loss
of plastic state. The new codec should keep the version 1 default for
envelopes without plastic state, and reject a nonempty plastic vector
under version 1. Both scene suites exercise that rejection, successful
reads of legacy absence and a version 2 vector, and rejection of an
unknown version. This version policy is coordinated with M6. The
scratch schema establishes wire compatibility. It does not exercise
the production codec or an accepted mixed restart.

A detached combined worktree now exercises the owning C++ codec. Trial
commit 74d2e91 is based on the conflict-free tree of M5 602226e and M6
81cbd6a. Its accepted guided-piston fixture saves liquid, rigid pose
history, and compressed snow in FLSI version 2. Missing bindings and
inadmissible snow reject. The version 2 bytes round-trip exactly. The
liquid scheduler restored from those bytes reproduces the next accepted
envelope byte for byte after the uninterrupted scheduler rejects a
trial and retries. The fixture carries the saved snow history through
that step. Plastic evolution and reciprocal snow/liquid contact remain
open. Both milestone branches retain their separate owning sources.

In the detached worktree, the Windows Debug CPU build used the existing
CLI11 Conan toolchain, M4 CPU qualification receipt, and the reference
Python environment. The following run passed all 110 selected tests,
including the owning schema conformance and existing v1 replay tests.

```powershell
cmake --build --preset windows-debug-cpu --target fluidity_native_tests fluidity_m5_dry_scene_tests --parallel 8
ctest --preset windows-debug-cpu -R 'LiquidSubstep3D|RigidBody3D|M5DryScenes.native|LiquidCacheSchema.solid_interaction_checkpoint' --output-on-failure
```

Follow-up trial commit aac27ed advances the compressed snow with the
actual dry solver. A full-duration snow candidate moves and is discarded
when liquid acceptance fails. The checkpoint after rejection matches the
original version 2 envelope exactly. The retry advances both the
uninterrupted and restored snow records over the accepted half-step.
The resulting snow positions differ from the discarded full-duration
candidate, and the two accepted envelopes match byte for byte.
The focused native piston, legacy checkpoint, and M5 dry-scene checks
all passed. The fixture stages a separate dry update and liquid pressure
step. It exercises common checkpoint replay before reciprocal coupling.

Draft PR 84 carries the owning codec integration on the published M6
native-solids base with main 160e2fa. The five-file change is 354b0fd.
It supports a plastic payload in both solid-only and coupled FLSI
envelopes and shares the existing liquid field with M6. All seven
selected Windows Debug CPU tests passed, including schema conformance,
legacy rigid replay, and byte-identical joint liquid/plastic round trip.
The guided-piston scheduler changes remain in M6's stack. The detached
advancing-snow restart trial above supplies separate scheduler evidence.

The host mixed-contact bridge applies a supplied face batch to liquid
and plastic candidate records. It builds a separate global particle ID
order so the stored liquid sort order and plastic record order are
preserved. It checks aliasing and validates all final velocities before
publishing either span. The fixture exercises late face rejection,
final-velocity overflow, an empty batch and a successful retry. This
bridge is a host operation in both focused test binaries. It does not
provide pressure, swept detection or mixed accepted-step publication.

The planar mixed bridge now filters supplied reconstructed interfaces
using the same certified moving-halfspace sweep as wall contact. The
manufactured 0.25 m gap closes at 0.0625 s with a −4 m/s relative
normal velocity. At 0.125 s the returned hit fraction is 0.5 and the
pair impulse is 6 N·s. At 0.0625 s the hit is at the endpoint. A
0.03125 s attempt misses, and a later overlapping candidate rejects
without applying an earlier hit. This host bridge tests motion of
reconstructed planes. It still needs geometry from production particles,
post-impact position integration and pressure/acceptance coupling.

The bounded advection bridge uses the incoming and outgoing stored
velocities for a piecewise-constant drift with a common hit fraction.
The 0.125 s fixture advances from x = 0 and x = 0.25 m to the same
impact location, then to x = 0.1875 m. A 0.10 m path limit rejects
both candidate spans. The plastic sorted pose is preserved until the
next binning stage. Unequal hit times require another integrator.
This fixture does not exercise separate production stored and advecting
histories, evolving particle-derived geometry, or pressure coupling.

M6 commit 62100a6 adds isolated traction geometry for a finite box
intersected by a horizontal liquid plane. The supplied hydrostatic
pressure fixture balances buoyancy and weight. Production fractions,
moving wall flux, solved traction feedback and mixed replay are separate
from that geometry test.

The analytic mixed test now reconstructs its planar gap from particle
records instead of supplying plane coordinates independently. An
authored signed Cartesian normal selects the opposed extrema of the
face's contributors. Explicit offsets represent distance from particle
centers to the local interface. The fixture uses 0.0625 m on both
sides, yielding the same 0.25 m gap and half-step hit. At the end,
water is at x = 0.125 m and snow at x = 0.25 m, preserving the sum
of the two center offsets. Reversed body orientation yields the same
gap. Tied front samples with conflicting normal velocities reject.
This is a planar envelope over supplied face contributors. It does
not derive normals, footprint overlap or general free-surface geometry.

The mixed gate needs a versioned resolved scene carrying snow material
parameters, liquid pressure and distinct body memberships in the same
accepted substep. The standalone M5 face accumulator keeps separate mass and momentum by body
ID, and rejects a fifth body with `E-CONTACT-BODY-CAPACITY`. The production
step still needs to populate and consume those records transactionally.
The standalone face method applies a detected pair impulse to those
momentum records before reconstructing the outgoing velocities. It
leaves a third body untouched and preserves the whole face on failure.
The host face bridge now returns per-particle increments using the same
weights as the deposit. For a body with face mass M and impulse I, each
contributor with weight w receives w I / M, so particle mass times the
increments sums to I. The host batch bridge accepts a single vector-contact weight partition.
It resolves one supplied pair per face, verifies contributors against the
stable particle table and sums their increments in face-slot order.
The test checks that a later malformed face returns no staged output.
The bridge does not construct detection geometry, order simultaneous
pairs on one face, integrate pressure, or publish accepted mixed state. The coupled
scheduler must resolve contact with signed incoming relative normal speed,
apply each impulse once, preserve nonpenetration over the actual obstacle
sweep, and account for pressure traction and obstacle work without counting
the contact reaction again. Where partitioned pressure feedback is used,
record its lag and added-mass behavior.

The pressure-contact fixture runs the native or CUDA pressure solver on
the same frozen one-cell problem and then uses the host contact advance.
The sole outlet has unit area and a required flux of 2 m³/s. With
dt = 0.125 s, unit mobility, and unit gradient distance, the oracle
requires 16 Pa and an outlet velocity of 2 m/s. A fixture-only unit-weight
transfer gives that velocity to one liquid particle. The planar contact
occurs at half the step, transfers 6 N·s to snow, and gives both bodies
1 m/s after impact. Final liquid and snow x positions are 0.125 m and
0.25 m. The absolute tolerance is 1e-12 in each reported quantity's
units, including the pressure constraint defect in s⁻¹. A 0.10 m path
bound rejects the trial after projection and leaves the original particle
positions and velocities intact. The 0.25 m bound accepts the same trial.
The fixture uses a prescribed outlet and exercises one-way pressure input
to contact. Production still needs pressure traction feedback and the
accepted-state transaction below.

Run the fixture with both implementations.

```powershell
cmake --build --preset windows-release-cuda --target fluidity_m5_dry_scene_tests fluidity_cuda_m5_dry_scene_tests --parallel 6
ctest --preset windows-release-cuda -R '^M5DryScenes\.' --output-on-failure
```

The reciprocal pressure trial uses one full sealed wall, an incompressible
unit cell, and an open outlet with unit area and mobility. A wall moving
inward at speed u gives outlet flux u and pressure u/dt. The local solve
samples the stationary and unit-speed pressure responses, balances the
snow momentum against the pressure impulse, then projects at the solved
speed and checks the balance. The 0.25 kg snow sample entering at 10 m/s
solves to 2 m/s, 16 Pa, and −2 N·s along the inward direction. At the
solved wall speed the pressure work on snow is −4 J and the opposite
work on the liquid boundary is +4 J. The reflected high-face case
uses 2 kg at −3 m/s and gives −2 m/s, +2 N·s and the same work. Native
and CUDA pressure projection meet the 1e-12 absolute fixture tolerances,
and selecting an open face as the wall rejects.

The trial owns its frozen face copy and returns the candidate without
writing either material's state. The scheduler still needs to generate
the snow interface, distribute pressure traction over its contributors,
and advance and publish both materials together.

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
