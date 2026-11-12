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

The bounded planar solver landed in PR 86, the shared FLSI codec in
PR 84, and the joint replay regression in PR 88. The standalone
M5PlanarCoupling native and CUDA targets now run the moving-wall,
pressure, contact, acceptance and FLSI replay fixtures. The milestone
dry-scene target retains the sand and snow qualification cases.

| Gate | Status | Evidence |
|---|---|---|
| Granular constitutive invariants | Pass | Elastic compression, translated tensile apex, compressed shear return to the cone at fixed trace, plastic volume transfer, and invalid-state output preservation. |
| Snow constitutive invariants | Pass | Compression and tension clamps, reconstructed total volume, hardened corotational stress, repeated compaction, and invalid-state output preservation. |
| Full 3D material state and wave bound | Pass for the tested domain | Rotated and repeated stretches, reflection and singular rejection, isotropic and hardened wave oracles, and rejection of axial and principal-plane loss of ellipticity. Principal-plane checks do not certify ellipticity in every fully oblique direction. The Hessian row bound covers continuous elastic wave speed. The tested discrete acoustic factor is 0.25 for the bounded scenes. |
| Q-SOLID uncut quadrature | Pass | Analytic affine gradient gives 87.5 W particle power and −87.5 W grid power, with 3.25e-16 scaled residual on both backends versus 1e-6 allowed. This measures uncut stress quadrature and the matching gather derivative. |
| Q-SAND 30° pile | Open | The landed Coulomb deformation update reduces late timestep sensitivity. The nominal reference case reaches 32.825896° at 0.75 s and flattens to 29.113507° at 1.5 s. All four grid-width and timestep combinations miss the 30–35° range at the extended endpoint. |
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
| Reciprocal pressure for a planar snow interface | Pass for one- and two-cell frozen fixtures | Native and CUDA recover analytic flux, pressure, speed, and impulse with either wall orientation. The two-cell case reverses which pressure row is incident to the wall. An interior wall selection rejects. |
| Snow pressure traction scatter | Pass for one planar wall | Unequal-mass contributors receive the transpose-weighted pressure impulse. The recovered wall speed and total particle impulse match the native or CUDA pressure trial. Invalid geometry, a different body, and a failed pressure probe preserve particle state. |
| Joint planar pressure advection | Pass for one- and two-cell frozen fixtures | The projected regular-cell field updates liquid positions, volume and slopes. Snow advances with its pressure-corrected velocity. Crossing into an adjacent pressure cell succeeds. A late unsupported liquid destination or snow motion failure preserves both states. |
| Moving subcell snow geometry | Pass for a single-cell planar outlet | The next pressure attempt reconstructs the moved wall from snow positions and recomputes wet volume and outlet distance. Two reciprocal native and CUDA steps match the analytic pressures and trajectories. An inconsistent wet volume or a pressure-driven wall pose that disagrees with the advected snow front rejects without publication. |
| Joint planar pressure acceptance and retry | Pass for a moving cut-cell fixture | The liquid frame transaction publishes pressure, carrier velocities, particle state and clock before the staged snow record is copied back. A motion rejection preserves both committed states and yields a half-step retry. Both backends pass. |
| Planar snow–water frame driver | Pass for the bounded moving cut-cell scene | Reconstructs geometry between accepted steps, completes a frame after a rejected full step and two accepted half-steps, and publishes no frame when retries are exhausted. Native and CUDA pressure pass the same checks. |
| Planar snow-layer compression under water pressure | Pass for the one-cell affine layer fixture | Native and CUDA produce 12 Pa and reduce thickness from 0.25 m to 0.1875 m. The elastic Jacobian clamps at 0.975 and the combined Jacobian is 0.75. The frame rejects non-affine normal motion and preserves both input states. |
| Joint planar snow–water checkpoint replay | Pass for the bounded scene on native and CUDA | An authenticated initial artifact advances through pressure compaction. Disk reopen restores FLSI v3 at frame 10 and reproduces the uninterrupted frame-20 checkpoint byte for byte on each backend. |
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

### Plastic return activity during extended settling

Count a granular return if its accepted elastic map or log plastic
volume differs from the elastic trial beyond the existing 1e-12
relative state comparison. Count changes to log plastic volume
separately. The fractions weight each count by the actual step
duration and divide by total particle-seconds from 0.75 to 1.50 s.

| h (m) | Nominal dt (s) | Projected fraction | Plastic volume change fraction |
|---|---:|---:|---:|
| 0.075 | 0.001875 | 1.000000000 | 0.362751323 |
| 0.075 | 0.0009375 | 1.000000000 | 0.255079365 |
| 0.0625 | 0.0015625 | 0.984656085 | 0.362797619 |
| 0.0625 | 0.00078125 | 1.000000000 | 0.277777778 |

Native and CUDA give identical printed fractions and reproduce the
original extended angles. Granular returns continue during over
98% of the sampled particle-time even while the late angular drift
and momentum checks pass. Changes in plastic volume account for
25.5–36.3% of the particle-time. These counts measure activity and
do not measure plastic strain magnitude. Run the extended column
commands above to reproduce them.

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

### Plane-strain yield and lateral elastic response

The material's 30° friction parameter is calibrated to triaxial
compression. For a manufactured plane-strain path, impose normal
log compression e < 0 and engineering shear γ, with both lateral
normal log strains zero. Then normal traction is (λ + 2μ)e,
shear traction is μγ, and the squared deviator norm is
μ²[(8/3)e² + 2γ²]. The compressed cone yields when its square root
equals α(3λ + 2μ)|e|. These expressions give traction angles at
first yield of 15.9823° for (λ, μ) = (700, 450) Pa and 30.5082°
for (1280, 160) Pa. Both have the same 1600 Pa longitudinal
modulus. The native and CUDA constitutive fixture brackets each
analytic boundary with shear factors 0.999 and 1.001 and checks
that the plastic volume is preserved on shear return.

The column comparison uses the second Lamé pair and keeps the 30°
material friction, floor, particle sampling, and timestep schedule.

| h (m) | Nominal dt (s) | Baseline angle at 1.50 s | Altered Lamé pair at 1.50 s |
|---|---:|---:|---:|
| 0.075 | 0.001875 | 20.801830° | 20.410013° |
| 0.075 | 0.0009375 | 28.040080° | 27.797435° |
| 0.0625 | 0.0015625 | 20.580671° | 20.228874° |
| 0.0625 | 0.00078125 | 26.585383° | 26.407412° |

Native and CUDA match at the printed precision and pass the
vertical momentum and late-drift checks. Shifting the manufactured
plane-strain yield angle by about 14.5°
changes the late pile angle by less than 0.4° in all four cases.
All four remain below 30°. The manufactured loading path is useful
for distinguishing constitutive yield from the assembled pile response.

```powershell
.\out\build\windows-release-cuda\fluidity_m5_dry_scene_tests.exe --q-sand-plane-strain-moduli
.\out\build\windows-release-cuda\fluidity_cuda_m5_dry_scene_tests.exe --q-sand-plane-strain-moduli
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

### Coupled Coulomb gradient and APIC slope control

Use the floor projection derivative both in the post-impact
deformation update and on the stored APIC slope rows. The
deformation derivative follows the incoming advection velocity,
while the APIC derivative follows the incoming stored velocity.
Both use the same Coulomb branch as their respective velocities.
The pile geometry, material and floor coefficient stay fixed.

| h (m) | Nominal dt (s) | Baseline angle at 1.50 s | Projected slope angle | Late projected fraction |
|---|---:|---:|---:|---:|
| 0.075 | 0.001875 | 20.801830° | 29.083229° | 0.455211640 |
| 0.0625 | 0.0015625 | 20.580671° | 26.790663° | 0.407275132 |
| 0.075 | 0.0009375 | 28.040080° | 28.883736° | 0.637063492 |
| 0.0625 | 0.00078125 | 26.585383° | 26.915365° | 0.598330026 |

Native and CUDA agree at the printed precision and pass the vertical
momentum and late-drift checks. The control reduces the late plastic
return fraction to 41–64% of
particle-time and narrows the late timestep gap to under 0.21° at
both widths. The grid-width gap remains about 2°, and all angles
remain below 30°. The milestone branch uses the original solver.
To reproduce, check out 92df727 on codex/m5-sand-floor-control, rebuild
the two Release scene targets, and run the extended column commands above.

Run the two changes separately to identify the source of the reduced
late creep. The APIC-only control uses the original deformation
integration. The deformation-only control uses the original APIC
slope reset at impact.

| h (m) | Nominal dt (s) | APIC only at 1.50 s | Deformation only at 1.50 s | Deformation-only late projected fraction |
|---|---:|---:|---:|---:|
| 0.075 | 0.001875 | 20.759356° | 29.113507° | 0.457301587 |
| 0.0625 | 0.0015625 | 20.538690° | 26.761332° | 0.411904762 |
| 0.075 | 0.0009375 | 27.848481° | 29.006384° | 0.639761905 |
| 0.0625 | 0.00078125 | 26.408903° | 27.033753° | 0.602711640 |

The final native and CUDA angles match to six decimal places and
the momentum and late-drift checks pass. The APIC-only control
shifts the baseline by at most 0.20°. The deformation-only control
reproduces the combined angles within 0.13° and the reduction in
late plastic activity. The remaining grid-width sensitivity and
angles below 30° keep the gate open. Reproduce with 60f221a
(deformation only) or 1696516 (APIC only) on
codex/m5-sand-floor-control and the same extended column commands.

PR 89 landed the deformation-only update. Its standalone native
and CUDA dry solver suites compare the projected gradient against
central differences of the actual wall solver, covering sliding,
sticking, frictionless response and separation. The angular and
plastic activity measurements above remain the extended Q-SAND
qualification for the landed change. Earlier baseline measurements
in this report use the original floor deformation update.

### Dense particles with the landed floor update

Double the particle sampling in each direction from a 0.05 m to a
0.025 m lattice. The same column volume and total mass are now
represented by 504 particles instead of 63. Use the landed PR 89
floor update and repeat the centered 1.50 s settling cases without
changing the grid widths, timesteps, or envelope estimator.

| h (m) | Nominal dt (s) | 63-particle envelope | 504-particle envelope | Dense quadrant fit |
|---|---:|---:|---:|---:|
| 0.075 | 0.001875 | 29.113507° | 26.634632° | 15.398842° |
| 0.0625 | 0.0015625 | 26.761332° | 25.002454° | 21.925421° |
| 0.075 | 0.0009375 | 29.006384° | 26.113921° | 6.973853° |
| 0.0625 | 0.00078125 | 27.033753° | 23.529643° | 13.925098° |

Native values are shown. CUDA differs by at most 0.000002° at the
printed precision. The drift and vertical momentum checks pass on
both backends. Increasing the sampling density lowers the envelope
in every case.
The denser pile spreads farther even though its final peak height
is larger. All dense quadrant fits are shallower than the envelope,
so the dense sampling and independent estimator both leave the
repose gate open.

```powershell
.\out\build\windows-release-cuda\fluidity_m5_dry_scene_tests.exe --q-sand-dense-column-long-settle
.\out\build\windows-release-cuda\fluidity_cuda_m5_dry_scene_tests.exe --q-sand-dense-column-long-settle
```

### Dense sand at three seconds

Continue the centered 504-particle column drop to 3.0 s at nominal step
h/80. Keep the original 0.70 s and 0.75 s sample boundaries. The extended
run takes its final drift sample at 2.95 s and does not add a substep
boundary at the separate 1.45 s or 1.50 s endpoints. Apply the same
acceptance checks and measure yield activity over the final 0.05 s.

| h (m) | Nominal dt (s) | Envelope | Final drift | Kinetic energy (J) | Projected fraction | Plastic volume change fraction |
|---|---:|---:|---:|---:|---:|---:|
| 0.075 | 0.0009375 | 21.853351° | 0.054106° | 0.000019 | 0.370040 | 0.370040 |
| 0.0625 | 0.00078125 | 20.623778° | 0.036452° | 0.000031 | 0.333333 | 0.325397 |

Native and CUDA match at the displayed precision. The fractions are
time-weighted particle counts over the window, using the solver's
existing 1e-12 scaled change threshold. They measure how frequently the return modifies the state. The final quadrant fits were 14.44° and 18.08°.
Both runs met the momentum bound on both backends. The low kinetic
energy and small final envelope drift coexist with continuing plastic
updates and a surface shallower than the Q-SAND window.

The mean signed change in log plastic Jacobian over the same window is
0.002756145 and 0.002007689 for the 0.075 m and 0.0625 m grids. Summing
absolute per-step changes gives the same values to the printed precision,
showing sustained dilation. Changes ending within one particle spacing
(0.025 m) of the floor contribute 61.4% and 57.1% of the respective absolute
sums. This is a share of total change across all particles. The measurement
also shows dilation above the floor layer. Native and CUDA differ by at
most 3e-10 in the mean signed change.

~~~powershell
.\out\build\windows-release-cuda\fluidity_m5_dry_scene_tests.exe --q-sand-dense-three-second
.\out\build\windows-release-cuda\fluidity_cuda_m5_dry_scene_tests.exe --q-sand-dense-three-second
~~~

### Exact volume change under a frozen velocity gradient

The specified solid update uses the Euler increment I + dt L. For a
frozen velocity gradient the exact increment has log determinant
dt tr(L). A temporary control isolates this volume error by scaling
each Euler trial map by exp((dt tr(L) - log det(I + dt L)) / 3).
The positive scalar changes the determinant while preserving principal
directions and deviatoric log strain for the given trial. Apply the
same rule to each part of a split floor-impact step.

| h (m) | Baseline envelope | Corrected envelope, native | Baseline mean log plastic change | Corrected mean log plastic change, native |
|---|---:|---:|---:|---:|
| 0.075 | 21.853351° | 21.861961° | 0.002756145 | 0.002744017 |
| 0.0625 | 20.623778° | 20.633854° | 0.002007689 | 0.002016616 |

CUDA matches the corrected envelope within 0.000001° at printed
precision. Momentum and late-drift checks pass on both backends. The
angle changes by less than 0.011°, and the late plastic dilation changes
by less than 0.5%. The measured dilation persists when the trial volume
follows the trace of the frozen gradient exactly.

To reproduce from 5e4201e, insert the scalar correction before the return
in TrialDeformation in native/src/dry_solid_step_3d.cc. Use the same
velocity gradient and duration that produced that trial. Rebuild both dry
scene targets and run the three-second commands above. The production
source uses the specified Euler update.

### Affine transfer feedback in the dense pile

Run the same centered three-second cases with two temporary changes to
the stored granular velocity slopes. The first sets the entire affine
fit to zero before the next deposit. The second subtracts one third of
the fit's trace from each diagonal entry. Both controls retain the
actual gathered velocity gradient used for deformation, the constitutive
law, and the wall update. They change only the affine momentum feedback.

| h (m) | Baseline envelope | No affine envelope | Trace-free envelope | Baseline late dilation | No affine late dilation | Trace-free late dilation |
|---|---:|---:|---:|---:|---:|---:|
| 0.075 | 21.853351° | 44.659101° | 16.183305° | 0.002756145 | 0.000107975 | 0.001766199 |
| 0.0625 | 20.623778° | 42.931884° | 16.503923° | 0.002007689 | 0.000309803 | 0.001935966 |

The dilation values are mean signed changes in log plastic Jacobian
over the final 0.05 s. Native values are shown. CUDA envelope results
differ by at most 0.000003° at printed precision. Momentum and late-drift
checks pass. With the fit set to zero, all measured late dilation occurs
within 0.025 m of the floor. Removing only the trace makes the piles
shallower, with dilation still present above the floor layer. The
combined response establishes sensitivity to affine feedback. It does
not yet isolate the effects of the symmetric deviatoric and skew parts.
Neither control meets the repose gate.

To reproduce from f053966, adjust the granular assignment of
particle.velocity_slopes at the end of the dry step in
native/src/dry_solid_step_3d.cc as described above. Rebuild both dry
scene targets and run the three-second commands. The production
solver uses the complete affine fit.

### Symmetric and skew affine transfer controls

Repeat the three-second cases with only the symmetric part of the
granular affine fit, then with only its skew part. For a fitted matrix
C, store (C + Cᵀ)/2 in the first control and (C - Cᵀ)/2 in the second.
Make the substitution after the normal slope assignment in the dry
step. Both controls retain the true gathered deformation gradient,
pressure-dependent return, and swept floor response.

| h (m) | Symmetric envelope | Skew envelope | Symmetric late dilation | Skew late dilation |
|---|---:|---:|---:|---:|
| 0.075 | 23.063512° | 44.406647° | 0.001414746 | 0.000120151 |
| 0.0625 | 23.065463° | 42.273166° | 0.000862563 | 0.000353422 |

Native values are shown. CUDA agrees within 0.000003° at printed
precision. Both backends meet the momentum and late-drift checks.
The symmetric fit reproduces most of the spreading from the complete
fit. The skew fit finishes within 0.66° of the no-affine result.
The trace-free control above had spread farther than the complete fit,
so disabling the isotropic part alone does not account for the response.
Neither split reaches 30–35°. These controls change the mechanics of
the affine transfer and serve as diagnostics. The dry solver uses the
complete fit.

Reproduce from 1df1815 with the corresponding substitution in
native/src/dry_solid_step_3d.cc. Rebuild both dry-scene targets and run
the three-second commands above.

### Vertical grid alignment at the sand floor

Repeat the 504-particle column drop to 1.5 s with nominal step h/80.
Translate the grid origin vertically while keeping the floor plane, column,
and gravity fixed. This changes the location of the MAC samples relative
to the floor without changing the physical contact or the estimator.

| h (m) | Grid origin y/h | Native envelope | CUDA envelope |
|---|---:|---:|---:|
| 0.075 | 0.00 | 26.113921° | 26.113920° |
| 0.075 | 0.25 | 26.149408° | 26.149408° |
| 0.075 | 0.50 | 24.685605° | 24.685605° |
| 0.0625 | 0.00 | 23.529643° | 23.529643° |
| 0.0625 | 0.25 | 23.791141° | 23.791141° |
| 0.0625 | 0.50 | 22.529780° | 22.529780° |

At each grid width a half-cell shift lowers the envelope. The largest
span across vertical alignments is 1.47°, and the steepest envelope is
still 3.85° below the lower acceptance bound. Maximum late drift is
0.343°. All runs satisfy the vertical momentum bound of omitted stress
impulse plus 1e-6 N·s. Both commands exit 1 because none of the six cases
reaches the repose window. These measurements quantify alignment
sensitivity while leaving the cause of the remaining shallow slope open.

~~~powershell
.\out\build\windows-release-cuda\fluidity_m5_dry_scene_tests.exe --q-sand-floor-grid-phase
.\out\build\windows-release-cuda\fluidity_cuda_m5_dry_scene_tests.exe --q-sand-floor-grid-phase
~~~

### Low-mass stress acceleration control

The vertical alignment runs omit 0.000034–0.000060 N·s of stress impulse
out of 34.26–41.49 N·s proposed on native. To test whether this small
aggregate nevertheless changes the result, repeat the same six cases
with the stress acceleration denominator set to deposited face mass.
Leave the empty-face check, motion bound, and transfer settings intact.

| h (m) | Grid origin y/h | Native envelope | CUDA envelope |
|---|---:|---:|---:|
| 0.075 | 0.00 | 26.113421° | 26.113421° |
| 0.075 | 0.25 | Face motion bound at step 270 | 26.149349° |
| 0.075 | 0.50 | 24.685833° | 24.685833° |
| 0.0625 | 0.00 | Face motion bound at step 1693 | 23.529277° |
| 0.0625 | 0.25 | Face motion bound at step 727 | Face motion bound at step 727 |
| 0.0625 | 0.50 | 22.529888° | 22.529889° |

Every completed case stays within 0.0006° of its regularized baseline and
within 1e-6 N·s of the vertical momentum balance. All completed angles
are below 30°. The motion rejections show the safeguard matters to the
fixed-step fixture even though its effect on completed angles is small.

To reproduce from 9996ba1, temporarily change the assignment of
force_mass in native/src/dry_solid_step_3d.cc from
std::max(mass, minimum_force_mass) to mass. Rebuild both dry-scene targets
and run the two --q-sand-floor-grid-phase commands above. Restore the
original assignment and rebuild before running the normal suite.

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

### Compatible floor stress quadrature

The M6 elastic blade and M5 floor fixture now share one accumulator for
reference-volume stress forces on the normalized compatible stencil.
This keeps force assembly on exactly the same face support as mass and
affine momentum deposition. The floor fixture excludes face anchors
below y = 0 and uses a wall-compatible affine field with normal velocity
3y. Native and CUDA independently form the compatible stencil and
gather the complete velocity gradient. Force assembly runs on the host
for both transfer paths.

The analytic particle work is -2.125 J. Both paths give 3.42852101 J of
grid stress work and 1.30352101 J of cut work. The full particle work
plus grid work agrees with the cut work to 1e-11 J. The normalized
forces match the independent cut oracle within 1e-11 N per face,
with zero force below the floor and a net stress impulse within
1e-11 N·s. No ghost fit falls back in the new history. The nonzero
cut work measures the difference between the complete gather and
compatible deposition. The pile solver still needs a paired
compatible floor transfer and contact budget.

Run the native and CUDA M5DryScenes tests for the floor fixture, and
the MacTransfer3D elastic blade cases for the shared M6 caller.

### Compatible floor transfer in the dry step

The opt-in granular floor path uses compatible affine deposition, the
matching normalized stress derivative, and the transpose of compatible
mass weights for PIC velocity. The complete same-side ghost gather
supplies the true deformation gradient and the APIC fit. Rank fallback
disables the affine fit. The stationary floor still resolves normal and
friction impulses through the swept particle contact. The grid leaves
the incident velocity available to that sweep. The mode uses PIC stored
velocity and the bounded boundary-transfer capacities.

When old and new gather histories borrow the same validated velocity,
validity, compatibility and wall arrays, the host and owning CUDA paths
reuse the continued sample. Stage-time validation and accumulation order
remain intact. The cut-floor fixture compares shared and independently
copied samples on each backend and requires identical velocities,
affine fits, derivatives, residuals, and fallback flags. The existing
distinct-history boundary and elastic-blade fixtures also pass.

At 0.75 s the 63-particle column gives the following results. Native
and CUDA print the same angles and cut work at this precision.

| h (m) | dt (s) | Envelope | Late drift | Cut stress work (J) | Vertical momentum residual (N·s) |
|---|---:|---:|---:|---:|---:|
| 0.075 | 0.001875 | 35.186895° | 0.819790° | 0.036394718 | -0.000013837 |
| 0.0625 | 0.0015625 | 32.361234° | 0.611646° | -0.029489823 | -0.000100054 |

Both cases meet the kinetic energy, late drift, and vertical momentum
bounds. The coarse angle exceeds 35°. The one-particle fixture checks
the original analytic Coulomb impulse on the new transfer path and
zero cut work for zero stress. Cut work measures the difference between
the complete pre-contact deformation field and the normalized stress
deposition. It does not include work from the subsequent particle sweep.

The 1.50 s native sweep gives the following results at both timesteps.
The vertical momentum and late-drift bounds still pass. The coarse
half-step case enters the angle window. The other three miss it.

| h (m) | dt (s) | Envelope at 0.75 s | Envelope at 1.50 s | Late drift | Cut stress work (J) |
|---|---:|---:|---:|---:|---:|
| 0.075 | 0.001875 | 35.186895° | 29.962967° | 0.114904° | -0.097516182 |
| 0.075 | 0.0009375 | 42.374498° | 32.042153° | 0.400340° | 0.020710864 |
| 0.0625 | 0.0015625 | 32.361234° | 27.424648° | 0.223715° | -0.111295504 |
| 0.0625 | 0.00078125 | 37.170504° | 29.277572° | 0.354377° | -0.026814291 |

The measured slope remains sensitive to the timestep and the stopping
time. The floor path remains opt-in, and Q-SAND remains open.

~~~powershell
.\out\build\windows-release-cuda\fluidity_m5_dry_scene_tests.exe --q-sand-compatible-floor
.\out\build\windows-release-cuda\fluidity_cuda_m5_dry_scene_tests.exe --q-sand-compatible-floor
.\out\build\windows-release-cuda\fluidity_m5_dry_scene_tests.exe --q-sand-compatible-floor-long-settle
.\out\build\windows-release-cuda\fluidity_cuda_m5_dry_scene_tests.exe --q-sand-compatible-floor-long-settle
~~~

The dry-step budget now also measures the signed translational kinetic
energy change across the particle contact sweep. The analytic fixture
loses 0.875 J from a 1 J initial state. It passes on native and CUDA
with either floor transfer. The two native 0.75 s compatible pile
runs lose 1.66679019 J and 1.51219119 J through the sweep. These losses
are separate from cut stress work and exclude affine kinetic energy.

### Compatible PIC derivative control

Temporarily replace the complete ghost derivative with the derivative
of the normalized compatible PIC field. Keep the ghost APIC fit and
particle sweep. The stress deposit and the tested deformation derivative
then use the same weights, so the pre-contact cut-work mismatch vanishes
to roundoff.

| h (m) | dt (s) | Baseline 1.50 s envelope | Paired derivative 1.50 s envelope | Paired late dilation |
|---|---:|---:|---:|---:|
| 0.075 | 0.001875 | 29.962967° | 28.407709° | 0.005784775 |
| 0.075 | 0.0009375 | 32.042153° | 31.915729° | 0.000471606 |
| 0.0625 | 0.0015625 | 27.424648° | 26.082740° | 0.002049083 |
| 0.0625 | 0.00078125 | 29.277572° | 29.126156° | 0.002037700 |

These native runs met the momentum and late-drift checks. Late
dilation fell in all four cases, but the final angles also fell and
three cases still missed the gate. The production compatible path
retains the complete derivative. To reproduce from 999366f, compute
each deformation row as the sum of normalized weight gradient times
face speed in the compatible PIC loop. Use that same row in the
cut-work receipt and run the extended compatible floor command.

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

The initial PR 84 trial carried the owning codec integration on the published M6
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
cmake --build --preset windows-release-cuda --target fluidity_m5_planar_coupling_tests fluidity_cuda_m5_planar_coupling_tests --parallel 6
ctest --preset windows-release-cuda -R '^M5PlanarCoupling\.' --output-on-failure
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

The pressure solve also accepts connected MAC cells within the native
solver capacity. The selected full sealed wall must touch exactly one
pressure cell. Its incident pressure row supplies traction. With two
unit cells in series, each open face carries 2 m³/s and each link
drops 16 Pa. A 2 kg wall entering at 4 m/s solves to 2 m/s with
32 Pa incident pressure and −4 N·s impulse. Reversing the domain puts
the 32 Pa in the second row and reverses the impulse. Native and CUDA
match the analytic values within 2e-6 in the measured units. Selecting
the sealed face between the two cells rejects. The scatter uses the
same pressure solve.

The pressure scatter uses area fractions w to interpolate the wall
velocity. Its inverse effective mass is Σ(w²/m). Applying the
transpose weights gives each contributor velocity change w I/m for
face impulse I, recovering the wall velocity used by projection.
For two equal-area contributors of mass 0.125 and 0.0625 kg, the
incoming speeds 12 and 16 m/s produce an effective mass of 1/6 kg.
The trial gives −2 N·s total impulse, particle speeds 4 and 0 m/s,
and a weighted wall speed of 2 m/s. Both backend fixtures recover
these values within 1e-12. Tangential velocities, positions, plastic
history and the third body are preserved. Closing the liquid outlet
makes the unit-speed pressure probe fail. The staged trial publishes
no snow velocities in that case.

The joint candidate uses the production regular-cell velocity and
divergence evaluator for liquid and a certified translating-wall sweep
for each material. A single snow contributor at x = −1.125 m enters
at 3 m/s and solves to 2 m/s. Liquid at x = −0.75 m also moves at
2 m/s. After 0.125 s the two positions are −0.875 m and −0.50 m,
and the liquid volume ratio stays 0.95. The 0.30 m path budget
accepts these paths. Adding a 3 m/s tangential snow velocity exceeds
the budget after the liquid candidate is staged. The rejection
preserves both positions and velocities. The fixture passes with
native and CUDA pressure to 1e-12 absolute tolerance.

The connected two-cell fixture starts liquid at x = −0.125 m.
At 2 m/s it crosses the full shared face at x = 0 and finishes at
x = 0.125 m. The starting cell's regular velocity field has zero
divergence and preserves the liquid volume ratio of 0.95. The snow
position advances to −0.875 m. A second liquid particle would cross
the far outlet into an unsupported cell. That late failure preserves
the first liquid particle and the snow candidate. Native and CUDA
pressure pass with absolute tolerance 2e-6 in the measured units.
This host trial permits one crossing through an open shared face.
The full scene still needs moving snow geometry, general paths and
scheduler acceptance.

The subcell fixture reconstructs the snow front at x = −0.75 m
from its particle and 0.125 m surface offset. The initial wet
volume is 0.75 m³ and the outlet distance is half the wet thickness.
The first step gives 2 m/s and 6 Pa. Reconstruction from the moved
snow gives x = −0.50 m and 0.50 m³ at the next step. The second
step gives 1.2 m/s and 2.4 Pa, and the liquid volume ratio stays 0.95
within the 2e-7 absolute tolerance used for the mixed-precision solve.
Both native and CUDA pass. A supplied plane inconsistent with the
wet volume rejects without changing either state. This geometry is
limited to a planar snow front in one cell with stationary sealed
transverse faces and a full ambient-pressure outlet.

The host attempt adapter uses the existing liquid frame transaction
with the snow records staged separately. It verifies the committed
grid, cells and face slots and starts the solve from committed cell
pressure. In the cut-cell fixture, a 0.20 m motion bound rejects the
0.125 s attempt and leaves the accepted clock, pressure, carrier and
particle values unchanged. The 0.0625 s retry commits 12 Pa, the
2 m/s wall carrier, liquid x = −0.375 m and snow x = −0.750 m.
The accepted ordinal and epoch each advance once. Native and CUDA
pressure pass the same 1e-12 absolute fixture checks. The adapter
uses a preallocated liquid transaction and host scratch for the
coupled stage. The production scheduler still needs to select and
drive this geometry from the authored scene and own the joint
checkpoint.

The bounded frame driver owns the liquid transaction and snow
candidates until frame completion. It rebuilds the front after each
accepted step and uses the transaction's prescribed retry duration
within each frozen attempt. With a 0.125 s frame and 0.20 m motion
limit, the full step rejects and two 0.0625 s steps complete the
frame. The final accepted pressure is 60/11 Pa and the wall speed
is 12/11 m/s. The final accepted step and epoch are two, the frame
ordinal is one, and the substep ordinal resets to zero. Both
pressure backends match within 2e-6. A 0.01 m limit exhausts retries
and returns no frame while preserving the caller's inputs. The
driver uses explicit planar scene settings and supports liquid
without sources or cavities. It does not yet serve as a resolved
Blender scene runtime or write the accepted joint checkpoint.

The combined branch codex/m5-planar-restart at 18cffb5 adds the
owning FLSI codec from PR 84 to the coupled planar driver. It
writes the completed 0.125 s frame with liquid state and nonzero snow
compaction history. A resumed attempt with an exhausted motion limit
leaves the saved bytes intact. With the original limit, both the
uninterrupted and restored runs advance snow and liquid to 0.1875 s.
The two next envelopes match byte for byte on each backend, including
accepted step and epoch three and frame ordinal two. The Release
native and CUDA scene suites and FLSI conformance test passed.

The landed fixture runs with the standalone planar targets and the
owning schema conformance test.

```powershell
cmake --build --preset windows-release-cuda --target fluidity_m5_planar_coupling_tests fluidity_cuda_m5_planar_coupling_tests --parallel 8
ctest --preset windows-release-cuda -R '^M5PlanarCoupling\.|^LiquidCacheSchema.solid_interaction_checkpoint' --output-on-failure
```

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

### Pressure-driven planar snow deformation

The optional snow material enables normal strain for one snow body. The
pressure contributors must lie on the front layer. All normal particle
displacements must fit the affine thickness change within 1e-10 cell widths.
The driver applies the normal stretch to the elastic trial deformation and
uses the snow constitutive return to update the elastic map and log plastic
Jacobian in the same frame transaction.

The analytic fixture advances 0.0625 s with the front and back particles
0.25 m apart. The pressure reaction reduces front speed from 4 to 2 m/s
while the back advances at 3 m/s. The final coordinates are -0.75 m and
-0.9375 m. The pressure is 12 Pa. The elastic Jacobian is 0.975 and the
product of the elastic and plastic Jacobians is 0.75. The checks use
absolute tolerances of 1e-12 in the respective units. An incompatible
interior particle exhausts retries and preserves the input states.

The restart fixture encodes that computed two-particle state at 0.0625 s
through the shared FLSI codec. Advancing decoded and uninterrupted state
to 0.125 s yields byte-identical checkpoints within each backend.

The compaction frame uses the accepted snow stress in the next step.
It integrates reference volume times the normal Kirchhoff stress,
then projects the internal force onto the affine normal mode about
the snow center of mass. For a normal coordinate q, the force on a
particle is -m (q - q_center) times integrated stress divided by
the mass moment of inertia in that coordinate. Its total impulse is
zero, and its work on an affine velocity field matches the stress
power with opposite sign. The predictor rejects a dimensionless
modal timestep above one using the existing frozen acoustic bound.
The stiff-layer fixture exercises the resulting half-step retry.

The analytic replay uses 12 Pa shear modulus and 7 Pa Lamé lambda
to keep its 0.0625 s steps within the mode limit. The first step
starts stress-free. At the second endpoint the recovered pressure
is 5.463767 Pa, the thickness is 0.06850841 m, and the cumulative
elastic–plastic Jacobian is 0.27403365. The elastic Jacobian stays
at 0.975. Native and CUDA check these values and the net snow
momentum change against the pressure impulse. The clamp tolerance
is 1e-12. The other second-frame checks use 2e-6 in their respective
units. This is a one-dimensional affine mode with a frozen tangent
step limit. General three-dimensional stress transfer and stability
remain part of the production gate.

At each accepted substep, including the final one, the frame rebuilds the
wall from the moved snow and checks that it stays in the selected cell.
The reconstructed pose must agree with the integrated pressure-driven
trajectory within the step duration times the existing scatter speed
tolerance, plus 1e-10 cell widths. The rear-only traction regression
previously accepted a moving liquid wall while the foremost snow particle
stayed still. Native and CUDA now reject that frame and preserve the inputs.

The high-face reflection also gives 12 Pa, a 0.975 elastic Jacobian,
and a 0.75 combined Jacobian. The reflected liquid and snow positions
agree with their analytic coordinates to 1e-12 m. The rear-only traction
case rejects with either orientation. A separate low-face test sets
the initial front speed to -12 m/s. The liquid stays in its pressure
cell but the snow wall retreats past the low face on the last substep.
Both backends reject and preserve the input states.

The frame also checks the traction list before each pressure solve. Each
contributor must lie within 1e-10 cell widths of the signed reconstructed
front. A stationary rear-only contributor used to pass because it produced
zero wall displacement. It now rejects before projection. The reflected
fixture checks the same front requirement from the opposite side. A
separate fixture starts with two particles tied at the front and applies
traction to only the retreating particle. The final pose check still
rejects the split front and preserves both input states.

The planar transaction preserves each liquid particle's affine enable
flag while completing PIC initialization. The fixture starts with affine
transfer disabled and carries that state through acceptance and checkpoint
replay on both backends. A pressure row whose committed cell is marked as
air rejects before the trial begins, preserving the occupancy and clock.

The shared plastic binding validator checks material parameters at unit
elastic stretch before examining the particle payload. This also checks
a resolved body with no particles. Native and CUDA checkpoint tests reject
an invalid snow compression limit on empty payload encode and decode, with
the encode builder unchanged.

Resolved scene model 2 adds the plastic-body binding table to FLRS. It
carries body IDs, material IDs, and one typed granular or snow material per
entry. Model 1 retains its existing canonical bytes and fingerprint. The
codec validates ID ordering, material admissibility, and the version and
payload agreement. The liquid-only CLI rejects model 2 before creating a
cache. The cache fixture feeds the decoded bindings into shared FLSI
encode and decode and recovers the snow particle's plastic Jacobian.

The planar compaction fixture now passes the decoded model 2 snow binding
into the pressure frame and uses the same binding for FLSI save and resume.
The driver validates the particle body and material IDs before projecting.
A material ID mismatch rejects without advancing either input state.
Native and CUDA both reproduce the pressure and compaction oracle and the
byte-identical second-frame replay.

M6's active rotating-box branch at 2ab2983 includes bd06a0f, which stages
reciprocal CPU particle contact in the production liquid scheduler's
oriented-box attempt. That accepted-step boundary is relevant to M5's
scheduler integration. The box-specific impulse update has no plastic
particle or snow constitutive path. The M5 binding change uses the shared
FLSI interface and leaves that branch's scheduler files untouched.

The resolved compaction replay now uses FLSI v3. It embeds the canonical
FLRS fingerprint and requires the caller's scene to match before restoring
either material. Geometry, random seed, and accepted-step ordinals also
agree before publication. The test changes snow hardening while keeping
the saved particle admissible. The scene-bound decoder rejects the change,
and native and CUDA replay byte-identical second-frame checkpoints when
the scene matches. Existing unbound FLSI v1 and v2 callers retain their
codec. The cache now stores the scene-bound bytes as one content-addressed FLSI
blob with liquid scheduler resume metadata nested alongside the liquid
state. The existing checkpoint reference ties both materials to the
committed liquid point frame. An interruption after manifest publication
leaves the prior mixed checkpoint readable. Repair publishes the new
reference, and retention keeps the newest two mixed payloads. The cache
fixture checks all three cases and verifies the saved snow strain.

The bounded mixed compaction fixture now starts from a canonical unbound
FLSI initial artifact. The resolved FLRS scene names that artifact by
length and digest, and the loader verifies those exact bytes before
decoding. Geometry, seed, time-zero clock, source counters, and plastic
material bindings must agree. Native and CUDA reject a byte change and
a mismatched scene seed, then advance the verified initial state through
compaction and scene-bound restart. Initial state uses the unbound
format because the FLRS fingerprint already includes the initial
artifact digest. The CLI still gates mixed execution pending production
scheduler integration.

The disk fixture publishes all twenty completed compaction frames and
the scene-bound checkpoints at frames 0, 10, and 20. It starts from
the authenticated mixed input, opens the cache for replay at frame 10,
and verifies the recovered liquid, resume metadata, and compressed snow.
The resumed state drives frames 11–20. Its persisted FLSI bytes match
the uninterrupted run within each backend. The fixture uses the same
bounded affine snow layer and pressure geometry as the analytic tests.

The authored Blender snow–water scene still requires production liquid
integration with general moving-interface reconstruction and internal snow
stress evolution. Run both planar fixtures with the build and CTest
commands in the checkpoint replay section above.

The CLI authenticates a supplied mixed initial artifact before reporting
the scheduler gate. A same-length byte change returns `DATA_LOSS` on
`initial_state`. A valid artifact reaches the scheduler gate, and both
paths leave the cache unpublished. The `SimulateCli.execution_validation`
test covers this preflight.

Pass one case name after the extended qualification flag to use a fresh
process for each CUDA case.

```powershell
foreach ($case in 'fine', 'coarse', 'coarse-half', 'fine-half') {
  .\out\build\windows-release-cuda\fluidity_cuda_m5_dry_scene_tests.exe --q-sand-compatible-floor-long-settle $case
}
```
The nominal CUDA cases reproduced the native measurements. Each process
returned a failed qualification because the final angle was below 30°.

| Case | Final angle | Late drift | Vertical momentum residual (N·s) |
|---|---:|---:|---:|
| Fine | 27.424648° | 0.223715° | -0.000100054 |
| Coarse | 29.962967° | 0.114904° | -0.000015571 |

The milestone branch includes upstream `db4636d`, which adds the M6
rigid-frame path and liquid gather and tank-contact momentum accounting.
After rebuilding the CLI and fixture, `SimulateCli.rigid_frame`,
`SimulateCli.cpu_execution`, and `SimulateCli.execution_validation` all
passed. All six `M5*.native` CTest targets also passed in the integrated
build. The CUDA planar coupling, plastic checkpoint, snow compaction,
and constitutive targets also passed after rebuilding. The Windows
incremental build missed header dependencies in the old piston and snow
test objects. Recompiling those translation units resolved the stale
link symbol and the stale budget layout in the rollback fixture.
The integrated rigid path retains the mixed initial-state gate.
The remaining production evidence is an authored Blender snow–water bake
with general moving-interface reconstruction, reciprocal pressure and
contact through the liquid scheduler, internal snow stress evolution, and
joint restart through that same execution path.
