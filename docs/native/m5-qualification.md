# M5 qualification on Windows

Local branch `codex/m5-granular-snow`, based on verified upstream
`549fb3f844841b79d14b7beb9ecfb6db9727c976`. This report covers the bounded
M5 implementations against §§2.3, 5.2, 8.1–8.2 and 15.2 of the
[design](../superpowers/specs/2026-11-07-fluidity-design.md).

The branch includes upstream main through 6472b77. Constitutive models,
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
| Q-SAND 30° pile | Open | The six-second sparse baseline fails on native and SM120. Only the coarse nominal angle is in range. The three other angles are 23.42°, 17.10°, and 13.06°. All four dense native cases miss angle and radius at three seconds. The native compatible-floor path also fails three of four angles at six seconds. |
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
| Production distinct-body water coupling | Pass for the authored uniform snow–water scene | The joint pressure projection includes reciprocal snow mobility, verifies the backend flux and accepts both materials in the same step. Native and SM120 reproduce the joint checkpoint. |
| Authored Blender snow–water scene | Pass | Blender 5.2.3 bakes and reopens 20 frames with the Snow preset, animated platen, emitted water and separate materials on native and SM120. The frame-20 checkpoint records irreversible compaction. |

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
All four CUDA cases reproduced the native measurements. The coarse
half-step case passed. The three other cases missed the angle band.

| Case | Final angle | Late drift | Vertical momentum residual (N·s) |
|---|---:|---:|---:|
| Fine | 27.424648° | 0.223715° | -0.000100054 |
| Coarse | 29.962967° | 0.114904° | -0.000015571 |
| Coarse half-step | 32.042153° | 0.400340° | -0.000118859 |
| Fine half-step | 29.277572° | 0.354377° | -0.000040270 |

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

The resolved mixed input and joint checkpoint changes landed in
[PR 94](https://github.com/tychota/FluidiTy/pull/94) at `37dc107`.
A clean Windows Release CPU build of the standalone PR passed all 81
selected cache, schema, plastic checkpoint, and CLI regressions. The
merge into the milestone branch preserved its tested source tree.

### Sampled solid geometry for the pressure grid

The shared clipper uses signed liquid and solid samples at lattice
corners. Each of the pressure grid's six tetrahedra interpolates both
fields linearly. Wet volume lies inside the liquid and outside the solid.
A shared face uses the same two triangles and four solid samples in
either incident cell. A zero solid plateau is closed.

For a unit cell pair and the oblique solid boundary x + y = 1.25,
the wet volumes below y = 0.75 are 0.125 and 0.71875 m³.
Both cells give exactly the same 0.75 m² aperture within each backend.
Native and SM120 agree with the analytic volumes and aperture within
2e-14 in their respective units. A native fixture with only the high
corner exterior checks non-affine samples, giving 1 m³ volume and
1 m² on each high face of a 2 m cell. All 37 native boundary geometry
tests and the CUDA M5 invariant executable pass.

```powershell
ctest --test-dir out/build/m5-resolved-clean -R '^LiquidBoundaryGeometry3D\.' --output-on-failure
ctest --test-dir out/build/windows-release-cuda -R '^M5Constitutive\.cuda$' --output-on-failure
```

The common pressure-grid freeze accepts sorted solid corner samples and
one signed wall-normal velocity per owned face slot. The oblique fixture
produces two pressure cells with eleven distinct faces. The shared face
carries one 0.75 m² aperture and 0.2 m/s wall-normal velocity, giving
0.05 m³/s blocked-area wall flux in the internal pressure link.
A missing solid corner rejects before publishing frozen geometry.
The owning and arena-backed SM120 paths return identical cell volumes,
face slots, apertures, and wall velocities. All 74 native geometry and
workspace checks and six CUDA boundary checks pass.

```powershell
ctest --test-dir out/build/m5-resolved-clean -R '^LiquidBoundary(Geometry|Workspace)3D\.' --output-on-failure
ctest --test-dir out/build/windows-release-cuda -R '^CudaLiquidBoundary3D\.' --output-on-failure
```

The clipper now rejects a pressure cell with disconnected wet pieces.
Wet tetrahedra connect only across a shared triangle with positive wet
area. A fixture with wet lobes at opposite off-diagonal corners
reproduced acceptance before the fix and now returns
`FAILED_PRECONDITION` on native and SM120. All 74 native geometry and
workspace tests and six CUDA boundary tests passed after the change.

The sampled input currently requires an otherwise unobstructed domain.
The mixed scheduler still needs particle-derived samples, composition
with the tank and moving obstacle, and reciprocal traction scatter
using the same geometry.

The sampled geometry and split-cell checks landed in
[PR 98](https://github.com/tychota/FluidiTy/pull/98) at `39f9de7`.
The standalone clean Windows Release CPU build passed all 74 geometry
and workspace tests. The clean milestone build with upstream adaptive
pressure and cache changes passed 161 selected native M5, geometry,
workspace, cache, schema, and CLI tests.

### Nodal wall flux and pressure work

The blocked-face basis weights integrate the sampled nodal velocities
into the signed pressure-link flux. The oblique two-cell oracle has a
0.75 m² open area and 0.625 m³/s wall flux. Its blocked area moves at an
average 2.5 m/s. [PR 100](https://github.com/tychota/FluidiTy/pull/100)
landed at `6022201` after 213 native and SM120 checks with upstream M6
pressure routing. A clean milestone build also passed six native M5
targets and three CLI tests.

The reciprocal pressure branch at `f9459cf` applies the transpose of the
same weights to the solved pressure difference. Its analytic test
recovers 2 N and 5 W for a pressure difference of 8 Pa. The native
two-cell projection also recovers the wall-flux power from the corner
forces and velocities. Transferring those loads to the owning particles
and accepting both materials in one production attempt remain open.


### Particle-derived boundary and pressure loads

[PR 101](https://github.com/tychota/FluidiTy/pull/101) landed at `6031c07`.
The particle sampler preserves the owner ID and moment arm for each
corner. The corner-particle fixture reproduces `1 - 0.125/sqrt(6)` m³
of wet volume. Its pressure impulse and affine work agree with the
blocked area and signed flux, and a stale frozen flux is rejected.
All 83 native and CUDA checks passed on the M6 base. The milestone
integration passed six native M5 targets and three CLI checks.


### Pressure-loaded snow trial

[PR 102](https://github.com/tychota/FluidiTy/pull/102) landed at `60c4712`
after 193 selected native tests passed on `6031c07`. The fixture stages
reciprocal loads on three blocked faces from solved liquid pressure.
The loaded trial advances with the expected translation and shear, and
an unknown owner leaves the accepted checkpoint unchanged. The milestone
integration passed six native M5 and three CLI tests.


### Joint snow and liquid scheduler acceptance

[PR 103](https://github.com/tychota/FluidiTy/pull/103) landed at `715ad59`
after all 143 native and CUDA scheduler tests passed. The shared
fixture verifies a pressure-induced snow velocity change, rejection
at the final acceptance stage without a snow candidate or changed
liquid checkpoint, and successful half-step retry. The CUDA case
checks the SM120 device receipt and the accepted device clock. The
snow solid step uses host transfer in both cases. All nine milestone
M5 and CLI checks passed after integrating the merge.


### Moving platen in the joint scheduler

[PR 104](https://github.com/tychota/FluidiTy/pull/104) landed at
`0a1b63d` after 197 native and 34 SM120 checks passed. Hosted CI was
queued at merge. The oblique analytic halfspace and sampled snow share
one face, with 0.375 m² open area and 1.25 m³/s total prescribed wall
flux. The device oracle checks both to 2e-14 in their respective units.
The host reaction fixture recovers 0.03 N·s and 0.09 J for snow from the
sampled share of the blocked face.

On both liquid backends the same vertical overhead platen pose is used
at the start of each accepted substep. The two-step fixture compares
unloaded snow against an independent dry advance to 2e-14 m in position
and 2e-14 m/s in velocity. It checks nonpenetration against each end
pose and checks the accepted liquid clock. The snow transfer runs on
the host. The production joint scheduler supports one vertical
overhead platen on the uniform liquid grid.

```powershell
cmake --build out/pr-m5-snow-platen/out/build/snow-platen-cpu --target fluidity_native_tests
ctest --test-dir out/pr-m5-snow-platen/out/build/snow-platen-cpu -R '^(LiquidBoundary(Geometry|Workspace)3D\.|LiquidSubstep3D\.)' --no-tests=error --output-on-failure
cmake --build out/pr-m5-snow-platen/out/build/snow-platen-sm120 --target fluidity_cuda_operator_tests fluidity_cuda_m5_constitutive_tests
ctest --test-dir out/pr-m5-snow-platen/out/build/snow-platen-sm120 -R '^(CudaLiquidSubstep3D\.|CudaLiquidBoundary3D\.|M5Constitutive\.cuda$)' --no-tests=error --output-on-failure
```

The authored gate remains open. The model-2 CLI must decode the mixed
initial state, advance both materials with the authored source and
moving obstacle, and publish separate channels and a joint checkpoint.
The Blender bake must read those channels and reproduce the
uninterrupted mixed state after an authenticated checkpoint reopen.

The clean integrated milestone build passed all six native M5 targets
and all fourteen CLI tests. The build includes the
`fluidity_simulate_fixture` target required by the CLI tests.

```powershell
cmake --build out/build/m5-platen-integrated-clean --target fluidity_m5_dry_scene_tests fluidity_m5_planar_coupling_tests fluidity_m5_plastic_checkpoint_tests fluidity_m5_snow_compaction_tests fluidity_m5_contact_scatter_tests fluidity_m5_constitutive_tests fluidity_simulate fluidity_simulate_fixture
ctest --test-dir out/build/m5-platen-integrated-clean -R '^M5.*\.native$|^SimulateCli' --no-tests=error --output-on-failure
```


### Joint frame scheduling and retry

[PR 105](https://github.com/tychota/FluidiTy/pull/105) landed at
`8a936f6` after all 120 native and 29 SM120 scheduler tests passed.
Hosted CI was queued at merge. The joint driver includes the snow
acoustic limit in the existing liquid timestep calculation, publishes
accepted snow before notifying the frame observer, and restores both
materials before replaying a frame after capacity growth.

The shared moving-platen fixture rejects a full 0.01 s attempt after
both trials, then accepts two 0.005 s steps. The capacity fixture
reduces the acoustic CFL to 0.0045, forcing the same half-step schedule,
then injects capacity exhaustion after the first half-step. It checks
that the replay observer sees the original liquid clock and original
snow position and velocity. Both fixtures check nonpenetration and
the completed-frame ordinal.

```powershell
cmake --build out/pr-m5-snow-frame/out/build/snow-frame-cpu --target fluidity_native_tests
ctest --test-dir out/pr-m5-snow-frame/out/build/snow-frame-cpu -R '^LiquidSubstep3D\.' --no-tests=error --output-on-failure
cmake --build out/pr-m5-snow-frame/out/build/snow-frame-sm120 --target fluidity_cuda_operator_tests
ctest --test-dir out/pr-m5-snow-frame/out/build/snow-frame-sm120 -R '^CudaLiquidSubstep3D\.' --no-tests=error --output-on-failure
```

The model-2 CLI still rejects mixed execution. The authored gate needs
the resolved snow settings and grid, a source-driven run through this
frame driver, separate Blender material channels, and an authenticated
restart comparison from a retained joint checkpoint.


### Model-2 CLI and retained mixed checkpoints

[PR 106](https://github.com/tychota/FluidiTy/pull/106) landed at
`172d4b6` after all 14 native and 21 SM120-build CLI tests passed.
Hosted CI was queued at merge. The CLI authenticates the initial FLSI
payload, advances one snow body on a uniform grid with a water source
and optional vertical platen, and publishes a point frame and retained
joint checkpoints. Snow uses the high bit of the point ID and tag.
The remaining tag bits identify its material. The cache checks mixed
point bytes against both checkpoint states at retained frames.

The fixture bakes 20 frames, observes two emitted liquid particles,
and checks the first snow point lies below and within 10 µm of the
moving platen. It reopens the retained frame-10 checkpoint and
reproduces the committed point and checkpoint files through frame 20.
Read-only replay reconstructs frame 19. A damaged initial payload
fails before cache creation. The fixture isolates platen motion from
the source to test the CLI and cache lifecycle.

```powershell
cmake --build out/pr-m5-mixed-cli/out/build/mixed-cli-cpu --target fluidity_simulate fluidity_simulate_fixture
ctest --test-dir out/pr-m5-mixed-cli/out/build/mixed-cli-cpu -R '^SimulateCli\.' --no-tests=error --output-on-failure
cmake --build out/pr-m5-mixed-cli/out/build/mixed-cli-sm120 --target fluidity_simulate fluidity_simulate_fixture
ctest --test-dir out/pr-m5-mixed-cli/out/build/mixed-cli-sm120 -R '^SimulateCli\.' --no-tests=error --output-on-failure
```

The authored gate still needs Blender snow authoring and an
end-to-end bake with obstacle compaction, water reaching the mound,
separate Blender material assignment, and authenticated playback
after restart.


### Authored Blender qualification attempt

[Draft PR 107](https://github.com/tychota/FluidiTy/pull/107) at
`931399d` adds Blender Snow controls and a native model-2 resolver.
Blender 5.2.3 verified separate water and snow materials, repeat
attachment, and both native Alembic point channels after reopening a
saved project backed by the independently qualified mixed CLI cache.

The authored 30-particle mound, animated platen and water source
resolve and publish frames 0 through 2. The native run exhausts its
duration retries in frame 3 after hitting a dry face-motion bound.
Maximum published snow speed increases from about 0.20 m/s in frame 1
to 2.21 m/s in frame 2. Smaller maximum timesteps of 0.0025, 0.001
and 0.0005 s fail before publishing frame 1. These results warrant
checking reciprocal pressure load, wall flux and kinetic work per
attempt before altering the scene or acceptance bounds. The authored
and Q-SAND gates remain open.

```powershell
$tree = (Resolve-Path out/pr-m5-blender-snow).Path
$build = Join-Path $tree 'out/build/blender-snow-cpu'
$blender = 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe'
cmake --build $build --target fluidity_simulate fluidity_blender_bridge
& $blender --background --factory-startup --python-exit-code 1 --python "$tree/tools/blender/verify_m5_snow_water.py" -- $tree $build "$tree/out/authored-repro" cpu
& $blender --background --factory-startup --python-exit-code 1 --python "$tree/tools/blender/verify_m5_snow_points.py" -- $tree $build (Resolve-Path out/pr-m5-mixed-cli/out/diagnostic-mixed/cache).Path "$tree/out/verified-mixed-channels.blend"
```


### Pressure feedback at the authored failure

A temporary native diagnostic on the authored case showed that the
pressure impulse is nearly independent of timestep over the retries
from the same state. At t = 0.08984375 s, the maximum pressure rose
from 193,822 Pa to 387,644 Pa and 775,287 Pa as the trial duration
halved twice. Maximum wall flux stayed 3.51633 m³/s and the largest
corner impulse stayed 141.959 N·s. Snow translational kinetic energy
rose from 284.43 J to about 4,007.26 J when that impulse was staged,
before the dry advance. The result identifies an unstable lag in the
pressure and snow velocity feedback that a shorter timestep alone
cannot resolve in this fixture. The temporary instrumentation was removed.

Draft PR 107 commit `bce5e94` builds the frozen snow pressure mobility
from the existing reciprocal traction coefficients and the APIC
mass metric. The native unit-mass cut-cell fixture checks the sign and
magnitude of the induced flux using independent linear and affine
impulse work, and checks linear scaling under pressure reversal.
All 47 selected native geometry and joint snow scheduler tests passed.

```powershell
cmake --build out/pr-m5-blender-snow/out/build/blender-snow-cpu --target fluidity_native_tests
ctest --test-dir out/pr-m5-blender-snow/out/build/blender-snow-cpu -R '^(LiquidBoundaryGeometry3D\.|LiquidSubstep3D\.Snow)' --no-tests=error --output-on-failure
```

The projection still needs to include this mobility in its pressure
system and apply the resulting snow impulse and liquid velocity in
the same accepted step. The authored bake remains open.


### Authored Blender snow–water bake

[PR 107](https://github.com/tychota/FluidiTy/pull/107) landed at
`d77bc6f` after 178 native and 36 SM120 tests passed with upstream
`888faba`. Hosted CI was queued at merge. The augmented pressure
system includes the frozen reciprocal snow mobility in the same solve.
A one-cell oracle checks the analytic pressure from liquid conductance
plus snow mobility and rejects a backend pressure that would impose an
inconsistent flux. The production acceptance checks the snow flux
mismatch at 2e-5 s⁻¹ per cell before applying the solved traction.

Blender 5.2.3 ran Apply Snow Preset, Check Scene, Bake and Attach on both
backends. The animated platen ends at 0.58 m, below the original
0.625 m snow surface. The 20-frame bake preserved all 30 snow particles
and emitted two water particles. Both backends measured a minimum
snow total volume ratio of 0.648350539 and a minimum log plastic
Jacobian of −0.420240034 in the retained checkpoint. The closest
emitted water center was 0.631821917 m from snow, outside its 0.4375 m
particle exclusion radius and inside the 0.8875 m combined field
support. Resume from frame 10 reproduced the frame-20 checkpoint bytes,
and both material-specific Alembic objects played after reopening the
saved Blender project. The augmented mobility is assembled on the host,
with the normal native or SM120 liquid projection supplying the accepted
pressure and face velocities.

```powershell
$tree = (Resolve-Path out/pr-m5-blender-snow).Path
$blender = 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe'
$cpu = Join-Path $tree 'out/build/blender-snow-cpu'
$cuda = Join-Path $tree 'out/build/blender-snow-sm120'
cmake --build $cpu --target fluidity_native_tests fluidity_simulate fluidity_blender_bridge fluidity_simulate_fixture
cmake --build $cuda --target fluidity_cuda_operator_tests fluidity_simulate fluidity_blender_bridge fluidity_simulate_fixture
ctest --test-dir $cpu -R '^(LiquidBoundaryGeometry3D\.|LiquidSubstep3D\.|SimulateCli\.)' --no-tests=error --output-on-failure
ctest --test-dir $cuda -R '^(CudaLiquidSubstep3D\.|CudaLiquidBoundary3D\.|SimulateCli\.cuda_mixed$)' --no-tests=error --output-on-failure
& $blender --background --factory-startup --python-exit-code 1 --python "$tree/tools/blender/verify_m5_snow_water.py" -- $tree $cpu "$tree/out/authored-native-repro" cpu
& $blender --background --factory-startup --python-exit-code 1 --python "$tree/tools/blender/verify_m5_snow_water.py" -- $tree $cuda "$tree/out/authored-cuda-repro" cuda
```

Q-SAND remains open at the extended settling endpoint.


### Symmetric APIC decay in physical time

The diagnostic on local branch `codex/m5-sand-affine`, commit
`cb69c2b`, decays the symmetric part of the stored affine fit by
exp(−rate × dt) after each accepted step. It retains the skew part,
the full deformation update, and the 30° constitutive friction angle.
All cases use the existing compatible-floor column, particle spacing,
grid widths, timesteps, and 1.50 s endpoint. The solver itself is unchanged.

| Case | Baseline | 2 s⁻¹ | 8 s⁻¹ | 32 s⁻¹ |
|---|---:|---:|---:|---:|
| Coarse | 29.962967° | 29.964813° | 29.897517° | 30.072721° |
| Coarse half-step | 32.042153° | 32.149358° | 32.485220° | 33.921460° |
| Fine | 27.424648° | 27.505456° | 27.677068° | 28.239650° |
| Fine half-step | 29.277572° | 29.353186° | 29.584487° | 30.475180° |

All twelve native cases meet the drift and momentum checks. The highest
rate passes three angles but leaves the nominal fine-grid angle at
28.24°. The coarse half-step angle also approaches the upper limit as
the decay grows. Uniform damping of the symmetric fit does not establish
a converged repose result. Both default dry-scene CTest targets passed. At 32 s⁻¹, the nominal
fine run accumulated 0.00468208 mean log-plastic-J increase during
the last 0.05 s. Particles ending within one particle spacing of the
floor supplied 0.00456823 of that path, or 97.6%. The coarse half-step
run had only 0.00032122 mean increase. This localizes the remaining
feedback to the floor layer in the failing fine run.

```powershell
$build = 'out/pr-m5-sand-affine/out/build/sand-affine-sm120'
cmake --build $build --target fluidity_m5_dry_scene_tests fluidity_cuda_m5_dry_scene_tests
ctest --test-dir $build -R '^M5DryScenes\.' --no-tests=error --output-on-failure
foreach ($rate in 2, 8, 32) {
  foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
    & "$build/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay $rate $case
  }
}
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$build/fluidity_cuda_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 32 $case
}
```


### Floor-normal ghost continuation

The original 32 s⁻¹ SM120 sweep reproduced all four native angles at the
printed precision. The nominal fine case still failed at 28.239650°.
Native regional controls at the same rate gave 26.560198° with decay
inside one particle spacing of the floor and 28.991526° with decay
above that layer. At the uniform rate the nominal fine run accumulated
0.00468208 mean absolute log-plastic-J path. Of this, 0.00466315 came
from steps whose particle centers started and ended above the floor.
Particles arriving at the floor supplied only 0.00001893. Floor
residents and departures supplied effectively zero.

[Draft PR 111](https://github.com/tychota/FluidiTy/pull/111) supplies the
stationary wall normal to the compatible below-floor ghost gather. PIC
transport and the swept Coulomb law keep their incoming velocity. The
one-particle slip oracle independently checks unchanged linear impulse,
zero plastic volume change and the polar rotation from trial shear
−0.0025. Both native and CUDA default dry-scene checks passed.

| Case | Native angle at 1.5 s | Native angle at 3 s |
|---|---:|---:|
| Coarse | 33.714729° | 36.597489° |
| Coarse half-step | 32.702577° | 29.470173° |
| Fine | 32.270910° | 38.026750° |
| Fine half-step | 31.172480° | 28.443207° |

All four 1.5 s cases meet the declared angle, drift, energy and momentum
bounds without affine damping. All four leave the angle band at 3 s.
The nominal fine case keeps its 0.153218 m radius while its height
increases from 0.096751 to 0.119822 m. Its accumulated cut stress work
rises from +0.686445 to +1.049375 J. The dry solver pairs stress deposition
with normalized weights while deformation uses the complete ghost
derivative. These measurements motivate checking the transpose of the
wall-constrained deformation gather for stress deposition. The 1.5 s
snapshot alone does not qualify a settled pile.

Use the local `codex/m5-sand-affine` diagnostic branch to rerun the
three-second controls. Use the PR branch for the original four cases.
The SM120 1.5 s candidate matrix is running separately.

```powershell
$build = 'out/pr-m5-sand-affine/out/build/sand-affine-sm120'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$build/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```


### Transpose of the wall-constrained deformation gather

The local diagnostic at `4c56e0d` applies the transpose of the exact
stationary floor continuation to the full-stencil stress force. The
existing affine fit supplies the donor coefficients. Rank fallback
scatters to the same nearest donor, and fixed below-floor normal
velocities receive no force work. The isolated stress oracle measured
a work residual of −7.11e-15 J on native and −6.22e-15 J using the
SM120 gather. Both default dry-scene tests passed.

| Case | Paired at 1.5 s | Paired at 3 s | Paired with grid-supported friction at 1.5 s | Paired with grid-supported friction at 3 s |
|---|---:|---:|---:|---:|
| Coarse | 27.481554° | 16.658214° | 27.714905° | 17.149176° |
| Coarse half-step | 33.813342° | 26.349700° | 34.386059° | 27.105769° |
| Fine | Motion bound | Motion bound | Motion bound | Motion bound |
| Fine half-step | 32.051263° | 27.187768° | 32.417311° | 27.633086° |

Completed cases passed the vertical momentum bound. Cut-stress work
mismatch stayed below 3e-16 J. The friction control uses each particle's
compressive normal impulse from the fixed ghost constraint to cap
additional tangential friction before the swept response. Its angular
effect is limited, and the nominal fine case still rejects at step 170.
At that step an X face carries 2.61e-9 kg and receives −0.04596 N of
scattered stress. The existing 3.75e-6 kg force-mass floor still allows
the face velocity to reach −19.67 m/s, past the 18 m/s limit. The
transient face instrumentation was removed. This identifies the
next numerical constraint on the paired transfer. Donor selection
needs to account for mass without breaking the work pairing.

```powershell
$build = 'out/pr-m5-sand-affine/out/build/sand-affine-sm120'
$bin = 'out/pr-m5-sand-affine/out/floor-transpose-debug-bin'
cmake --build $build --target fluidity_m5_dry_scene_tests fluidity_cuda_m5_dry_scene_tests
ctest --test-dir $build -R '^M5DryScenes\.' --no-tests=error --output-on-failure
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```


### Mass-weighted paired floor reconstruction

The four SM120 cases for draft PR 111 finished and reproduced
33.714729°, 32.702577°, 32.270910°, and 31.172480° at 1.5 s. The
native three-second overshoot and flattening evidence remains.

Local commit `f2e0739` weights each affine QR donor by deposited
face mass and uses the same reconstruction coefficients in stress
deposition and deformation. The work oracle preserves the affine
field across a donor mass ratio of 1e8. Its work residual was
1.81e-12 J on native and SM120. The nominal fine case finished the
1.5 s interval at 36.312561° on both backends. The native continuation
still hit the face-motion bound, at step 1176 instead of step 170.

Local commit `52e2c90` also reconstructs faces below the existing
force-mass floor. Their deposited PIC momentum remains available to
transport. The same reconstruction supplies the deformation sample
and sends the transposed stress force to eligible donors. Every
native run finished through 3 s. The absolute vertical momentum
residual stayed below 1e-8 N·s and the cut-work mismatch below
3e-16 J. The angles still miss the gate.

| Case | Weighted at 1.5 s | Weighted at 3 s | Weighted with cutoff at 1.5 s | Weighted with cutoff at 3 s |
|---|---:|---:|---:|---:|
| Coarse | 34.925916° | 37.649598° | 34.636386° | 35.659991° |
| Coarse half-step | 37.105986° | 32.182352° | 37.555997° | 32.243641° |
| Fine | 36.312561° | Motion bound | 37.508583° | 37.729420° |
| Fine half-step | 35.038188° | 24.872650° | 35.704764° | 29.311134° |

Use the isolated `codex/m5-sand-affine` commits and the same two-horizon
commands from the preceding section. The weighted binaries are in
`out/pr-m5-sand-affine/out/floor-mass-weighted-bin`. The native cutoff
binary is in `out/pr-m5-sand-affine/out/floor-supported-bin`.


### Odd normal reflection at the stationary plane

Local commit `87a20c9` fixes the normal deformation sample at the floor
plane and reflects below-floor normal samples with odd parity. The
paired stress transfer uses the same sign and donor coefficients.
The fixture retains the incident PIC velocity for swept Coulomb
contact. The native analytic slip oracle passed with the reflected
simple-shear rotation and unchanged linear impulse.

| Case | Reflected at 1.5 s | Reflected at 3 s |
|---|---:|---:|
| Coarse | 35.259118° | 33.916100° |
| Coarse half-step | 43.117606° | 42.088767° |
| Fine | 32.534357° | 32.979313° |
| Fine half-step | 38.747054° | 29.627498° |

All eight native runs finished. The vertical momentum residual stayed
below 8e-9 N·s and cut work mismatch below 4e-16 J. The nominal fine
case remained in range at both horizons, but the half-step cases did
not. The preceding wall-plane-only control at `24f888b` changed
reaction allocation without passing the matching tags to the
deformation gather. Its angle results do not establish the effect of
a wall-plane-only constraint and are excluded from this comparison.


### Reflected-floor timestep refinement and contact affine state

Local diagnostic `1e30a76` keeps the original 63-particle column and
the same three-second endpoint while reducing the reflected solver's
nominal timestep. All four runs completed and missed the angle band.

| Grid | Quarter step at 3 s | Eighth step at 3 s |
|---|---:|---:|
| Coarse | 38.129921° | 24.390220° |
| Fine | 39.551096° | 40.896382° |

Local commit `62c447f` preserves the derivative of the swept Coulomb
velocity projection as the stored affine fit at floor impact. The
native analytic and dry-scene regressions passed. The full scene
outcomes were:

| Case | Angle at 1.5 s | Angle at 3 s |
|---|---:|---:|
| Coarse | 30.988615° | 30.981431° |
| Coarse half-step | 41.821079° | 41.044791° |
| Fine | 30.656335° | 31.003871° |
| Fine half-step | 38.747054° | 29.627498° |

The nominal cases have stable angles inside the band, but miss the
original radius bound. At 3 s their radii are 0.164953 and 0.161415 m,
respectively, above the 0.16 m limit. The half-step cases miss the
angle criterion. The cutoff and work-pairing gains remain, with
vertical residuals below 1e-8 N·s and cut-work mismatch below
4e-16 J. The gate stays open.


The SM120 default dry-scene and transpose oracles also passed for
`62c447f`. The work residual with mass weighting was 1.82e-12 J.

The accepted-step contact trace samples the same integration without
inserting step boundaries. By about 0.25 s, all four runs report zero
swept contacts and zero swept impulse, but the reflected grid has
already supplied 3.96–4.91 N·s of vertical wall impulse. At about
0.35 s the fine half-step has still recorded no swept contact, yet its
grid reaction is 9.09 N·s and its height has rebounded to 0.203 m.
The nominal fine run has recorded 207 swept contacts and reaches
0.114 m at the same time. This locates the divergence in the early
floor response. The next check is whether wall support activates
only when the solid reaches the floor and whether the same active
constraint drives the velocity gather and stress transpose.

```powershell
$bin = 'out/pr-m5-sand-affine/out/floor-contact-affine-bin'
foreach ($grid in 'coarse', 'fine') {
  foreach ($step in 'nominal', 'half') {
    & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-contact-trace $grid $step
  }
}
```


### Contact-activated grid support

Local commit `46c9d4b` delays the reflected floor constraint until an
accepted particle reaches the wall. Before that event the paired
gather and stress transpose use unconstrained support. The compressed
single-particle oracle has nonzero stress and no wall reaction while
its center is 0.2 cell widths above the floor. Both default native and
SM120 dry-scene targets passed with this oracle.

The fine half-step trace reports zero grid reaction and zero swept
contacts at 0.15 s. By 0.20 s it reports 477 swept contacts and
0.298291 N·s of grid reaction. At 0.35 s its height is 0.160144 m,
versus 0.202572 m for the continuously active reflected constraint.
This removes the earlier premature grid-supported rebound.

| Case | Global activation at 1.5 s | Global activation at 3 s | Contact-local activation at 1.5 s | Contact-local activation at 3 s |
|---|---:|---:|---:|---:|
| Coarse | 32.417839° | 31.264670° | 35.233573° | 32.098814° |
| Coarse half-step | 38.078230° | 33.821046° | 39.835357° | 36.198762° |
| Fine | 29.939029° | 27.313441° | No compatible donor | No compatible donor |
| Fine half-step | 34.924794° | 28.055917° | 37.166363° | No compatible donor |

Only the global coarse 1.5 s and coarse half-step 3 s cases pass all
original checks. The other angles in range miss the radius bound.
The contact-local experiment only enables the constraint on stencils
of particles already touching the floor. Its native default checks
pass, but the fine case loses a compatible ghost donor at step 863,
and the fine half-step at step 2807. Completed runs retain the
work pairing and vertical momentum balance. Q-SAND stays open.

```powershell
$bin = 'out/pr-m5-sand-affine/out/floor-contact-active-bin'
& "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-contact-trace fine half
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

The contact-local binaries use the same commands from
`out/pr-m5-sand-affine/out/floor-local-contact-bin`.


### Bounded recovery of a missing contact-local donor

The contact-local control now tries a constant, same-component nearest
donor within four cells if its two-cell affine neighborhood is empty.
The forward reconstruction and stress transpose use the same unit
coefficient. The two previously rejecting fine-grid native cases
both finished through 3 s. The nominal case settled at 26.677083°
with a 0.193416 m radius. The half-step case reached 26.427206° with
a 0.205951 m radius. Both missed the original angle and radius checks.
The work mismatch remained below 4e-16 J and vertical momentum
residuals stayed below 1.4e-8 N·s. Both default native and SM120
dry-scene targets passed before the dedicated fallback oracle was added.

```powershell
$bin = 'out/pr-m5-sand-affine/out/floor-local-contact-bin'
foreach ($case in 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Reflected deformation gradient during impact

Local experiment `b2d03c6` uses the full reflected-grid gradient for the
deformation update over an impact step. Swept particle velocity and
stored affine velocity still use the Coulomb projection. The previous
contact split also projected the deformation gradient after impact.
Both dry-scene targets passed after rebuilding the final source, including
the impact oracle for elastic compression and the bounded donor oracle
for force pairing and rollback.

| Native case | 1.5 s angle and radius | 3 s angle and radius |
|---|---|---|
| Coarse | 34.757332°, 0.134870 m (pass) | 32.002240°, 0.136839 m (pass) |
| Coarse half-step | 39.490763°, 0.130452 m | 36.029863°, 0.138532 m |
| Fine | 32.035738°, 0.165352 m | Dry face-motion bound at step 1152 |
| Fine half-step | 39.101883°, 0.144883 m | 29.792169°, 0.180244 m |

The completed cases retained less than 5.1e-16 J of cut stress work
mismatch and less than 5.6e-9 N·s of vertical momentum residual. The
control leaves Q-SAND open. The native fine case rejected an X face
carrying 1.05459119e-5 kg with 0.14972351 N of stress force. The
pre-force velocity was −0.00075168 m/s and the post-force velocity was
22.18253479 m/s against the 18 m/s motion bound. The SM120 coarse and
coarse half-step runs at 3 s matched the native angle, radius, work
and momentum figures at the printed precision.

```powershell
$bin = 'out/pr-m5-sand-affine/out/floor-wall-gradient-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Doubled minimum face-support mass

Starting at `b2d03c6`, change the `minimum_force_mass` multiplier in
`native/src/dry_solid_step_3d.cc` from `1e-4` to `2e-4`, rebuild the
native dry-scene target and repeat the 3 s matrix. The thin face
identified above is below the new reconstruction threshold.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 32.392180° | 0.135394 m |
| Coarse half-step | 36.667660° | 0.136258 m |
| Fine | 27.170002° | 0.182763 m |
| Fine half-step | 30.047349° | 0.178421 m |

The fine nominal case now reaches the endpoint. Only the coarse
nominal case passes all checks. All runs had late drift below 0.09°,
kinetic energy below 0.00017 J, cut stress work mismatch below
6.3e-16 J, and vertical momentum residual below 3.2e-9 N·s. The
experiment restored the source multiplier after building.

### Requalification against upstream 873ae68

Upstream `873ae68` includes the M4 graded runtime and updated shared
liquid substep code. The M5 review branch integrated it at `efb35dd`.
The existing Windows Ninja build reported zero header dependencies for
both planar test objects and reused stale objects after the update.
The native planar test crashed and the CUDA planar test failed in that
incremental run. Both passed in a fresh Release SM120 build.

The fresh build passed all 11 M5 CTest targets. Blender 5.2.3 repeated
the authored 20-frame bake, frame-10 resume and playback on native and
SM120. Both runs retained the same minimum snow volume ratio
(0.648350539), minimum log plastic Jacobian (−0.420240034), and emitted
water gap (0.631821917 m). The tracked native sources passed
clang-format 19.1.5.

Use a fresh Release build directory with the qualified Windows CUDA
toolchain, `120-real;120-virtual` architectures, the shared CUDA runtime,
and `FLUIDITY_BUILD_BLENDER_BRIDGE=ON` with Alembic and Imath available.

```powershell
$b = 'out/build/m5-main-873ae68-clean'
$targets = @(
  'fluidity_m5_dry_scene_tests', 'fluidity_m5_planar_coupling_tests',
  'fluidity_m5_plastic_checkpoint_tests', 'fluidity_m5_snow_compaction_tests',
  'fluidity_m5_contact_scatter_tests', 'fluidity_m5_constitutive_tests',
  'fluidity_cuda_m5_dry_scene_tests', 'fluidity_cuda_m5_planar_coupling_tests',
  'fluidity_cuda_m5_plastic_checkpoint_tests', 'fluidity_cuda_m5_snow_compaction_tests',
  'fluidity_cuda_m5_constitutive_tests', 'fluidity_simulate', 'fluidity_blender_bridge'
)
cmake --build $b --target @targets --parallel 8
ctest --test-dir $b -R '^M5' --no-tests=error --output-on-failure
$tree = (Get-Location).Path
$bin = (Resolve-Path $b).Path
$blender = 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe'
foreach ($backend in 'cpu', 'cuda') {
  & $blender --background --factory-startup --python-exit-code 1 --python "$tree/tools/blender/verify_m5_snow_water.py" -- $tree $bin "$tree/out/authored-upstream-873ae68-$backend" $backend
}
```

### Predicted first floor contact

Starting at `b2d03c6`, build the same contact-local stencil mask for
particles already on the floor or predicted to reach it within the
step. The predicted height is
`y + dt * (accepted_velocity_y + dt * gravity_y)`. The accepted
Coulomb sweep and paired transfer are unchanged. Rebuild the native
dry-scene target and repeat the 3 s matrix above. The default native
dry-scene target passed.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 32.119155° | 0.136599 m |
| Coarse half-step | 36.122549° | 0.138623 m |
| Fine | Dry face-motion bound at step 1196 | |
| Fine half-step | 29.857831° | 0.180331 m |

The three completed cases retained cut stress work mismatch below
2.2e-16 J and vertical momentum residual below 5.3e-8 N·s. Activating
the mask during the predicted impact step moves the fine nominal
rejection from step 1152 to 1196 and leaves the qualifying cases
unchanged. The experimental source was restored after compiling.

### Radial distribution and late plastic volume

At `b2d03c6`, sort the 63 final horizontal particle radii and report
the order statistic at index `floor((count - 1) * percentile / 100)`.
Repeat the reflected-gradient 3 s controls with the original maximum
radius still used for acceptance. The diagnostic changed no solver
state and was removed after compilation.

| Native case at 3 s | Median radius | 90th percentile | At or beyond 0.16 m | Mean absolute log plastic volume change, final 0.05 s |
|---|---:|---:|---:|---:|
| Coarse | 0.094061 m | 0.131892 m | 0 of 63 | 0.00229371 |
| Coarse half-step | 0.091762 m | 0.133758 m | 0 of 63 | 0 |
| Fine half-step | 0.097002 m | 0.165592 m | 8 of 63 | 0 |

The fine half-step radius failure affects eight particles. Both
completed half-step cases already miss the gate with zero late plastic
volume change. The coarse nominal case passes while still dilating in
the floor layer. These observations narrow the remaining search to
how the shape develops before the final window, including shear
plasticity and earlier radial transport.

### Impact-to-settling timestep divergence

The contact trace at `b2d03c6` shows all four runs still within
2.1 mm of each other in maximum height at the first sample near
0.20 s. By 0.35 s the half-step cases stand higher and spread less
than their nominal counterparts.

| Native case | Height at 0.35 s | Radius at 0.35 s | Height at 0.50 s | Radius at 0.50 s |
|---|---:|---:|---:|---:|
| Coarse | 0.126368 m | 0.117818 m | 0.107442 m | 0.123573 m |
| Coarse half-step | 0.155805 m | 0.111181 m | 0.128777 m | 0.114983 m |
| Fine | 0.131607 m | 0.118928 m | 0.115730 m | 0.133786 m |
| Fine half-step | 0.153225 m | 0.107474 m | 0.135463 m | 0.115883 m |

The trace prints the first accepted time at or after each requested
sample. The reported times in the table differ by at most 0.000625 s
within a column. At the 0.35 s sample the fine half-step case has
0.00780 J of kinetic energy versus 0.05629 J at the nominal step.

```powershell
$bin = 'out/pr-m5-sand-affine/out/floor-wall-final-bin'
foreach ($grid in 'coarse', 'fine') {
  foreach ($step in 'nominal', 'half') {
    & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-contact-trace $grid $step
  }
}
```

At the same 3 s endpoint, reducing the timestep further gave the
following results. Neither grid converges into the acceptance band
over these refinements.

| Native case | Quarter nominal step angle / radius | Eighth nominal step angle / radius |
|---|---|---|
| Coarse | 39.358297° / 0.130208 m | 47.749195° / 0.114218 m |
| Fine | 38.591885° / 0.147827 m | 43.549693° / 0.132036 m |

```powershell
foreach ($grid in 'coarse', 'fine') {
  foreach ($step in 'quarter', 'eighth') {
    & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-reflected-timestep $grid $step
  }
}
```

The timestep-switch control uses the same 3 s fixture and chooses
the other duration at the first step whose starting time is at
least 0.50 s. It adds no sampling boundary at 0.50 s. In the fixture
time loop, substitute `t >= 0.50 ? duration * scale : duration`
for `duration` in both uses in the `next_time` calculation.
Use scale 0.5 for nominal-to-half or 2.0 for half-to-nominal.

| Native case | Nominal then half angle / radius | Half then nominal angle / radius |
|---|---|---|
| Coarse | 34.076557° / 0.135997 m (pass) | 33.418104° / 0.142172 m (pass) |
| Fine | 28.148776° / 0.185503 m | 27.512464° / 0.180912 m |

Both switched fine cases finish and miss both geometric limits.
Their results are closer to each other than to the results of
quartering the fixed step. All four switched runs retain vertical
momentum residual below 8.2e-9 N·s and late angle drift below 0.09°.
The variable-step fixture was restored after compilation. The
original fixed-step acceptance matrix still governs Q-SAND.

### Contact-local grid-reaction friction

The reflected-gradient candidate can apply tangential friction directly
to an elevated particle if its stress stencil contributes to the grid
wall reaction. Starting at `b2d03c6`, two native controls restrict that
velocity correction to particles already touching the floor. The first
drops the elevated contribution. The second distributes each elevated
positive impulse to contacts within horizontal distance `1.5h`, using
normalized weights `max(0, 1 - distance_squared / (2.25h²))` before the
unchanged Coulomb cap. The sweep, stress transpose and acceptance checks
are unchanged. The first control passed the default native dry scenes.

| Native case at 3 s | Suppress elevated correction | Redistribute elevated correction |
|---|---|---|
| Coarse | Face-motion bound at step 1461 | 23.051525° / 0.165683 m |
| Coarse half-step | 22.576002° / 0.166646 m | 25.612154° / 0.170636 m |
| Fine | Face-motion bound at step 1144 | Face-motion bound at step 1606 |
| Fine half-step | Face-motion bound at step 3561 | 19.864435° / 0.244496 m |

The completed cases retained vertical momentum residual below
5.7e-9 N·s. Moving friction capacity to contacts recovers some
completed runs, but the piles spread further than in the
reflected-gradient candidate. The coarse candidate pass is sensitive
to damping above the floor. A qualification candidate still needs a
wall traction discretization that transmits tangential stress through
the grid and an isolated test of its impulse and work. Both source
controls were restored after compiling.

```powershell
foreach ($bin in 'sand-supported-friction-bin', 'sand-contact-friction-transfer-bin') {
  foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
    & "out/pr-m5-sand-affine/out/$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
  }
}
```

### Isolated mass-weighted wall friction

Local sand experiment `2de5756` solves the stationary tangential
impulse in the diagonal grid mobility metric. For contact velocity
`u`, positive tangent mobility `D`, and Coulomb impulse cap `c`,
it minimizes `u·j + 0.5 j·D j` over `||j|| <= c`. Sticking cancels
the tangent velocity. In sliding, the scalar multiplier on the
Coulomb disk shifts both diagonal mobilities.

The independent oracle uses tangent velocity (3, 4) m/s and
mobilities (2, 4) kg⁻¹. It checks sticking impulse (−1.5, −1) N·s,
sliding impulse (−0.75, −2/3) N·s and sliding kinetic work
−3.465277777777778 J to 1e-12 in fixture units. It also covers zero
friction and rejects zero mobility. The native and SM120 dry-scene
targets both passed. Both execute this friction oracle on the host.
This test verifies the contact solve given a normal load. Grid
traction gather and transpose scatter into the sand step are pending.

```powershell
$b = 'out/pr-m5-sand-affine/out/build/sand-affine-sm120'
cmake --build $b --target fluidity_m5_dry_scene_tests fluidity_cuda_m5_dry_scene_tests
ctest --test-dir $b -R '^M5DryScenes\.' --no-tests=error --output-on-failure
```

### Grid gather and transpose scatter for tangential traction

Local sand commit `0d3f985` applies the isolated mass-metric solve to
the grid before particle gathering. It normalizes each tangent stencil
over supported face masses, gathers with those weights, and scatters
the impulse with the transpose weights. Each tangent mobility is the
sum of squared normalized weights divided by face mass. The existing
per-particle positive normal reaction sets each traction cap. This
still places traction quadrature at the stress source, including
sources above the floor.

The compressed, sliding one-particle fixture verifies a tangent impulse
opposing motion, a magnitude matching the Coulomb cap to 1e-10 N·s,
negative grid friction work, and paired cut stress work below 1e-12 J.
Grid friction work is counted separately from the particle sweep.
Both native and SM120 dry-scene targets passed this fixture. Friction
updates also check the existing face motion bound.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 39.616695° | 0.133711 m |
| Coarse half-step | 41.855289° | 0.132023 m |
| Fine | 33.846351° | 0.171221 m |
| Fine half-step | 34.730438° | 0.179730 m |

All four native cases reach the endpoint with cut stress work mismatch
below 2.6e-16 J and vertical momentum residual below 4.1e-8 N·s.
The coarse cases fail the angle limit and the fine cases fail the
radius limit. Q-SAND remains open.
The SM120 fine nominal case matched the native angle, radius, stress
work and vertical momentum values at printed precision.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-grid-friction-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
& "$bin/fluidity_cuda_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 fine all 3
```

### Tangential traction at the wall reaction faces

Local experiment `7dbe274` locates the friction quadrature on the
stationary plane. For each constrained normal row, its signed force
contribution is the original face force times the row coefficient sum
minus one. Rows at the same horizontal position are combined before
taking the positive normal load. Each wall point gathers its own
tangent velocity from supported quadratic faces and scatters the
mass-metric impulse using the same normalized weights.

The loaded single-particle fixture initially lacked horizontal halo
blocks for the new wall-point stencils. With the required support
allocated, both native and SM120 dry-scene targets passed, including
the Coulomb cap, negative grid friction work, and stress-work pairing.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 35.302086° | 0.183455 m |
| Coarse half-step | 39.530375° | 0.161270 m |
| Fine | Dry face-motion bound at step 828 | |
| Fine half-step | 34.867803° | 0.179200 m |

The completed cases retained cut stress work mismatch below 2.1e-16 J
and vertical momentum residual below 5.7e-9 N·s. Moving the quadrature
to the wall reduces both coarse angles relative to the particle-centered
grid traction, with larger radii and one fine-grid rejection. The
original Q-SAND checks remain open.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-wall-face-friction-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### One-sided wall support and sweep ordering

Starting from wall-face traction, exclude tangent faces below the wall
from the weight normalization, gather, and transpose scatter. Local
commit `59e1426` also splits the Coulomb cap equally across a forward
and a reverse traversal of the wall points. The sum of the magnitudes
of the two impulses at a point stays within its original cap. Both
versions passed the native and SM120 loaded-traction dry-scene tests.

| Native case at 3 s | One-sided forward angle / radius | One-sided forward and reverse angle / radius |
|---|---|---|
| Coarse | 35.376436° / 0.186758 m | 35.932364° / 0.178563 m |
| Coarse half-step | 40.969791° / 0.155660 m | 41.107594° / 0.153975 m |
| Fine | Dry face-motion bound at step 891 | Dry face-motion bound at step 846 |
| Fine half-step | 34.614158° / 0.180542 m | 34.345649° / 0.179405 m |

The completed cases retained cut stress work mismatch below 6.2e-16 J
and vertical momentum residual below 2.2e-8 N·s. The sweep order changes
the resulting geometry without changing which gates fail. In the fine
nominal runs the rejection still comes from the stress-update motion
bound, before tangential friction is applied.

```powershell
foreach ($bin in 'sand-wall-one-sided-bin', 'sand-wall-bidirectional-bin') {
  foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
    & "out/pr-m5-sand-affine/out/$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
  }
}
```

### Motion retries and thin support with wall-face traction

At `59e1426`, the fine nominal run first rejects at 1.321875 s.
Its X face in slot 5014 carries 5.73538182e-6 kg, above the
3.75e-6 kg reconstruction threshold. The stress force is 0.11043345 N.
The face speed rises from 0.01934248 to 30.10491940 m/s, above the
18 m/s bound. The diagnostic source was restored after compiling.

Local commit `d91b03d` adds a separate retry command for the same case.
It halves the rejected interval when the checked dry-step status
permits a duration retry, then attempts the nominal step again after
each acceptance. Native and SM120 reach 3 s in 2,893 accepted steps
with 2,176 rejected attempts. The output logs match byte for byte.
The pile reaches 28.498761° and 0.214432 m, missing both geometric
checks. The vertical momentum residual is 1.35e-8 N·s, and cut
stress work mismatch is 9.38e-16 J.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-motion-retry-bin'
& "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-motion-retry *> native-retry.log
& "$bin/fluidity_cuda_m5_dry_scene_tests.exe" --q-sand-motion-retry *> cuda-retry.log
```

A fixed-step control doubles the `minimum_force_mass` multiplier
from `1e-4` to `2e-4` with the same wall traction. The new cutoff is
above the originally rejected face mass. The fine nominal rejection
moves from step 846 to 895. The remaining fixed-step cases finish
with coarse 36.949208° / 0.172551 m, coarse half-step 41.531990° /
0.151460 m, and fine half-step 34.719518° / 0.177724 m. Rebuild the
native dry-scene target and use the fixed-step matrix above to
reproduce that control. The threshold was restored after compilation.
Q-SAND remains open.

### Rebuild stress donors using the face motion bound

Local sand commit `7a7cef0` rebuilds the paired stress transpose and
deformation reconstruction when stress would push a thin donor beyond
the existing face-motion bound. Only structurally eligible faces
below one percent of the minimum particle mass can be removed. The
pre-stress drift must already satisfy the motion bound. Each pass
restarts from the original face force, removes at least one donor
or finishes, and rejects if eight passes are exhausted.

Both native and SM120 dry-scene targets passed. All four native
fixed-step piles reach 3 s. The coarse nominal, coarse half-step,
and fine half-step results match the preceding control at printed
precision. The fine nominal case reaches 29.704868° and a radius of
0.200837 m. Its late angle drift is 0.023640°, kinetic energy is
0.000122 J, cut stress work mismatch is 6.74e-16 J, and vertical
momentum residual is 1.05e-8 N·s. The motion rejection is resolved
for this case while its geometry still misses both checks.
Native and SM120 fine nominal output logs match byte for byte.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-stress-donor-retry-bin'
foreach ($case in 'fine', 'fine-half', 'coarse', 'coarse-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
& "$bin/fluidity_cuda_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 fine all 3
```

### Horizontal momentum and the frozen-gradient volume control

The original fixed-step reconstruction matrix closes the horizontal
impulse balance against grid traction and swept wall impulse. The
largest native per-step residual is 1.56e-10 N·s, and the largest
sum of residual norms over any run is 1.53e-7 N·s. This does not
explain the spreading through a net unaccounted horizontal impulse.

Local control `5e4e59e` multiplies the compatible-floor trial
deformation by a scalar so its incremental determinant equals
`exp(dt * trace(L))` for a frozen gradient. This diagnostic changes
the explicit Euler update in design §2.3. The unchanged Euler
matrix remains the qualification reference. The one-particle
compression oracle now checks `exp(-0.01)`, and both native and
SM120 dry-scene targets passed the experimental oracle.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 36.352235° | 0.175461 m |
| Coarse half-step | 41.132684° | 0.153541 m |
| Fine | 31.353594° | 0.186970 m |
| Fine half-step | 34.340220° | 0.179144 m |

The fine nominal run improves both geometric measurements and still
misses the radius limit. The fine half-step result changes little.
All cases retain cut stress work mismatch below 5.2e-16 J and
vertical momentum residual below 2.4e-8 N·s. Q-SAND stays open.

```powershell
foreach ($bin in 'sand-tangent-momentum-bin', 'sand-volume-increment-bin') {
  foreach ($case in 'fine', 'fine-half', 'coarse', 'coarse-half') {
    & "out/pr-m5-sand-affine/out/$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
  }
}
```

### Horizontal alignment and particle sampling with wall-face traction

Use the motion-aware donor reconstruction at `7a7cef0` with the
specified Euler deformation increment. For horizontal alignment,
retain the 63-particle release, original timestep, and 3 s endpoint.
Set the grid origin to `(phase * h, 0, phase * h)` while keeping the
column and floor fixed in world coordinates.

| Native case at 3 s | Zero phase angle / radius | Quarter-cell phase angle / radius | Half-cell phase angle / radius |
|---|---|---|---|
| Coarse | 35.932364° / 0.178563 m | 36.491562° / 0.172773 m | 35.904466° / 0.180511 m |
| Coarse half-step | 41.107594° / 0.153975 m | 42.272557° / 0.146590 m | 40.468889° / 0.156928 m |
| Fine | 29.704868° / 0.200837 m | 26.854028° / 0.205632 m | 28.556517° / 0.200622 m |
| Fine half-step | 34.345649° / 0.179405 m | 31.683658° / 0.194212 m | 32.831118° / 0.195432 m |

Changing the phase moves the fine-grid angles by up to 2.85°.
Every coarse half-step angle remains above 40°, and every fine
radius remains above 0.16 m.

A separate zero-phase control doubles particles per axis, from 63
to 504 midpoint samples at the same column dimensions and total
mass. With the original half-step durations, the coarse result is
33.915408° / 0.203072 m and the fine result is 29.106618° /
0.218145 m. The fine quadrant fits range from 23.001748° to
24.247701°. The denser sampling increases the radius on both grids.

```powershell
$root = 'out/pr-m5-sand-affine/out'
foreach ($phase in 'quarter', 'half') {
  foreach ($grid in 'fine', 'coarse') {
    foreach ($step in 'nominal', 'half') {
      & "$root/sand-grid-phase-wall-bin/fluidity_m5_dry_scene_tests.exe" --q-sand-wall-grid-phase $grid $step $phase
    }
  }
}
foreach ($grid in 'fine', 'coarse') {
  & "$root/sand-dense-wall-bin/fluidity_m5_dry_scene_tests.exe" --q-sand-wall-dense $grid half
}
```

These temporary harnesses use the existing fixture's grid-origin
parameter or its particle multiplier of two. The solver and harness
sources were restored after linking. Neither control qualifies Q-SAND.

### Wall-compatible affine extension

At the zero-phase Euler reference, replace the compatible-floor
post-contact affine projection with the fitted grid slope and set
the normal row's two tangential derivatives to zero. The resulting
affine normal velocity vanishes along the contact plane. This
control keeps the fitted tangential rows and off-plane normal
derivative. It leaves the current-step traction and deformation
updates in place. Both native and SM120 dry-scene targets passed.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 28.348113° | 0.202630 m |
| Coarse half-step | 37.963739° | 0.171778 m |
| Fine | 28.657890° | 0.210721 m |
| Fine half-step | 31.797157° | 0.200325 m |

All four radii increase compared with the Coulomb-projected affine
reference. Coarse nominal late drift also exceeds 2° at 2.184878°.
Cut stress work mismatch stays below 3.5e-16 J and vertical momentum
residual below 1.8e-8 N·s. Source was restored after compiling.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-wall-affine-extension-bin'
foreach ($case in 'fine', 'fine-half', 'coarse', 'coarse-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Clear the affine fit after wall contact

The zero-phase Euler control leaves the stored affine slope zero
after a swept wall contact. The Coulomb velocity update and wall-face
traction remain in place. The temporary source was restored after
linking.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 27.660759° | 0.242658 m |
| Coarse half-step | 40.631438° | 0.158705 m |
| Fine | 34.127422° | 0.182198 m |
| Fine half-step | 36.071663° | 0.175083 m |

Clearing the slope reduces both fine radii compared with the
Coulomb-projected affine result. Both are still above 0.16 m.
The coarse nominal radius grows by 0.064095 m. Horizontal per-step
momentum residual stays below 1.6e-10 N·s and cut stress work
mismatch below 5.2e-16 J.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-wall-reset-affine-bin'
foreach ($case in 'fine', 'fine-half', 'coarse', 'coarse-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Requalification against upstream 195af13

The milestone branch integrated upstream `195af13` at `64a9ed3`.
This upstream revision includes the persistent CUDA adaptive pressure
arena, graded physical geometry and wall-strain changes, and updated
shared liquid substep code.

The clean Release SM120 build passed all 11 M5 CTest targets.
Blender 5.2.3 repeated the authored 20-frame bake, frame-10 resume,
and separate material playback on native and SM120. Both measured
minimum snow volume ratio 0.648350539, minimum log plastic Jacobian
−0.420240034, and emitted water gap 0.631821917 m. Tracked native
sources passed clang-format 19.1.5.

Use the clean-build and Blender commands from the requalification
against 873ae68 with build directory `out/build/m5-main-195af13-clean`
and output directories `out/authored-upstream-195af13-cpu` and
`out/authored-upstream-195af13-cuda`. Q-SAND remains open.

### Symmetric affine decay with wall-face traction

The zero-phase Euler reference uses the same wall-face solver and
motion-aware reconstruction. Apply the existing symmetric stored
affine decay at 8 s⁻¹. The skew part and current-step deformation
gradient are unchanged.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 37.315521° | 0.171163 m |
| Coarse half-step | 41.554803° | 0.150835 m |
| Fine | 32.336228° | 0.183392 m |
| Fine half-step | 34.583904° | 0.178144 m |

The fine nominal angle enters the acceptance range. Its radius
still exceeds 0.16 m. Applying the same decay only in the floor
layer gives 32.138315° / 0.185583 m. Applying it only in the interior
gives 31.879106° / 0.185418 m. Both locations contribute to the
observed improvement. The coarse angles remain above 35°. The
temporary Euler override was restored after compiling.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-wall-decay-bin'
foreach ($case in 'fine', 'fine-half', 'coarse', 'coarse-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 8 $case all 3
}
foreach ($region in 'floor', 'interior') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 8 fine $region 3
}
```

### Floor-supported cone with wall-face traction

Run the existing 45° cone fixture with 371 midpoint samples, zero
floor clearance, and the wall-face solver with the specified Euler
increment. Retain the column matrix's grid widths, timesteps, material
and 3 s endpoint. This control isolates the role of release geometry
and drop. The column remains the qualification fixture.

| Native case at 3 s | Envelope angle | Quadrant slope range |
|---|---:|---:|
| Coarse | 39.208113° | 49.014896–53.987325° |
| Coarse half-step | 40.001391° | 48.623546–51.312037° |
| Fine | 39.324139° | 50.432561–53.494205° |
| Fine half-step | 39.973375° | 49.341096–51.974507° |

All final maximum radii equal the initial sampled radius of 0.160078 m
at printed precision. Late envelope drift is below 0.031°, kinetic
energy below 4e-5 J, and vertical momentum residual below 1.8e-7 N·s.
The cone reaches a consistent steep shape across the four cases.
The steep result therefore persists without the column impact.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-wall-cone-bin'
foreach ($grid in 'fine', 'coarse') {
  foreach ($step in 'nominal', 'half') {
    & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-wall-cone $grid $step
  }
}
```

The temporary entry point enables the existing cone fixture with
particle multiplier two, zero release height, and compatible floor
traction. Both solver and harness sources were restored after linking.

### Floor friction in the supported cone control

Keep the fine nominal cone, the 3 s endpoint, and the 30° material.
Change the stationary floor friction coefficient alone.

| Floor friction | Envelope angle | Radius | Quadrant slope range |
|---:|---:|---:|---:|
| 0.70 | 39.324139° | 0.160078 m | 50.432561–53.494205° |
| 0.35 | 36.998082° | 0.160488 m | 50.651142–50.706666° |
| 0 | 0.543053° | 0.450327 m | 0° |

Halving the floor coefficient decreases the envelope angle by 2.33°
with less than 0.5 mm additional radial spread. The surface fits
stay above 50° in that run. The frictionless cone spreads nearly
flat. Thus the supported shape is sensitive to basal friction,
while the moderate change leaves a steep fitted surface. The new
runs retain vertical momentum residual below 9e-8 N·s. The original
column and floor coefficient still govern qualification.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-wall-cone-friction-bin'
foreach ($friction in 'none', 'low') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-wall-cone-friction $friction
}
```

The temporary harness changes only the floor coefficient in the
fine nominal cone control. The solver and harness sources were
restored after linking.

### Inclined-layer traction admitted by the specified material

The specified 30° Drucker–Prager cone is calibrated in triaxial
compression. For a cohesionless incline, take pressure `p` normal
to the slope and shear traction `p*t`, where `t = tan(slope)`.
With the two lateral compressive stresses both equal to `a*p`,
the squared deviator-to-compressive-trace ratio is
`[(2/3)(a-1)^2 + 2t^2] / (2a+1)^2`. Its minimum for this traction
occurs at `a = 1 + 2t^2`, giving `2t^2 / (9 + 12t^2)`.

The squared cone limit for the specified material is `8/75`.
This stress family reaches yield at `t = sqrt(4/3)`, or
49.106605°. It includes stress states that are admissible at slopes
above 35°. This calculation assumes freely selectable lateral
stress. It makes no prediction of the lateral stress, deformation
compatibility, or final shape in the finite pile.

The shared constitutive oracle constructs the principal stretches
for traction ratios at 0.999 and 1.001 times that limit. It checks
that the former stays elastic, the latter yields, and both retain
plastic volume. The native test and actual SM120 constitutive kernel
both pass. The original material law and Q-SAND acceptance window
remain in place.

```powershell
$b = 'out/build/m5-main-195af13-clean'
cmake --build $b --target fluidity_m5_constitutive_tests fluidity_cuda_m5_constitutive_tests
ctest --test-dir $b -R '^M5Constitutive\.' --no-tests=error --output-on-failure
```

### Accepted stresses in the settled cone

Probe the fine nominal wall-face cone at 3 s with the specified
Euler increment. Select particles at least 0.025 m above the floor
and at least 0.025 m from the axis. Evaluate each accepted Kirchhoff
stress and divide its deviator norm by 0.326598632 times the
compressive trace magnitude. Evaluate traction on a plane whose
outward normal tilts radially by the measured envelope angle.

All 60 sampled particles are in compression. Mean yield utilization
is 0.961280208, and 49 samples reach at least 95% of yield. The
maximum traction angle on the chosen planes is 29.737809°, and no
sample exceeds 35°. The envelope itself is 39.324139°. The material
admits higher traction angles in the analytic inclined-layer family,
but these sampled cone stresses do not exhibit them on the chosen
planes. The probe uses a common envelope tilt and samples interior
particles, so it does not establish local free-surface traction.

```powershell
& 'out/pr-m5-sand-affine/out/sand-cone-stress-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-cone-stress
```

The temporary diagnostic reads the accepted state with
`EvaluateAcceptedDrySolid3D` after the cone step loop. The source
and fixture were restored after linking. Q-SAND remains open.

### Release-shape dependence at constant cone mass

Rescale the 371-particle fine nominal cone by
`(s, 1/s², s)` with `s = cbrt(1 / tan(target_slope))`.
This changes the footprint and height while keeping the same
particle masses, reference volumes, total mass and density.
The material, friction coefficient, Euler increment, grid and
3 s endpoint stay fixed.

| Initial nominal cone slope | Initial sampled envelope | Final envelope | Final radius | Final quadrant slopes |
|---:|---:|---:|---:|---:|
| 25° | 25.331126° | 20.347047° | 0.223779 m | 25.587645–27.620871° |
| 35° | 35.405264° | 30.833591° | 0.180274 m | 40.218763–45.408901° |
| 45° | 45.430165° | 39.324139° | 0.160078 m | 50.432561–53.494205° |

The final envelope retains the initial shape ordering. In the 35°
control the envelope is inside the target angle window, while all
four quadrant fits are above it. That command returns success under
the cone-specific criteria. The original column gate also requires
radius below 0.16 m and remains open. Both new runs retain vertical
momentum residual below 2.8e-8 N·s.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-cone-start-angle-bin'
foreach ($angle in '25', '35') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-cone-start-angle $angle
}
```

The temporary harness applies the transform to positions only and
keeps the cone selection and particle mass unchanged. Both sources
were restored after linking.

### Lamé pair with wall-face traction

Repeat the fixed-step column matrix using the wall-face solver and
Euler deformation. Change only the Lamé pair from (700, 450) Pa to
(1280, 160) Pa. The material friction angle stays at 30° and the
longitudinal modulus stays at 1600 Pa.

| Native case at 3 s | Baseline angle / radius | Alternate Lamé pair angle / radius |
|---|---|---|
| Coarse | 35.932364° / 0.178563 m | 36.161317° / 0.175584 m |
| Coarse half-step | 41.107594° / 0.153975 m | 41.023197° / 0.153872 m |
| Fine | 29.704868° / 0.200837 m | 31.350540° / 0.197523 m |
| Fine half-step | 34.345649° / 0.179405 m | 32.690472° / 0.184132 m |

The coarse angles move by less than 0.23°. Both fine angles enter
or stay in the acceptance range, with radii above 0.16 m. All four
runs finish with vertical momentum residual at most 2.01e-8 N·s.
The temporary source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-wall-lame-bin'
foreach ($grid in 'fine', 'coarse') {
  foreach ($step in 'nominal', 'half') {
    & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-wall-lame $grid $step
  }
}
```

### Swept deformation split with wall-face traction

Use the zero-phase Euler reference with wall-face traction and
motion-aware donor reconstruction. Enable the existing granular
deformation split on the compatible floor. The incoming gradient
advances to the sweep's impact fraction, followed by the
Coulomb-projected gradient for the remaining interval. The stored
velocity and affine update are unchanged.

| Native case at 3 s | Full reflected gradient angle / radius | Swept split angle / radius |
|---|---|---|
| Coarse | 35.932364° / 0.178563 m | 21.358074° / 0.323951 m |
| Coarse half-step | 41.107594° / 0.153975 m | 39.898691° / 0.160444 m |
| Fine | 29.704868° / 0.200837 m | 25.393760° / 0.206636 m |
| Fine half-step | 34.345649° / 0.179405 m | 31.742018° / 0.176048 m |

The nominal cases spread further and fall below the angle range.
The fine half-step stays in the angular window and exceeds the
radius bound. The reported cut-stress pairing still tests the
reflected gather and transpose. It does not account for the
subsequent projected strain increment. The temporary source was
restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-wall-split-deformation-bin'
foreach ($case in 'fine', 'fine-half', 'coarse', 'coarse-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Retain the normal gradient in the swept split

In the wall-face Euler control, keep the full reflected normal row
during the post-impact interval and apply the swept projection to
the two tangential rows. This keeps the wall-normal compression
derivative while isolating the tangential part of the extra
deformation projection. The stored affine fit is unchanged.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 25.609362° | 0.312922 m |
| Coarse half-step | 42.584159° | 0.154748 m |
| Fine | 28.573689° | 0.207809 m |
| Fine half-step | 33.708095° | 0.184830 m |

Retaining the normal row raises both nominal angles compared with
the full swept projection, but they stay below 30°. The coarse
nominal radius remains above 0.31 m, and the fine nominal mean
log plastic Jacobian rises to 1.318743. Vertical momentum residual
stays below 1.9e-8 N·s. The temporary source was restored after
linking. Q-SAND remains open.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-wall-normal-gradient-bin'
foreach ($case in 'fine', 'fine-half', 'coarse', 'coarse-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Swept friction at persistent loaded contact

On the wall-face Euler path, keep the swept normal constraint
throughout. Set the sweep friction to zero for particles starting
at the floor while the grid has a positive total normal reaction.
Use that same coefficient for the stored affine projection.
New impacts retain the original swept friction.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 0° | 0.230846 m |
| Coarse half-step | 7.786764° | 0.229259 m |
| Fine | 24.638068° | 0.195629 m |
| Fine half-step | 31.036888° | 0.176448 m |

The coarse nominal pile collapses onto the floor. Both coarse
runs spread much farther. This control shows that the grid
traction alone does not supply enough tangential resistance when
the sweep still carries part of the normal impulse. The switch
uses a global positive reaction, so it does not determine whether
each persistent particle has local grid support. Vertical momentum
residual stays below 4.9e-8 N·s. Source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-wall-persistent-sweep-bin'
foreach ($case in 'fine', 'fine-half', 'coarse', 'coarse-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Swept friction without tangential grid traction

Keep the reflected normal stress reconstruction, Euler deformation,
and original swept contact and affine projection. Disable only the
wall-face tangential grid correction.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 12.847311° | 0.162870 m |
| Coarse half-step | 20.832620° | 0.164623 m |
| Fine | 17.595499° | 0.176308 m |
| Fine half-step | 21.824114° | 0.184563 m |

All four cases finish below 30°. Both forms of friction materially
affect the geometry in this split wall solver. The removed grid
traction was carried by the normal stress reaction, while the
remaining sweep uses its own incident normal velocity. Vertical
momentum residual stays below 3.5e-9 N·s, and stress-work mismatch
below 2e-16 J. Source was restored after linking. Q-SAND remains open.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-wall-sweep-only-bin'
foreach ($case in 'fine', 'fine-half', 'coarse', 'coarse-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Accumulated Coulomb impulse across wall sweeps

Replace the two half-cap wall traversals with four alternating
block-coordinate sweeps. At each wall point, subtract the velocity
contribution of its previous impulse, minimize grid kinetic energy
over the full local Coulomb disk, and scatter only the impulse
difference. The stored impulse therefore stays inside one disk
regardless of sweep count. The normal reaction, reflected
deformation gather and Euler increment are unchanged. Native and
SM120 dry-scene targets passed, including the loaded wall oracle.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 35.399164° | 0.177291 m |
| Coarse half-step | 41.111856° | 0.153680 m |
| Fine | 30.504264° | 0.186557 m |
| Fine half-step | 33.581269° | 0.180036 m |

The fine nominal case enters the angle window but exceeds the
radius bound. The coarse half-step angle changes by only 0.0043°
relative to the two half-cap sweeps. The block-coordinate test
uses a fixed sweep count and does not certify convergence.
Cut stress work mismatch stays below 2.2e-16 J and vertical
momentum residual below 7.9e-8 N·s. Both sources were restored
after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-wall-accumulated-friction-bin'
foreach ($case in 'fine', 'fine-half', 'coarse', 'coarse-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Wall-normal velocity clamp with local friction capacity

Keep the wall-face Euler reference and clamp inward velocity on
normal faces on the plane inside the active contact patch. Add
the clamp impulse to the signed grid wall budget and to the same
wall point's normal load for the tangential Coulomb solve. The
particle sweep still resolves remaining inward motion.

| Native case at 3 s | Angle | Radius | Grid / swept normal impulse |
|---|---:|---:|---:|
| Coarse | 37.862132° | 0.179583 m | 65.889 / 3.630 N·s |
| Coarse half-step | 42.345620° | 0.160159 m | 64.733 / 4.789 N·s |
| Fine | 33.339651° | 0.184599 m | 66.038 / 3.488 N·s |
| Fine half-step | 36.760331° | 0.180918 m | 65.284 / 4.229 N·s |

The grid supplies roughly 93–95% of the total normal impulse.
The fine nominal angle enters range, with its radius still above
0.16 m. The other angles exceed 35°. Vertical momentum residual
stays below 1.3e-8 N·s and cut stress work mismatch below 2e-16 J.
The temporary solver source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-wall-normal-clamp-bin'
foreach ($case in 'fine', 'fine-half', 'coarse', 'coarse-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Wall work through impact and settling

Trace the accepted work of the wall-face Euler reference through
the existing 1.5 s contact fixture. Grid work is the tangential
kinetic energy change across the face impulse. Sweep work includes
both normal and tangential changes in particle kinetic energy.
The table reports positive amounts of energy removed.

| Native case | Grid at ~0.35 s | Sweep at ~0.35 s | Grid at 1.5 s | Sweep at 1.5 s |
|---|---:|---:|---:|---:|
| Coarse | 0.576337 J | 0.646501 J | 0.736930 J | 0.774444 J |
| Coarse half-step | 0.329062 J | 0.634057 J | 0.448686 J | 0.704807 J |
| Fine | 0.488061 J | 0.578701 J | 0.650157 J | 0.701022 J |
| Fine half-step | 0.298457 J | 0.583086 J | 0.423979 J | 0.636900 J |

By 0.35 s, halving the timestep reduces the grid friction work
magnitude by 43% on coarse and 39% on fine. Sweep work changes
by less than 2% on each grid at that time. The larger grid loss
in the nominal runs accompanies a lower final envelope. The work
depends on both the reaction and incoming tangent velocity, so
this trace does not isolate which drives the difference.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-wall-energy-bin'
foreach ($grid in 'coarse', 'fine') {
  foreach ($step in 'nominal', 'half') {
    & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-contact-trace $grid $step
  }
}
```

The temporary harness sums the grid work budget at the existing
trace times. The solver and harness were restored after linking.
Q-SAND remains open.

### Friction capacity and incoming wall slip

At each accepted wall-face solve in the Euler contact trace, sum the
Coulomb impulse cap, the magnitude of the applied impulse, and the
cap times incoming tangential speed. Include both half-cap sweeps.
Divide the latter sum by the cap sum for a load-weighted speed.

| Native case through ~0.35 s | Cap | Applied magnitude sum | Cap-weighted incoming speed |
|---|---:|---:|---:|
| Coarse | 3.781267 N·s | 3.779745 N·s | 0.163485 m/s |
| Coarse half-step | 3.279660 N·s | 3.279174 N·s | 0.105319 m/s |
| Fine | 3.964792 N·s | 3.956191 N·s | 0.132857 m/s |
| Fine half-step | 3.452155 N·s | 3.451862 N·s | 0.090523 m/s |

All four traces use more than 99.7% of their cumulative cap through
this window. The half-steps reduce the cap by approximately 13% and
the weighted incoming speed by 32–36%. The lower slip speed is the
larger factor in the decrease in cap-times-speed. These are sums of
local magnitudes, separate from the net horizontal impulse budget.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-wall-cap-trace-bin'
foreach ($grid in 'coarse', 'fine') {
  foreach ($step in 'nominal', 'half') {
    & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-contact-trace $grid $step
  }
}
```

The temporary budget instrumentation was restored after linking.
All four traces reproduce the previous wall work values. Q-SAND
remains open.

### Interleave tangential stress and friction

In the zero-phase Euler wall-face solver, apply half of the
tangential stress impulse before each half-cap friction sweep.
Keep the normal stress update, reconstructed deformation gather,
and total friction caps fixed. The cut-stress diagnostic continues
to pair the stress transpose with the final gathered field. It
does not measure work separately for the two force kicks.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 35.750690° | 0.178511 m |
| Coarse half-step | 40.977227° | 0.154051 m |
| Fine | 31.571312° | 0.184764 m |
| Fine half-step | 34.177652° | 0.179629 m |

The fine nominal angle enters range and its radius remains high.
The other three cases retain the same failed geometric checks.
Vertical momentum residual stays below 2.2e-8 N·s. Source was
restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-wall-interleaved-stress-bin'
foreach ($case in 'fine', 'fine-half', 'coarse', 'coarse-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Requalification against upstream 2e96fea

The milestone branch integrated verified upstream `2e96fea` at
`1520372`, including the quadratic adaptive runtime and shared
liquid scheduler changes. A fresh Release SM120 build passed all
11 M5 CTest targets. Blender 5.2.3 repeated the 20-frame authored
snow–water bake, checkpoint resume and playback on native and SM120.
Both backends retained minimum snow volume ratio 0.648350539,
minimum log plastic Jacobian −0.420240034 and emitted water gap
0.631821917 m. Tracked native sources passed clang-format 19.1.5.

Use the preceding clean-build and Blender commands with build
directory `out/build/m5-main-2e96fea-clean` and output directories
`out/authored-upstream-2e96fea-cpu` and
`out/authored-upstream-2e96fea-cuda`. Q-SAND remains open.

### Matched absolute timesteps across grids

Use the zero-phase Euler wall-face solver and hold the physical
release fixed. Run each grid with the other grid's half-step
duration. Combine those new runs with the existing same-duration
results.

| Absolute dt | Coarse angle / radius | Fine angle / radius |
|---:|---|---|
| 0.0009375 s | 41.107594° / 0.153975 m | 33.288325° / 0.183801 m |
| 0.00078125 s | 42.318401° / 0.147696 m | 34.345649° / 0.179405 m |

At matched timesteps, the fine pile is 7.82–7.97° shallower and
0.0298–0.0317 m wider. Both fine angles pass the angle check and
both fine radii fail. Both coarse radii pass and both coarse
angles fail. The timestep mismatch in the original matrix therefore
does not explain these differences. The two new cases retain
vertical momentum residual below 2.4e-8 N·s.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-matched-timestep-bin'
foreach ($grid in 'fine', 'coarse') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-matched-timestep $grid
}
```

The temporary fixture only swaps the absolute half-step durations.
Both sources were restored after linking. Q-SAND remains open.

### Grid-width sequence at one timestep

Hold the wall-face Euler path, physical release, 63 particles and
absolute timestep 0.00078125 s fixed. Add grid widths 0.06875 and
0.05000 m to the matched-timestep comparison.

| Grid width | Native angle at 3 s | Radius |
|---:|---:|---:|
| 0.07500 m | 42.318401° | 0.147696 m |
| 0.06875 m | 38.321245° | 0.165608 m |
| 0.06250 m | 34.345649° | 0.179405 m |
| 0.05000 m | 30.190030° | 0.172222 m |

The envelope angle falls at each tested refinement. The radius
first grows and then decreases at the finest tested width. Neither
new case meets both geometric checks. Holding particle sampling
fixed across this sequence changes the number of particles per
cell, so these data do not establish spatial convergence. Both
new runs retain vertical momentum residual below 2.1e-8 N·s and
cut stress work mismatch below 1.1e-16 J.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-resolution-control-bin'
foreach ($grid in 'middle', 'finer') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-resolution-control $grid
}
```

The temporary entry point changes only the grid width. Both sources
were restored after linking. Q-SAND remains open.

### Dense sampling at 0.050 m grid width

Repeat the finest grid-width control with 504 particles instead of
63. Keep the column dimensions, total mass, 0.00078125 s step,
wall-face traction, and Euler deformation increment.

| Particles | Native envelope at 3 s | Radius | Mean log plastic Jacobian |
|---:|---:|---:|---:|
| 63 | 30.190030° | 0.172222 m | 0.139279 |
| 504 | 24.723652° | 0.253418 m | 0.577992 |

The denser release spreads another 0.081196 m and falls below 30°.
Its quadrant surface fits are 35.251946–39.147977°. Its envelope
changes only 0.001450° in the last 0.05 s, and the vertical momentum
residual is 4.84e-8 N·s. The 63-particle angle improvement therefore
does not persist under this particle refinement. The original
column gate remains open.

```powershell
& 'out/pr-m5-sand-affine/out/sand-finer-dense-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-finer-dense
```

The temporary entry point sets the existing particle multiplier to
two in the 0.050 m case. Both sources were restored after linking.

### Overall stiffness with wall-face traction

Multiply both Lamé moduli by four on the zero-phase Euler
wall-face path. Keep the Lamé ratio, 30° friction angle, density,
release geometry, grid widths, timesteps and 3 s endpoint.

| Native case at 3 s | Reference angle / radius | Fourfold stiffness angle / radius |
|---|---|---|
| Coarse | 35.932364° / 0.178563 m | 33.796743° / 0.173991 m |
| Coarse half-step | 41.107594° / 0.153975 m | 39.062128° / 0.159451 m |
| Fine | 29.704868° / 0.200837 m | 31.313130° / 0.179120 m |
| Fine half-step | 34.345649° / 0.179405 m | 35.939279° / 0.166313 m |

Both nominal angles enter the target band. Both radii remain
above 0.16 m. The fine half-step radius improves by 0.013092 m
and the angle moves above 35°. All four runs finish at their
original timesteps with vertical momentum residual below
1.7e-8 N·s. The material calibration and original acceptance
matrix remain unchanged.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-wall-stiffer-bin'
foreach ($grid in 'fine', 'coarse') {
  foreach ($step in 'nominal', 'half') {
    & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-wall-stiffer $grid $step
  }
}
```

The temporary entry point sets the existing modulus multiplier
to four. Both sources were restored after linking.

### Eightfold stiffness control

Repeat the same wall-face Euler column at eight times the
reference Lamé moduli. Keep all other inputs fixed.

| Native case at 3 s | Angle | Radius | Mean log plastic Jacobian |
|---|---:|---:|---:|
| Coarse | 35.401524° | 0.164587 m | 0.161759 |
| Coarse half-step | 38.948569° | 0.157965 m | 0.122788 |
| Fine | 31.594502° | 0.180954 m | 2.527271 |
| Fine half-step | 32.931358° | 0.181184 m | 0.135007 |

Increasing stiffness beyond fourfold does not consistently improve
geometry. The fine half-step angle enters the band, but its radius
increases. The fine nominal mean log plastic Jacobian reaches
2.527271 compared with 0.464481 in the fourfold control. All runs
finish with vertical momentum residual below 1.8e-8 N·s. Source
was restored after linking, and Q-SAND remains open.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-wall-stiffness-eight-bin'
foreach ($grid in 'fine', 'coarse') {
  foreach ($step in 'nominal', 'half') {
    & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-wall-stiffness-eight $grid $step
  }
}
```

### Plastic volume outliers in the stiff fine column

Trace accepted particle log plastic Jacobians in the native fine
nominal runs with fourfold and eightfold stiffness. The upper
percentile uses sorted index floor(0.9*(count-1)). Count values
above one separately on the floor and above it.

| Time | Stiffness | Median | 90th percentile | Maximum | On floor / above floor with log J_pl > 1 |
|---:|---:|---:|---:|---:|---:|
| ~1.50 s | 4× | 0.131147 | 0.254552 | 0.382069 | 0 / 0 |
| ~1.50 s | 8× | 0.130642 | 0.312290 | 0.809588 | 0 / 0 |
| ~2.00 s | 4× | 0.211973 | 0.556112 | 1.661981 | 4 / 0 |
| ~2.00 s | 8× | 0.255306 | 1.082394 | 4.709186 | 8 / 0 |
| ~2.50 s | 4× | 0.264989 | 0.899296 | 1.956542 | 7 / 0 |
| ~2.50 s | 8× | 0.269698 | 1.082394 | 10.692845 | 8 / 0 |

The medians are close at 2.50 s. The eightfold control's maximum
is much larger and all sampled values above one belong to floor
particles. The high plastic volume therefore develops late and
is concentrated at the floor in this trace. Both final geometries
match the prior stiffness results.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-plastic-distribution-bin'
foreach ($scale in 'four', 'eight') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-plastic-distribution $scale
}
```

The temporary instrumentation was restored after linking.
Q-SAND remains open.

### Tangential expansion at plastic-volume threshold crossings

In the eightfold fine nominal Euler run, log each accepted
crossing from log plastic Jacobian at or below one to above one.
Record the post-sweep height and normal velocity and the diagonal
of the reflected gradient used for deformation.

All nine crossings occur at zero accepted height and zero
normal velocity. The normal derivative is compressive in all
nine, ranging from −0.1894 to −0.00146 s⁻¹. Both tangential
derivatives are positive. In eight crossings the sum of the
tangential derivatives is 6.16–10.39 s⁻¹. The remaining crossing,
particle 50, has L_xx = 82.462300 s⁻¹ and advances log J_pl from
0.988310 to 1.099056 in one step. The large expansion is
therefore in the tangent field at these crossings.

```powershell
& 'out/pr-m5-sand-affine/out/sand-apex-crossing-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-apex-crossing
```

The temporary trace reads the accepted state and deformation
gradient immediately before saving the new plastic Jacobian.
Both experimental sources were restored after linking. The final
pile geometry matches the previous eightfold stiffness control.

### Bound extrapolating tangential ghost rows

Local sand commit `abf860b` uses the specified Euler deformation
increment. If a tangential ghost fit contains a coefficient below
−1e-12, replace that row with a unit coefficient on the nearest
local donor. Use the same row in the gather and stress transpose.
The fallback preserves constants and bounds the ghost velocity
by donor values. It gives up affine reproduction on that row and
uses the existing fallback flag for affine feedback. Normal ghost
rows keep the previous treatment.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 35.006469° | 0.165391 m |
| Coarse half-step | 42.000223° | 0.149039 m |
| Fine | 31.546988° | 0.153024 m |
| Fine half-step | 38.280805° | 0.148133 m |

The fine nominal case passes at reference stiffness.
Its native and SM120 output logs match byte for byte. Its eightfold
diagnostic also passes the geometry at 32.188881° / 0.150819 m,
with mean log plastic Jacobian 1.041356, down from 2.527271.
Both dry-scene targets pass their default tests. All native pile
runs retain stress-work mismatch below 5e-16 J and vertical
momentum residual below 1.6e-8 N·s. The other three original cases
still miss at least one geometric check.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-bounded-tangent-bin'
foreach ($case in 'fine', 'fine-half', 'coarse', 'coarse-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
& "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-bounded-tangent-stiff
& "$bin/fluidity_cuda_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 fine all 3
```

### Positive-weight tangent reconstruction

Starting at `abf860b`, replace the unit nearest-donor row with the
positive part of the original affine-fit coefficients, normalized
to sum to one. Apply the same weights in the gather and stress
transpose. This retains multiple donors without negative weights
or a change to the Euler deformation update. Native and SM120
dry-scene targets pass their default tests.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 35.450470° | 0.167837 m |
| Coarse half-step | 41.833675° | 0.149897 m |
| Fine | 32.123629° | 0.161057 m |
| Fine half-step | 38.839351° | 0.149937 m |

The fine nominal radius misses the bound by 1.057 mm. The other
three cases retain their prior failed geometric checks. Vertical
momentum residual stays below 1.2e-8 N·s and cut stress work
mismatch below 3.2e-16 J. The source was restored after linking,
leaving the nearest-donor experiment at `abf860b`.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-positive-tangent-bin'
foreach ($case in 'fine', 'fine-half', 'coarse', 'coarse-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Component-wise affine feedback after ghost fallback

Local commit `eb8877c` retains the fitted velocity slope rows whose
component stencils use no ghost fallback. A fallback clears only
its own velocity component's row. The bounded tangent ghost rule,
paired stress transpose, and Euler deformation gradient remain
unchanged. Native and SM120 dry-scene targets pass.

| Native case at 3 s | Angle | Radius | Result |
|---|---:|---:|---|
| Coarse | 34.329576° | 0.168211 m | Radius |
| Coarse half-step | 40.501234° | 0.156712 m | Angle |
| Fine | 31.443051° | 0.159557 m | Pass |
| Fine half-step | 34.693822° | 0.156304 m | Pass |

Both fine cases pass the original checks.
Native and SM120 fine half-step logs match byte for byte. The nominal fine radius
has 0.443 mm of margin. Across the matrix, cut stress work mismatch
is below 1.9e-16 J, vertical momentum residual below 9.3e-9 N·s,
and late angle drift below 0.13°. Both coarse cases remain open.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-component-affine-bin'
foreach ($case in 'fine', 'fine-half', 'coarse', 'coarse-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
& "$bin/fluidity_cuda_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 fine-half all 3
```

### Refit affine rows without fallback samples

Starting at `eb8877c`, refit each affected stored affine row from
non-fallback samples in the same component stencil. Use quadratic
sample weights and the existing bounded QR fit. Keep a zero row
when the remaining samples lack full rank. The reconstructed
deformation gradient and paired stress transpose are unchanged.
Both native and SM120 default dry-scene targets pass.

| Native case at 3 s | Clear affected row angle / radius | Refit affected row angle / radius |
|---|---|---|
| Coarse | 34.329576° / 0.168211 m | 24.948769° / 0.186773 m |
| Coarse half-step | 40.501234° / 0.156712 m | 27.879467° / 0.180336 m |
| Fine | 31.443051° / 0.159557 m | 19.421900° / 0.195526 m |
| Fine half-step | 34.693822° / 0.156304 m | 23.177521° / 0.181263 m |

The refit lowers all four angles and increases all four radii.
It overshoots the desired coarse half-step angle change and loses
both fine-grid passes. Vertical momentum residual stays below
2.1e-8 N·s. The temporary source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-refit-affine-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Full-stencil moment scale for partial affine support

Starting at `eb8877c`, compute a centered velocity-position moment
for each affected affine row over its non-fallback samples. Use
the original quadratic weights and multiply by `4/h²`, the inverse
second moment of the full uniform quadratic stencil. The partial
moment is left unnormalized by surviving weight. This avoids
inverting a covariance built from sparse support. The paired
stress and deformation operators are unchanged. Native and SM120
default dry-scene targets pass.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 28.067542° | 0.175966 m |
| Coarse half-step | 30.804592° | 0.171313 m |
| Fine | 23.274535° | 0.178705 m |
| Fine half-step | 27.847757° | 0.170715 m |

The coarse half-step enters the angular window and exceeds the
radius limit. Both fine cases lose their passes. The fixed moment
scale moderates the spreading relative to the full refit, with
vertical momentum residual below 2.5e-8 N·s. Source was restored
after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-moment-refit-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Isolate the normal and tangential partial moments

Split the full-stencil moment control by velocity component.
Refit only the affected tangential rows in one run and only the
affected normal row in the other. Clear the other affected rows
as in `eb8877c`. Both native and SM120 default dry-scene suites pass
for both controls.

| Native case at 3 s | Tangential moment angle / radius | Normal moment angle / radius |
|---|---|---|
| Coarse | 27.977012° / 0.176039 m | 34.493224° / 0.167806 m |
| Coarse half-step | 30.744058° / 0.172182 m | 40.454623° / 0.156861 m |
| Fine | 22.644307° / 0.183904 m | 31.369418° / 0.159845 m |
| Fine half-step | 27.784469° / 0.171326 m | 34.725458° / 0.159911 m |

The tangential-only moment reproduces most of the all-row change,
including loss of both fine passes. The normal-only moment retains
both fine passes with less than 0.16 mm of radius margin and leaves
the coarse failure modes intact. Vertical momentum residual stays
below 2.7e-8 N·s. The source was restored after both builds.

```powershell
foreach ($bin in 'sand-tangent-moment-bin', 'sand-normal-moment-bin') {
  foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
    & "out/pr-m5-sand-affine/out/$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
  }
}
```

### Off-diagonal tangential affine feedback

Use the tangential partial moment with the full-stencil scale, but
clear its diagonal entry on each affected row. Keep the normal
fallback and the deformation gather unchanged. This isolates the
off-diagonal feedback from the tangent row's direct extensional
component. Native and SM120 default dry-scene tests pass.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 30.833997° | 0.174635 m |
| Coarse half-step | 36.704327° | 0.166946 m |
| Fine | 28.047715° | 0.165407 m |
| Fine half-step | 30.156867° | 0.165907 m |

Compared with the full tangential moment, this variant raises all
four angles and reduces three radii. The coarse half-step angle
is closer to range than with row clearing, but both coarse
geometric failures and both fine radius failures remain. Vertical
momentum residual stays below 6.6e-9 N·s. Source was restored
after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-offdiagonal-moment-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Ignore zero-weight samples when clearing an affine row

Starting at `eb8877c`, check the raw quadratic stencil weight
before clearing a fitted row for a ghost fallback. A zero-weight
sample can no longer clear the row. Use the raw stencil so a
nonzero ghost contribution remains visible even when the
compatible deposit excluded the face. The paired deformation
gather and force transpose are unchanged.

All four 3 s native output logs match the row-clearing reference
byte for byte. The native and SM120 default dry-scene targets
also pass. Exactly zero-weight stencil entries do not explain
the remaining coarse failures. The temporary source was
restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-zero-weight-affine-bin'
foreach ($case in 'fine', 'fine-half', 'coarse', 'coarse-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Mass-and-distance selection for bounded tangent donors

Starting at `eb8877c`, choose the unit-weight tangent fallback donor
by maximum `deposited_mass / (1 + distance_in_cells²)`. This is the
same score used by the rank-deficient fallback. The grid gather,
stress transpose, and component-wise affine clearing still use
the same bounded row. Native and SM120 default dry-scene targets
pass.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 35.903882° | 0.181555 m |
| Coarse half-step | 41.090262° | 0.161172 m |
| Fine | 32.307622° | 0.170290 m |
| Fine half-step | 37.102970° | 0.161970 m |

The mass-biased choice increases all four radii and loses both fine
passes. Vertical momentum residual stays below 6.2e-9 N·s and
cut stress work mismatch below 4.6e-16 J. The nearest-donor
candidate was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-mass-tangent-bin'
foreach ($case in 'fine', 'fine-half', 'coarse', 'coarse-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Average equally near tangent fallback donors

Replace the first nearest donor with an equal-weight average over
all nearest donors, within 1e-12 squared cell units of the minimum.
This removes owned-slot tie ordering from the bounded tangential
row. The same row drives both the gather and stress transpose.
Native and SM120 default dry-scene targets pass.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 34.622626° | 0.170127 m |
| Coarse half-step | 40.844653° | 0.157125 m |
| Fine | 31.399622° | 0.161410 m |
| Fine half-step | 34.753734° | 0.158132 m |

The fine half-step still passes. The fine nominal radius grows
past the limit and both coarse failure modes persist. Vertical
momentum residual stays below 5.2e-9 N·s. The source was restored
after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-nearest-tie-bin'
foreach ($case in 'fine', 'fine-half', 'coarse', 'coarse-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Affine row clearing through the impact window

Trace the component-wise candidate `eb8877c` through the existing
1.5 s contact fixture. Count each row cleared by fallback, and sum
its fitted slope norm times the step duration before clearing.
The latter is a dimensionless accumulated norm over all particles.

| Native case through ~0.35 s | Steps | Cleared x / y / z rows | Removed x / y / z norm sums |
|---|---:|---|---|
| Coarse | 187 | 2870 / 0 / 2870 | 3.369 / 0 / 3.364 |
| Coarse half-step | 374 | 5067 / 0 / 5067 | 1.937 / 0 / 1.936 |
| Fine | 224 | 3429 / 37 / 3433 | 4.010 / 0.210 / 4.007 |
| Fine half-step | 448 | 6217 / 252 / 6217 | 2.363 / 0.321 / 2.361 |

Per step and per particle, the cleared tangent fraction is about
21–24% in every case. Normal clearing is rare on fine and absent
on coarse in this window. The integrated tangent slope norm removed
falls by 41–43% at the half-step, despite a comparable fraction of
rows being cleared. This diagnostic measures affine feedback and
does not change the current-step deformation gradient.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-clear-trace-bin'
foreach ($grid in 'coarse', 'fine') {
  foreach ($step in 'nominal', 'half') {
    & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-contact-trace $grid $step
  }
}
```

The temporary budget instrumentation was restored after linking.
The 3 s column matrix remains the qualification reference.

### Apply the affine fallback mask after the sweep projection

Starting at `eb8877c`, carry the row fallback mask through the
swept Coulomb projection and clear the same rows again before
storing the final affine fit. This prevents the derivative of
the sweep from reintroducing a cleared velocity component.
Native and SM120 default dry-scene suites pass.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 34.408843° | 0.167981 m |
| Coarse half-step | 40.498004° | 0.156717 m |
| Fine | 31.443051° | 0.159557 m |
| Fine half-step | 34.693822° | 0.156304 m |

Both fine cases retain their passing geometry at printed precision.
The coarse nominal radius improves by 0.230 mm and still misses.
The coarse half-step angle changes by 0.003230° and still misses.
The source was restored after linking. Q-SAND remains open.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-mask-sweep-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Direct regression for bounded tangent extrapolation

Local sand commit `3008848` adds four full-rank tangent donors on
one side of a missing target. The linear donor velocity is
`v_x = 2x` m/s with cell width 1 m. Ordinary affine extrapolation
gives target velocity 0 m/s. The bounded nearest-donor row gives
2 m/s. With a 3 N target force, each transpose preserves the total
3 N and its grid power matches the corresponding target power,
0 W or 6 W, to 1e-12 in fixture units. The fixture also checks the
fallback flag and the nearest donor's force.

Both native and SM120 dry-scene targets pass. The isolated
reconstruction oracle executes on the host in both targets.
It verifies the bounded operator and its transpose independently
of the pile geometry. The remaining coarse failures keep
Q-SAND open.

```powershell
$b = 'out/pr-m5-sand-affine/out/build/sand-affine-sm120'
cmake --build $b --target fluidity_m5_dry_scene_tests fluidity_cuda_m5_dry_scene_tests
ctest --test-dir $b -R '^M5DryScenes\.' --no-tests=error --output-on-failure
```

### Restrict bounded tangent rows to the wall layer

Starting at `3008848`, restrict the bounded tangential fallback
to targets at most 1.5 cell widths above the floor. This is
the radius of a quadratic stencil. Farther targets retain the
original affine reconstruction even with negative coefficients.
Native and SM120 default dry-scene targets pass.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 31.867744° | 0.171033 m |
| Coarse half-step | 39.317476° | 0.160104 m |
| Fine | 28.010243° | 0.162308 m |
| Fine half-step | 32.703293° | 0.160297 m |

Both fine passes are lost. The coarse half-step angle decreases
but remains above 35°. The restriction shows that bounding the
rows farther from the floor matters to the passing fine cases.
All four runs retain vertical momentum residual below 5.4e-9 N·s
and cut stress work mismatch below 3.1e-16 J. Source was restored
after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-floor-only-bound-bin'
foreach ($case in 'fine', 'fine-half', 'coarse', 'coarse-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Retain the bounded tangent affine fit

Starting at `3008848`, retain the completed field's tangent affine
rows even when they contain bounded ghost samples. Continue
clearing the normal row when its stencil uses fallback. Both
default dry-scene targets pass. All four native qualification
runs reject at the dry face-motion bound before 3 s.

| Native case | First rejected step |
|---|---:|
| Coarse | 284 |
| Coarse half-step | 1073 |
| Fine | 304 |
| Fine half-step | 1052 |

The fine nominal rejecting Z face carries 1.090134e-6 kg and
already moves at 18.824297 m/s after deposition, above the
18 m/s bound. Its stress force is zero. The coarse nominal
rejecting X face begins at −18.163190 m/s, and stress reduces
the magnitude to 18.145946 m/s. These nominal failures come
from the affine deposit's incoming face velocity. They differ
from the earlier stress-driven thin-face failure. Both source
controls were restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-bounded-affine-retain-bin'
foreach ($case in 'fine', 'fine-half', 'coarse', 'coarse-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Limit incoming tangent affine motion

Retain the tangent fit from the bounded field and scale each
tangent row before deposit by a common factor across its
compatible samples. Compute the weighted mean offset on the
same structural support as the deposit. This preserves the
zero weighted mean of the affine contribution. Leave the
normal component, translation, and final motion check in place.

| Native case | Limit at the face speed | Limit at 99% of face speed |
|---|---|---|
| Coarse | Face-motion bound, step 289 | Unlabelled FailedPrecondition, step 316 |
| Coarse half-step | Face-motion bound, step 1095 | Unlabelled FailedPrecondition, step 1110 |
| Fine | Face-motion bound, step 304 | Unlabelled FailedPrecondition, step 330 |
| Fine half-step | Unlabelled FailedPrecondition, step 1074 | Face-motion bound, step 1059 |

Both default native and SM120 dry-scene targets pass at each
setting. With the exact limit, the first fine nominal failure
has zero stress and a deposited speed of 18 m/s against a
computed bound of 17.999999999999744 m/s. The coarse nominal
failure has zero stress and speed 18.011788 m/s on a face with
1.157332e-10 kg. A one-percent margin moves the nominal failures
past those faces but does not recover any full pile run.
Both source controls were restored after linking.

```powershell
foreach ($bin in 'sand-affine-motion-limit-bin', 'sand-affine-limit-margin-bin') {
  foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
    & "out/pr-m5-sand-affine/out/$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
  }
}
```

### Identify the limiter's unlabelled grid rejections

Rebuild the 99%-of-limit affine-deposit control with diagnostics
at the sparse stencil and owner lookup failure sites. Three runs
reject because a face owner lies outside the allocated grid.

| Native case | First rejection | Missing face index / owner block |
|---|---|---|
| Coarse | Step 316 | (−5, −1, −9) / (−2, −1, −3) |
| Coarse half-step | Step 1110 | (12, −1, −1) / (3, −1, −1) |
| Fine | Step 330 | (1, −1, −9) / (0, −1, −3) |
| Fine half-step | Step 1059 face-motion bound | |

The original fixture allocates horizontal block indices −2
through 2. A full extra ring, and a ring confined to the bottom
two block layers, each exceed the existing 24,576-face transfer
capacity and reject at step zero. These runs therefore do not
supply settled geometry. The existing component-wise clearing
candidate remains intact and Q-SAND stays open.

```powershell
foreach ($bin in 'sand-affine-limit-status-bin', 'sand-affine-limit-halo-bin', 'sand-affine-floor-halo-bin') {
  foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
    & "out/pr-m5-sand-affine/out/$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
  }
}
```

The temporary status labels and halo controls were restored
after linking.

### Translation-relative affine limit

Retain the completed tangent fit and cap each centered tangent
affine velocity contribution at the particle's translational
speed, in addition to the face-motion limit. Scale the entire
row by the tightest eligible sample bound. This retains the
zero weighted mean of the affine contribution and leaves the
normal component unchanged. Both default dry-scene targets pass.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 29.729848° | 0.179070 m |
| Coarse half-step | 31.784807° | 0.178050 m |
| Fine | 24.652927° | 0.181935 m |
| Fine half-step | 29.952541° | 0.171396 m |

All four native runs now finish at the original timesteps. The
coarse half-step angle is in range, but all four radii exceed the
limit. The other three angles fall below 30°. Vertical momentum
residual stays below 1.5e-8 N·s. The source was restored after
linking, leaving the component-wise clearing candidate intact.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-relative-limit-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Component-wise translation-relative affine limit

Cap each centered tangent affine contribution by the magnitude
of translation along its own component. Scale the whole row by
the tightest sample bound to preserve its weighted mean.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 42.358288° | 0.150247 m |
| Coarse half-step | 54.403787° | 0.115074 m |
| Fine | 37.182639° | 0.149841 m |
| Fine half-step | 47.609860° | 0.132312 m |

All four runs finish with radii inside the acceptance interval
and vertical momentum residual below 4.3e-9 N·s. Every angle
exceeds 35°. The source was restored after linking. The
component-wise clearing candidate still has the two passing
fine cases, and Q-SAND remains open.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-component-limit-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Midpoint translation-relative affine limit

Set the centered tangent affine limit to the arithmetic mean of
the particle speed and the magnitude of translation along the
affected component. Keep the independent face-motion bound.
The native and SM120 default dry-scene tests both pass.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 31.031017° | 0.176675 m |
| Coarse half-step | 36.377690° | 0.169311 m |
| Fine | 27.002661° | 0.176119 m |
| Fine half-step | 31.463839° | 0.169696 m |

Every radius exceeds 0.16 m. The midpoint puts two angles in
range but fails all four complete pile checks. The largest
absolute vertical momentum residual is 5.52e-9 N·s. This
control gives no basis for replacing the two-pass clearing
candidate. The source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-mean-limit-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Limit tangent affine deposit only on cut support

Apply the component-wise translation limit and the face-motion
limit only to tangent stencils with a positive-weight sample
outside the structurally eligible floor support. Retain the
gathered tangent affine rows as in the other deposit limiter
controls. Both default dry-scene tests pass on native and SM120.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 30.681116° | 0.173318 m |
| Coarse half-step | 36.931411° | 0.166094 m |
| Fine | 26.197639° | 0.172307 m |
| Fine half-step | 30.522757° | 0.168428 m |

Every radius exceeds 0.16 m. Restricting the limiter to cut
support keeps the runs within the motion and grid bounds but
loses both fine-grid passes of the clearing candidate. The
maximum absolute vertical momentum residual is 9.42e-9 N·s.
The source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-cut-limit-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Full extended native and SM120 parity for affine row clearing

Run the three previously unchecked SM120 cases using the
component-wise clearing binaries from the matrix above. Compare
the complete logs with SHA-256. All three match byte for byte,
as did the previously recorded fine half-step case.

| Case at 3 s | Angle | Radius | Native and SM120 result |
|---|---:|---:|---|
| Coarse | 34.329576° | 0.168211 m | Radius fails |
| Coarse half-step | 40.501234° | 0.156712 m | Angle fails |
| Fine | 31.443051° | 0.159557 m | Pass |
| Fine half-step | 34.693822° | 0.156304 m | Pass |

The same coarse failures reproduce on both backends. Q-SAND
remains open. Parity here covers the four extended pile cases
with the component-wise clearing candidate.

```powershell
$s = 'out/pr-m5-sand-affine'
$bin = "$s/out/sand-component-affine-bin"
$logs = "$s/out/floor-wall-gradient-native"
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  $native = "$logs/component-affine-$case.log"
  $cuda = "$logs/component-affine-cuda-$case.log"
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3 *> $native
  $nativeExit = $LASTEXITCODE
  & "$bin/fluidity_cuda_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3 *> $cuda
  $cudaExit = $LASTEXITCODE
  if ($nativeExit -ne $cudaExit -or (Get-FileHash $native).Hash -ne (Get-FileHash $cuda).Hash) {
    throw "Backend mismatch for $case"
  }
}
```

### Extended settling with component-wise affine clearing

Starting from the clean candidate at 3008848, temporarily
change the explicit endpoint in the symmetric-affine-decay
diagnostic from 3 s to 6 s. Use zero decay and keep the scene,
timesteps and acceptance checks. Both default dry-scene CTest
targets pass with the diagnostic change. The extended matrix
below ran on native.

| Case | Angle at 3 s | Angle at 6 s | Radius at 6 s | Result at 6 s |
|---|---:|---:|---:|---|
| Coarse | 34.329576° | 28.832126° | 0.168211 m | Angle and radius |
| Coarse half-step | 40.501234° | 34.203702° | 0.164440 m | Radius |
| Fine | 31.443051° | 25.811618° | 0.159557 m | Angle |
| Fine half-step | 34.693822° | 30.869627° | 0.156304 m | Pass |

All four angles fall by more than 3.8° between 3 s and 6 s,
despite passing the 0.05 s angle-drift check at both endpoints.
The coarse half-step spreads another 7.728 mm and misses the
radius limit after entering the angle window. Longer settling
therefore does not resolve Q-SAND, and the two passing fine
cases at 3 s do not demonstrate long-time repose. Source was
restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-6s-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 6
}
```

### Locate plastic volume accumulation during late settling

Repeat the six-second clearing candidate runs and accumulate
absolute changes in log plastic Jacobian from 3 s to 6 s.
Classify floor residents using the existing contact tolerance
at both step endpoints. The floor layer includes particles
ending the step within one initial spacing (0.05 m) of the
floor. The layer column overlaps the other columns.

| Case | Total per particle | Floor residents | Above contact tolerance at both endpoints | Floor layer |
|---|---:|---:|---:|---:|
| Coarse | 0.01159529 | 0.01140852 | 0.00018676 | 0.01159529 |
| Coarse half-step | 0.03887955 | 0.01356335 | 0.02530078 | 0.03864232 |
| Fine | 0.01670783 | 0.01653805 | 0.00016318 | 0.01670783 |
| Fine half-step | 0.00761967 | 0.00754693 | 0.00007274 | 0.00754693 |

At least 99% of the plastic volume path is in the floor layer
in each case. In the coarse half-step case most of the path
occurs above the contact tolerance, so a change limited to
particles already on the floor would miss most of that case's
plastic volume accumulation. All geometry results match the
previous six-second runs at printed precision. Both default
dry-scene tests pass. The diagnostic changes were restored.

To rebuild the trace, use the preceding six-second fixture
change and set the diagnostic accumulation predicate in
TestScene to extended_column && t >= 3.0 for this matrix.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-creep-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 6
}
```

### Affine-exact midpoint donors for bounded tangent reconstruction

Starting from 3008848, search the existing tangent donor list
for a pair whose offsets about the target sum to zero, when
the full-rank fit has a negative coefficient. Use equal
weights on the first such pair in donor order and keep the
stored affine row eligible. The mean of those donor values
reproduces an affine velocity at the target and stays within
the donor range. When there is no pair, retain the candidate's
nearest-donor fallback. The gather and stress transpose use
the same coefficients.

Both default native and SM120 dry-scene tests pass.

| Native case at 3 s | Angle | Radius | Result |
|---|---:|---:|---|
| Coarse | 34.277038° | 0.168270 m | Radius |
| Coarse half-step | 40.490051° | 0.156751 m | Angle |
| Fine | 31.619054° | 0.158899 m | Pass |
| Fine half-step | 34.783849° | 0.155557 m | Pass |

The midpoint control preserves both fine passes and improves
their radius margins. Both coarse cases retain their failed
checks. The largest absolute vertical momentum residual is
5.94e-9 N·s and the largest cut stress work mismatch is
2.10e-16 J. The source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-midpoint-donor-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Prefer a same-height tangent donor

For the full-rank tangent reconstruction with negative
coefficients, choose the closest donor at the target height
when available. Keep the ordinary nearest donor otherwise.
Use the chosen donor for the gather and stress transpose and
keep the fallback flag. Both native and SM120 default
dry-scene tests pass.

| Native case at 3 s | Angle | Radius | Result |
|---|---:|---:|---|
| Coarse | 29.072970° | 0.169978 m | Angle and radius |
| Coarse half-step | 35.141869° | 0.159110 m | Angle |
| Fine | 29.112553° | 0.158882 m | Angle |
| Fine half-step | 32.589855° | 0.153616 m | Pass |

The same-height rule moves the coarse half-step angle down
5.36° but increases mean log plastic Jacobian in every case.
The coarse nominal mean rises from 0.328605 to 1.116065.
The fine nominal angle also falls below range. This control
does not resolve Q-SAND. The source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-level-donor-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Prefer a tangent donor directly above the target

When the full-rank tangent fit requires a negative coefficient,
prefer the closest eligible donor with the target's horizontal
coordinates and a greater height. Keep the ordinary nearest
donor when that column has no donor above the target. Use the
unit-weight row in both the gather and the stress transpose.
Both native and SM120 default dry-scene tests pass.

| Native case at 3 s | Angle | Radius | Result |
|---|---:|---:|---|
| Coarse | 36.226259° | 0.176449 m | Angle and radius |
| Coarse half-step | 41.134146° | 0.157529 m | Angle |
| Fine | 33.404733° | 0.164515 m | Radius |
| Fine half-step | 36.766082° | 0.159392 m | Angle |

All four angles and radii increase relative to nearest-donor
clearing. The fine half-step mean log plastic Jacobian falls
from 0.146 to 0.089, but the pile leaves the angle window.
This control loses both fine passes. The source was restored
after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-vertical-donor-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Sweep-split deformation with bounded tangent clearing

Starting at 3008848, enable the existing impact-split
granular deformation update on the compatible floor. Advance
with the gathered gradient until the sweep hits the floor,
then use its Coulomb-projected derivative for the remaining
interval. The stress transpose still pairs with the gathered
gradient before this projection.

| Native case at 3 s | Angle | Radius | Result |
|---|---:|---:|---|
| Coarse | 37.933422° | 0.168036 m | Angle and radius |
| Coarse half-step | 40.748841° | 0.154928 m | Angle |
| Fine | 31.651993° | 0.161478 m | Radius |
| Fine half-step | 35.020386° | 0.155106 m | Angle |

Neither default dry-scene CTest target passes with this
control. Both exit without a diagnostic identifying the
first failed assertion. All four extended native cases
finish and miss at least one geometric check. The nominal
coarse mean log plastic Jacobian increases from 0.328605
to 0.905271. The temporary source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-clearing-sweep-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

Instrumenting the default test runner identifies
TestWallImpulseBudget(true) as the first failure on both
native and SM120. The isolated fixture starts on the floor
with downward velocity. Its compatible-floor oracle requires
the reflected normal derivative of minus one per second to
compress the elastic determinant to 0.99 over 0.01 s while
the wall impulse stops the inward velocity. The swept split
instead projects the deformation gradient from the start of
the step because contact is immediate. The oracle therefore
tests a change to the wall deformation model independently of
the pile geometry. The instrumentation and solver control
were both restored.

### Retain the compatible normal derivative in the swept split

On the bounded tangent clearing candidate, retain the gathered
normal row in the post-impact gradient and apply the swept
Coulomb projection only to the tangential rows. Both default
dry-scene targets still fail. This run did not instrument their
first failing assertion.

| Native case at 3 s | Angle | Radius | Mean log plastic Jacobian |
|---|---:|---:|---:|
| Coarse | 40.106682° | 0.168126 m | 0.984805 |
| Coarse half-step | 40.721210° | 0.157737 m | 0.170025 |
| Fine | 36.910317° | 0.158798 m | 0.847430 |
| Fine half-step | 40.227663° | 0.154002 m | 1.582018 |

All four angles exceed the gate. The fine half-step accumulates
more than ten times the mean log plastic Jacobian of clearing
alone. The gather/transpose work diagnostic still measures the
unprojected gradient. The temporary source was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-clearing-tangent-sweep-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Cap negative weight in the mass-weighted tangent fit

For mass-weighted full-rank tangent rows with negative
coefficients, blend the affine row with its nearest-donor
fallback. Limit the affine fraction to the smaller of one
and 0.125 divided by the total negative weight before
blending. This preserves constants and caps the negative
weight of the resulting row at 0.125. Use the same row in
the gather and stress transpose. Retain the fallback flag
and use the original nearest-donor rule without masses.

Both default native and SM120 dry-scene tests pass. The
direct unweighted extrapolation oracle continues to check
the original rule.

| Native case at 3 s | Angle | Radius | Result |
|---|---:|---:|---|
| Coarse | 35.361725° | 0.168221 m | Angle and radius |
| Coarse half-step | 41.013383° | 0.156062 m | Angle |
| Fine | 31.906038° | 0.161153 m | Radius |
| Fine half-step | 34.970133° | 0.156398 m | Pass |

Both coarse cases still miss. The fine nominal radius crosses
the limit, leaving only one passing case. The temporary source
was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-negative-budget-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Requalification against upstream d6e8c6e

The milestone branch integrated upstream d6e8c6e at 68c7902,
including the graded liquid band overlap and Stokes relaxation
changes. A clean Windows Release SM120 build passed all 11 M5
CTest targets. Blender 5.2.3 passed the 20-frame authored
snow–water bake, frame-10 checkpoint resume and playback on
native and SM120. Both runs measured minimum snow volume ratio
0.648350539, minimum log plastic Jacobian −0.420240034 and
emitted water gap 0.631821917 m.

Repeat the earlier clean-build and Blender commands using
out/build/m5-main-d6e8c6e-release and the output directories
out/authored-upstream-d6e8c6e-cpu and
out/authored-upstream-d6e8c6e-cuda. Configure with the qualified
Conan Release CUDA toolchain and CLI11 2.7.2 package. The
extended Q-SAND gate remains open.

The tracked native sources pass the full clang-format 19.1.5
check from AGENTS.md. The tracked long-settling sand diagnostic
runs all four 1.5 s column cases on native and SM120. Both
executables exit 1. The final reported geometry and drift
match at printed precision. The nominal coarse, nominal fine,
half-step coarse and half-step fine angles are 29.113507°,
26.761332°, 29.006384° and 27.033753°, respectively. All four
radii are within the declared interval, and all four angles
fall below 30°. Full logs differ in stress budgets, momentum
residuals, plastic path measurements, and one 0.75 s angle.

```powershell
$b = 'out/build/m5-main-d6e8c6e-release'
& "$b/fluidity_m5_dry_scene_tests.exe" --q-sand-column-long-settle
& "$b/fluidity_cuda_m5_dry_scene_tests.exe" --q-sand-column-long-settle
```

The compatible-floor 1.5 s matrix also reproduced the
previous geometry on native and SM120. Both commands exited
1. The logs agree except for the last reported digits of the
two fine-grid vertical momentum residuals. Only the coarse
half-step passed the pile checks. The coarse nominal,
coarse half-step, fine nominal and fine half-step angles
were 29.962967°, 32.042153°, 27.424648° and 29.277572°.

```powershell
$b = 'out/build/m5-main-d6e8c6e-release'
& "$b/fluidity_m5_dry_scene_tests.exe" --q-sand-compatible-floor-long-settle
& "$b/fluidity_cuda_m5_dry_scene_tests.exe" --q-sand-compatible-floor-long-settle
```

### Compare the envelope with quadrant slopes on the clearing candidate

Enable the existing FitSandSurface diagnostic for the original
63-particle release by removing its sampling multiplier guard.
Keep the original acceptance estimator and thresholds. Both
default dry-scene tests pass, and the extended native results
retain their previous geometry at printed precision.

| Case at 3 s | Envelope | Signed quadrant surface slopes |
|---|---:|---:|
| Coarse | 34.329576° | −29.244900° to 0° |
| Coarse half-step | 40.501234° | −53.356394° to −41.730394° |
| Fine | 31.443051° | −21.608041° to 0° |
| Fine half-step | 34.693822° | −38.218587° to −6.538925° |

All four quadrants meet the minimum bin count in each case.
The spread between quadrant fits and the envelope, including
in the two passing fine cases, shows that the sparse release
gives weak evidence for a uniform surface slope. The quadrant
range combines pile shape and sampling effects. The source
was restored after linking. Q-SAND remains open.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-surface-audit-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Double the particle sampling of the clearing candidate

Set the sampling multiplier to two in the extended diagnostic
on 3008848. This gives 504 particles at 0.025 m spacing.
The release dimensions and reference density, grid widths,
timesteps, and acceptance checks are the same. Both default
native and SM120 dry-scene tests pass.

| Native case at 3 s | Envelope | Radius | Quadrant fit |
|---|---:|---:|---:|
| Coarse | 29.225650° | 0.210420 m | 25.436163° to 26.949057° |
| Coarse half-step | 32.676751° | 0.199398 m | 21.111038° to 21.281019° |
| Fine | 25.814558° | 0.221056 m | 13.186272° to 16.096608° |
| Fine half-step | 30.083739° | 0.204909 m | 17.267139° to 22.394147° |

Every dense case spreads farther than the corresponding
63-particle case and exceeds the radius limit. Every fitted
downhill angle falls below 30°. In the preceding sparse
table, negative fitted angles indicate surfaces rising with
radius because the estimator negates the fitted height
derivative. The dense diagnostic resolves downhill surfaces
in all quadrants and exposes shallower slopes than the
envelope. The temporary source was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-dense-audit-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Radial spread with denser sampling

For the same 504-particle clearing runs, sort the final
horizontal radii. Use index floor((count - 1) times percentile)
for each quantile. All particles have equal mass.

| Native case at 3 s | Median radius (m) | 90th percentile (m) | 95th percentile (m) | Count at or beyond 0.16 m |
|---|---:|---:|---:|---:|
| Coarse | 0.094939 | 0.161861 | 0.174068 | 55 |
| Coarse half-step | 0.090580 | 0.153633 | 0.163413 | 36 |
| Fine | 0.102345 | 0.174490 | 0.185365 | 87 |
| Fine half-step | 0.095300 | 0.159896 | 0.170568 | 51 |

Between 7.1% and 17.3% of the mass is outside the radius
limit. Even the 95th percentile exceeds that limit in every
case. The outer radius failure therefore spans many particles.
The existing quadrant fit and pile acceptance checks are
unchanged. Both default native and SM120 dry-scene tests pass.
The temporary source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-dense-radial-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Trace-consistent volume increment with bounded tangent clearing

On 3008848, enable the existing volume scaling for the compatible
floor deformation increment. It rescales the Euler increment to
determinant exp(dt times trace of the gathered gradient). The
gather and stress transpose still use the same velocity field.

| Native case at 3 s | Angle | Radius | Mean log plastic Jacobian |
|---|---:|---:|---:|
| Coarse | 34.417264° | 0.167651 m | 0.326613 |
| Coarse half-step | 40.522656° | 0.156163 m | 0.162437 |
| Fine | 31.458649° | 0.159095 m | 0.228256 |
| Fine half-step | 34.674819° | 0.156565 m | 0.142554 |

The correction changes mean log plastic Jacobian by less than
0.007 in each case and keeps the same two fine passes and two
coarse failures. This control does not implicate the Euler
determinant approximation as the dominant source of the geometry
error in this candidate. Both default CTest targets fail without
an assertion label. The isolated compatible-floor wall fixture
expects the Euler determinant of 0.99 under a normal gradient
of minus one per second, whereas the control uses exp(-0.01).
The temporary source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-trace-volume-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Integrate the skew part of the floor gradient as a rotation

Starting at 3008848, split the compatible-floor gradient into
its symmetric rate and skew spin. Advance the symmetric
gradient with the usual Euler increment and left-multiply the
result by the Rodrigues rotation of the spin over the step.
Pure skew then gives a rotation without a spurious volume or
stretch increment. The gather, stress scatter and acceptance
criteria are unchanged. Both default dry-scene targets pass.

| Native case at 3 s | Angle | Radius | Mean log plastic Jacobian |
|---|---:|---:|---:|
| Coarse | 34.921635° | 0.168019 m | 0.339524 |
| Coarse half-step | 40.497664° | 0.156731 m | 0.164615 |
| Fine | 31.961405° | 0.159452 m | 0.226044 |
| Fine half-step | 34.692508° | 0.156388 m | 0.136926 |

The maximum radius change from ordinary clearing is 0.192 mm.
Both fine cases still pass at 3 s and both coarse failures
persist. The coarse nominal mean log plastic Jacobian rises
by 0.011. The results do not support skew integration error
as the dominant cause of either coarse failure. The temporary
source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-rotation-split-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Halve the floor transfer mass threshold

On 3008848, lower the face force mass floor and donor and
wall-traction eligibility threshold from 1e-4 to 5e-5 times
minimum particle mass. Leave the paired reconstruction and
affine clearing rules in place. Both default dry-scene tests
pass on native and SM120.

| Native case at 3 s | Angle | Radius | Mean log plastic Jacobian |
|---|---:|---:|---:|
| Coarse | 34.361373° | 0.167440 m | 0.325000 |
| Coarse half-step | 40.282181° | 0.156694 m | 0.159777 |
| Fine | 30.974147° | 0.161721 m | 0.357368 |
| Fine half-step | 34.340979° | 0.158384 m | 0.197393 |

The coarse nominal radius improves by 0.771 mm and still
fails. The coarse half-step angle remains above 40°. Both
fine cases accumulate more plastic volume and the nominal
fine case loses its radius pass. The source was restored
after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-half-support-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Isolate the wall traction mass threshold

On 3008848, halve the minimum eligible face mass in the
tangential wall gather, mobility and impulse scatter only.
Keep the stress donor threshold and face acceleration mass
floor at 1e-4 times minimum particle mass. Both default
native and SM120 dry-scene tests pass.

| Native case at 3 s | Angle | Radius | Mean log plastic Jacobian |
|---|---:|---:|---:|
| Coarse | 34.398561° | 0.167928 m | 0.328692 |
| Coarse half-step | 40.522901° | 0.156508 m | 0.164579 |
| Fine | 31.491192° | 0.159365 m | 0.226366 |
| Fine half-step | 34.717691° | 0.156193 m | 0.136328 |

All four radii change by less than 0.3 mm and retain their
original pass or fail status. The fine-grid plastic volume
increase seen when halving the shared threshold does not
reproduce when changing traction eligibility alone. The
temporary source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-half-traction-support-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Isolate the stress support mass threshold

On 3008848, halve the stress donor threshold and the force
acceleration mass floor to 5e-5 times minimum particle mass.
Keep the tangential wall traction threshold at 1e-4 times
minimum particle mass. The same mass floor remains in the
stress work accounting. Both default dry-scene tests pass
on native and SM120.

| Native case at 3 s | Angle | Radius | Mean log plastic Jacobian |
|---|---:|---:|---:|
| Coarse | 34.284955° | 0.167796 m | 0.324151 |
| Coarse half-step | 40.283552° | 0.156850 m | 0.159805 |
| Fine | 30.946031° | 0.162001 m | 0.354115 |
| Fine half-step | 34.300342° | 0.158655 m | 0.197440 |

This reproduces the increased fine-grid plastic volume and
nominal fine radius failure seen with the shared threshold
change. The traction-only control did not reproduce either.
Both coarse failures persist. The temporary source was
restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-half-stress-support-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Raise only the stress donor eligibility threshold

On 3008848, raise the minimum eligible donor mass to
2e-4 times minimum particle mass. The force acceleration
denominator, wall traction and stored velocity eligibility
still use the original floor of 1e-4 times minimum particle
mass. Both default native and SM120 dry-scene tests pass.

| Native case at 3 s | Angle | Radius | Mean log plastic Jacobian |
|---|---:|---:|---:|
| Coarse | 35.177342° | 0.168437 m | 0.330278 |
| Coarse half-step | 40.945395° | 0.155414 m | 0.161780 |
| Fine | 32.048381° | 0.158799 m | 0.187090 |
| Fine half-step | 35.673307° | 0.155929 m | 0.107808 |

Both fine cases accumulate less plastic volume, but the fine
half-step exceeds the angle bound. The coarse nominal case
fails both geometry checks and the coarse half-step remains
too steep. The donor cutoff changes pile geometry even when
the force denominator is held fixed. The source was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-double-donor-support-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Raise only the force acceleration mass floor

On 3008848, double the mass floor in the face acceleration
denominator and the corresponding applied stress work calculation.
Keep donor selection and wall traction at the original thresholds.
Both default dry-scene tests pass on native and SM120.

| Native case at 3 s | Angle | Radius | Vertical momentum residual (N·s) |
|---|---:|---:|---:|
| Coarse | 34.505519° | 0.168238 m | 4.31e-5 |
| Coarse half-step | 40.510052° | 0.156758 m | −5.21e-5 |
| Fine | 31.489858° | 0.159589 m | −1.07e-4 |
| Fine half-step | 34.720394° | 0.156461 m | 8.34e-5 |

Both fine geometry checks still pass and both coarse failures
persist. The largest vertical momentum residual rises from
9.3e-9 to 1.07e-4 N·s. The cut stress work residual stays below
2.1e-16 J because it includes work associated with the omitted
impulse. A good work residual alone therefore does not certify
momentum conservation with this denominator. The source was
restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-double-force-mass-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Use actual deposited mass for compatible-floor acceleration

On 3008848, use the actual deposited mass in the compatible
floor acceleration denominator and account for the full stress
impulse as applied work. Keep donor eligibility and wall traction
unchanged. Both default native and SM120 dry-scene tests pass.

All four native 3 s logs match the original clearing runs byte
for byte. No case hits the motion bound. This confirms that
thin faces excluded from the stress donor set carry no stress
impulse in these runs. The acceleration mass floor therefore
does not drive the two coarse geometry failures at the original
donor threshold. The temporary source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-actual-force-mass-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Remove mass weighting from tangent fits

On 3008848, set tangent affine fit row scales to one and
retain the mass-weighted normal fit and rank fallback. Both
default native and SM120 dry-scene tests pass. Every native
3 s log matches the clearing candidate byte for byte. In this
fixture the tangent fit weighting has no observable effect
after the bounded reconstruction and clearing rules. The
temporary source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-unweighted-tangent-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Remove mass weighting from normal fits

On 3008848, set normal affine fit row scales to one. Retain
the tangent fit and rank fallback. Both default native and
SM120 dry-scene tests pass.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 34.343882° | 0.168194 m |
| Coarse half-step | 40.499841° | 0.156715 m |
| Fine | 31.470486° | 0.159433 m |
| Fine half-step | 34.695889° | 0.156278 m |

The largest angle change is 0.028° and the largest radius
change is 0.124 mm. Both coarse failures persist and both
fine cases retain their 3 s passes. The source was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-unweighted-normal-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Shorten the mass-weighted tangent donor search

On 3008848, reduce the affine fit radius for mass-weighted
tangent rows from two cell widths to sqrt(3) cell widths.
Keep the normal search and the existing constant fallback
radius. Both default dry-scene targets fail after the
stationary ghost transpose work check and before the
mass-weighted work result is printed.

| Native case at 3 s | Angle | Radius | Result |
|---|---:|---:|---|
| Coarse | 34.088360° | 0.171779 m | Radius |
| Coarse half-step | 40.487704° | 0.160588 m | Angle and radius |
| Fine | 31.038061° | 0.163967 m | Radius |
| Fine half-step | 34.204129° | 0.158460 m | Pass |

All radii increase relative to the original clearing runs.
The nominal fine case loses its pass, and the coarse
half-step adds a radius failure. The temporary source was
restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-local-tangent-radius-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Widen the mass-weighted tangent donor search

On 3008848, extend the affine fit radius of mass-weighted
tangent rows from two to sqrt(5) cell widths. Raise the
local host array capacity to 64 for this control. Keep the
normal radius, transpose, and nearest-donor fallback rule.
Both default dry-scene tests pass.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 34.328193° | 0.168211 m |
| Coarse half-step | 40.501234° | 0.156712 m |
| Fine | 31.431318° | 0.159557 m |
| Fine half-step | 34.693731° | 0.156304 m |

All four radii match the original clearing candidate at
printed precision and angles move by less than 0.012°.
Both coarse failures persist. The temporary source was
restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-wide-tangent-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Count tangent continuation fallbacks

On 3008848, count tangent ghost rows and rows marked fallback
at the start of the compatible gather. Both default dry-scene
tests pass. Removing the added trace lines gives the original
native pile output in each case.

| Native case at 3 s | Fallback rows / tangent rows | Fallback rows / tangent rows in final steps covering 0.05 s |
|---|---:|---:|
| Coarse | 124361 / 124361 | 3213 / 3213 |
| Coarse half-step | 186186 / 186186 | 3614 / 3614 |
| Fine | 186424 / 186424 | 4190 / 4190 |
| Fine half-step | 261117 / 261117 | 7912 / 7912 |

Every missing tangent target requiring continuation uses a
fallback throughout the matrix. The counter includes all
fallback reasons and does not distinguish rank deficiency
from rejection of negative affine coefficients. Changing fit
weights can affect these cases only if it changes the fallback
decision or donor. The temporary instrumentation was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-fallback-trace-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Classify tangent fallback attempts

Instrument the donor reconstruction on 3008848 before each
gather-needed tangent fallback. These are construction attempts,
including rows superseded when the motion check removes a donor
and rebuilds the map. Both default dry-scene tests pass. Removing
the trace lines recovers each original native 3 s log.

| Case | Empty local neighborhood | Rank-deficient fit | Negative affine coefficient |
|---|---:|---:|---:|
| Coarse | 0 | 10456 | 128747 |
| Coarse half-step | 0 | 0 | 186186 |
| Fine | 0 | 30254 | 167069 |
| Fine half-step | 0 | 2686 | 258431 |

Negative coefficients trigger 85–100% of fallback attempts
in each case. The coarse half-step never uses a rank fallback.
The accepted-row counts in the preceding trace exclude
discarded construction passes. The temporary instrumentation
was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-fallback-reasons-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3 |
    Where-Object { $_ -like 'fallback reason *' } | Group-Object
}
```

### Average tangent fallback donors by distance

On 3008848, replace the unit nearest-donor coefficient for
mass-weighted full-rank tangent rows with negative fit
coefficients by normalized weights proportional to
1 / (1 + squared distance in cells)^2. The row preserves
constants, stays within the donor range, and is shared by the
gather and stress transpose. Rank-deficient and unweighted
rows keep their prior fallback. Both default dry-scene tests
pass on native and SM120.

| Native case at 3 s | Angle | Radius | Result |
|---|---:|---:|---|
| Coarse | 32.185587° | 0.169696 m | Radius |
| Coarse half-step | 40.482629° | 0.156004 m | Angle |
| Fine | 30.533417° | 0.161951 m | Radius |
| Fine half-step | 34.571130° | 0.158870 m | Pass |

Both coarse failures persist and nominal fine loses its pass.
Maximum vertical momentum residual is 6.65e-9 N·s. The
temporary source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-distance-average-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Project the tangent target onto the two nearest donors

On 3008848, for mass-weighted full-rank tangent rows with
negative coefficients, project the target onto the segment
between the nearest donor and the next nearest donor. Clamp
the projection to the segment and use its two linear weights
for both gather and force scatter. Keep the fallback flag.
Both default native and SM120 dry-scene tests pass.

| Native case at 3 s | Angle | Radius | Result |
|---|---:|---:|---|
| Coarse | 34.104916° | 0.169664 m | Radius |
| Coarse half-step | 40.837993° | 0.157062 m | Angle |
| Fine | 31.204066° | 0.160834 m | Radius |
| Fine half-step | 34.554525° | 0.157460 m | Pass |

The segment rule preserves the fine half-step pass but
increases all four radii. Both coarse failures persist.
The temporary source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-nearest-segment-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Use the nearest donor for rank-deficient tangent fits

On 3008848, keep the geometrically nearest donor for
rank-deficient tangent rows instead of selecting by deposited
mass divided by one plus squared distance. The normal
fallback and full-rank fit rules are unchanged. Both default
native and SM120 dry-scene tests pass.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 34.328114° | 0.168211 m |
| Coarse half-step | 40.501234° | 0.156712 m |
| Fine | 31.431266° | 0.159557 m |
| Fine half-step | 34.693731° | 0.156304 m |

All radii match the baseline at printed precision. The
largest angle change is 0.012°. Both coarse failures
persist. The source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-rank-nearest-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Retain in-plane derivatives on tangent fallback rows

On 3008848, retain the x and z derivatives of fitted tangent
rows that sample fallback faces and clear their y derivative.
Clear the entire affected normal row as before. This changes
the stored affine velocity for the next deposit. The current
deformation gradient still uses the paired ghost field.

Both default dry-scene tests pass on native and SM120.
None of the extended native pile runs reaches 3 s. The
coarse, coarse half-step and fine runs reject on the face
motion bound at steps 530, 1197 and 357, respectively. The
fine half-step returns an unlabeled FailedPrecondition at
step 1986. Removing the wall-normal derivative alone is
insufficient to retain stable affine feedback in this fixture.
The temporary source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-in-plane-retain-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Retain one quarter of the in-plane affine fit on fallback

On 3008848, scale the fitted tangent row's x and z derivatives
by 0.25 when its stencil uses fallback and clear its y derivative.
Clear the affected normal row as before. Both default dry-scene
tests pass on native and SM120. All four extended native runs
finish within the motion bound.

| Native case at 3 s | Angle | Radius | Result |
|---|---:|---:|---|
| Coarse | 32.832010° | 0.172028 m | Radius |
| Coarse half-step | 39.657508° | 0.158163 m | Angle |
| Fine | 30.721120° | 0.159111 m | Pass |
| Fine half-step | 33.967595° | 0.159180 m | Pass |

The partial fit avoids the early rejections of full in-plane
retention. It reduces both coarse angles and increases both
coarse radii, leaving the two original failures. The temporary
source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-quarter-in-plane-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Retain only in-plane spin on tangent fallback rows

On 3008848, retain the skew part of the horizontal 2-by-2
affine fit on each affected tangent row. Use half the
difference of the original off-diagonal derivatives with
opposite signs in the two tangent rows. Other derivatives
on an affected row are cleared. Both default dry-scene
tests pass on native and SM120.

| Native case at 3 s | Angle | Radius | Result |
|---|---:|---:|---|
| Coarse | 33.325732° | 0.170123 m | Radius |
| Coarse half-step | 40.489845° | 0.156731 m | Angle |
| Fine | 31.248287° | 0.159796 m | Pass |
| Fine half-step | 34.525517° | 0.156633 m | Pass |

All four runs reach 3 s. The coarse nominal radius increases
by 1.912 mm and the coarse half-step angle barely changes.
The temporary source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-in-plane-spin-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Retain only in-plane extension on tangent fallback rows

On 3008848, keep the x derivative of the x affine row and
the z derivative of the z row when the corresponding stencil
uses fallback. Clear all other derivatives in affected rows.
Both default native and SM120 dry-scene tests pass.

| Native case | First rejected step |
|---|---:|
| Coarse | 1150 |
| Coarse half-step | 2703 |
| Fine | 370 |
| Fine half-step | 2102 |

All four runs return an unlabeled FailedPrecondition before
3 s. Full extensional feedback alone is sufficient for early
rejection, while the preceding spin-only control completes
the matrix. The temporary source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-in-plane-extension-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Identify the extension-only support failures

Repeat the extension-only control with a trace at the sparse
face owner lookup. The rejection steps match the earlier
runs. Each missing owner lies outside the horizontal extent
of the allocated pile grid.

| Native case | Rejected step | Requested face lattice index |
|---|---:|---|
| Coarse | 1150 | (12, −1, −4) |
| Coarse half-step | 2703 | (−9, −1, −3) |
| Fine | 370 | (−9, −1, −7) |
| Fine half-step | 2102 | (−7, −1, −9) |

Both default native and SM120 dry-scene tests pass. The
extended failures arise in face support lookup. They provide
no settled geometry. The solver and sparse-grid instrumentation
were both restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-extension-status-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Measure spread before extension-only support rejection

On 3008848, repeat the extension-only control and record the
committed particle positions when the first step rejects.
The rejection steps reproduce the previous trace. Both default
native and SM120 dry-scene tests pass.

| Native case | Last accepted time (s) | Radius (m) | Particles at or beyond 0.16 m |
|---|---:|---:|---:|
| Coarse | 2.154375 | 0.699260 | 29 of 63 |
| Coarse half-step | 2.533125 | 0.534872 | 30 of 63 |
| Fine | 0.578125 | 0.505945 | 38 of 63 |
| Fine half-step | 1.642188 | 0.641397 | 44 of 63 |

The pile has already reached more than three times the
radius limit in every case. Between 46% and 70% of the
equal-mass particles are beyond the limit. These are
pre-rejection measurements, not settled radii. Enlarging
the grid alone would leave the observed early spreading
mechanism in place. Both temporary edits were restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-extension-radius-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Isolate compression and expansion in tangent affine feedback

On 3008848, retain only the diagonal derivative on each
affected tangent row. The compression control keeps negative
derivatives and the expansion control keeps positive derivatives.
Both default native and SM120 dry-scene tests pass for each control.

| Native case | Compression-only angle / radius at 3 s | Expansion-only outcome |
|---|---|---|
| Coarse | 34.583093° / 0.167699 m | Unlabeled FailedPrecondition at step 1149 |
| Coarse half-step | 40.508843° / 0.156688 m | 32.022436° / 0.161609 m |
| Fine | 31.475272° / 0.159046 m | Unlabeled FailedPrecondition at step 377 |
| Fine half-step | 34.717325° / 0.156230 m | Unlabeled FailedPrecondition at step 2071 |

Compression-only retains both fine passes and both coarse
failures. Expansion-only reproduces early rejection in three
cases. Its coarse half-step reaches the angle window but exceeds
the radius limit. Positive extensional feedback is sufficient
for the early failures in this control. The sources were restored.

```powershell
foreach ($mode in 'compression-only', 'expansion-only') {
  $bin = "out/pr-m5-sand-affine/out/sand-affine-$mode-bin"
  foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
    & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
  }
}
```

### Cap positive tangent extension per step

On 3008848, retain the positive diagonal derivative of each
affected tangent row up to 0.01 divided by the step duration.
This caps the dimensionless derivative times duration at 0.01.
Clear the other derivatives on affected rows. Both default
native and SM120 dry-scene tests pass.

| Native case | Outcome at 3 s |
|---|---|
| Coarse | 30.142670° / 0.172523 m, radius fails |
| Coarse half-step | 32.976917° / 0.159007 m, passes |
| Fine | 19.749158° / 0.216608 m, angle and radius fail |
| Fine half-step | Unlabeled FailedPrecondition at step 2310 |

The cap gives the coarse half-step its first complete pass
in this extension control, with 0.993 mm of radius margin.
The fine results show that the same bound does not transfer
across grid widths. The temporary source was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-bounded-extension-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Scale the positive extension cap with the acoustic rate

On 3008848, cap each affected positive diagonal tangent
derivative at 0.2 times the accepted particle acoustic bound
divided by cell width. This replaces the previous 0.01/dt
cap with a rate that has no explicit timestep dependence.
Both default native and SM120 dry-scene tests pass.

| Native case | Outcome at 3 s |
|---|---|
| Coarse | 24.548186° / 0.221322 m |
| Coarse half-step | 32.577760° / 0.160288 m |
| Fine | 15.377568° / 0.276944 m |
| Fine half-step | Unlabeled FailedPrecondition at step 2429 |

Every case fails at least one check. The coarse half-step
misses the radius limit by 0.288 mm, and both nominal cases
spread farther than with the timestep-scaled cap. Removing
explicit timestep dependence from this control does not
remove timestep sensitivity. The temporary source was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-acoustic-extension-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Limit retained tangent rows by the local face velocity range

On 3008848, scale an affected tangent row uniformly so
the compatible PIC mean plus its centered affine correction
lies within the minimum and maximum velocities on the current
positive-weight compatible stencil. This preserves the weighted
mean for that stencil. The control acts at gather time, so the
bound applies to the sampled stencil before particle advection
and the subsequent wall projection. Both default dry-scene
tests pass on native and SM120.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 28.581131° | 0.173046 m |
| Coarse half-step | 35.582541° | 0.160602 m |
| Fine | 23.383563° | 0.185300 m |
| Fine half-step | 29.562396° | 0.169252 m |

All four complete, and all four exceed the radius limit.
The local range bound avoids the support rejections of full
affine retention but loses both fine passes. The temporary
source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-local-envelope-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Scale in-plane affine retention by resolved face support

On 3008848, sum compatible non-fallback face weights, scaling
each contribution by the smaller of one and the face mass
divided by 0.1 times the minimum particle mass. Use that sum
to scale the stored in-plane tangent derivatives on affected
rows. Clear their wall-normal derivative and clear affected
normal rows. Both default dry-scene tests pass.

| Native case | First rejected step | Status |
|---|---:|---|
| Coarse | 424 | Face motion bound |
| Coarse half-step | 1321 | Unlabeled FailedPrecondition |
| Fine | 405 | Face motion bound |
| Fine half-step | 2137 | Unlabeled FailedPrecondition |

All four reject before 3 s. Weighting by the resolved
fraction of the stencil alone leaves enough in-plane
feedback to produce early rejections. The source was
restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-support-weighted-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Match absolute timesteps across grids on affine clearing

On 3008848, exchange the coarse and fine absolute durations
at both nominal and half step. Keep the solver, release,
3 s endpoint, and acceptance checks. Both default dry-scene
tests pass on native and SM120.

| Absolute dt (s) | Coarse angle / radius | Fine angle / radius |
|---:|---:|---:|
| 0.00187500 | 34.329576° / 0.168211 m | 30.573436° / 0.157574 m |
| 0.00156250 | 35.439121° / 0.166939 m | 31.443051° / 0.159557 m |
| 0.00093750 | 40.501234° / 0.156712 m | 33.754916° / 0.159084 m |
| 0.00078125 | 42.008730° / 0.151892 m | 34.693822° / 0.156304 m |

All four fine-grid results pass the 3 s geometry checks.
All four coarse-grid results fail at least one check. At
the same absolute timestep the fine angle is 3.76–7.31°
shallower. The gap grows as the absolute timestep shrinks
across these cases. Equalizing timestep does not remove
the grid-width difference. The temporary fixture edit was
restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-clearing-matched-dt-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Translate the clearing candidate grid by half a cell

On 3008848, set the fixture grid origin to (h/2, 0, h/2).
Keep the release coordinates, floor, timesteps, and estimator.
Both default native and SM120 dry-scene tests pass.

| Native case at 3 s | Angle | Radius |
|---|---:|---:|
| Coarse | 32.519113° | 0.180497 m |
| Coarse half-step | 36.732337° | 0.161567 m |
| Fine | 29.155897° | 0.179221 m |
| Fine half-step | 34.880324° | 0.171549 m |

All four radii exceed 0.16 m. The largest radius increase
is 19.664 mm in the nominal fine case. The coarse half-step
angle moves 3.769° toward the window and still exceeds 35°.
Both fine passes disappear under translation of the grid
alone. The temporary fixture edit was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-clearing-half-phase-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Translate only the x grid coordinate

On 3008848, shift the grid origin to (h/2, 0, 0). Report
maximum absolute x and z coordinates separately, in addition
to the unchanged radial envelope. Both default dry-scene
tests pass on native and SM120.

| Native case at 3 s | Angle | Radius (m) | Maximum absolute x (m) | Maximum absolute z (m) |
|---|---:|---:|---:|---:|
| Coarse | 31.535825° | 0.178566 | 0.132723 | 0.141835 |
| Coarse half-step | 38.765678° | 0.159629 | 0.114436 | 0.121270 |
| Fine | 28.238483° | 0.171877 | 0.125646 | 0.143783 |
| Fine half-step | 32.534593° | 0.170746 | 0.109987 | 0.141036 |

All four cases fail. The x extent is smaller than the z
extent by 6.8–31.0 mm across the matrix. The result shows
directional spread under a one-axis translation in addition
to the radius sensitivity under a two-axis translation.
The temporary fixture edit was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-clearing-axis-phase-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Exchange the shifted grid axis

On 3008848, repeat the one-axis translation with grid origin
(0, 0, h/2). Both default native and SM120 dry-scene tests pass.

| Native case at 3 s | Angle | Radius (m) | Maximum absolute x (m) | Maximum absolute z (m) |
|---|---:|---:|---:|---:|
| Coarse | 31.130497° | 0.181106 | 0.142666 | 0.130954 |
| Coarse half-step | 38.753154° | 0.159600 | 0.121617 | 0.113951 |
| Fine | 28.548713° | 0.171107 | 0.144439 | 0.122547 |
| Fine half-step | 32.727603° | 0.169248 | 0.141194 | 0.108735 |

The larger extent follows the unshifted axis in every case.
After exchanging axes, the paired extents differ by at most
3.10 mm and the angles by at most 0.406°. All four cases
retain their failed checks. This comparison identifies grid
alignment as the main influence on the direction of spread
in these controls. The temporary fixture edit was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-clearing-z-phase-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Test the midpoint grid width at matched timesteps

On 3008848, set the diagnostic grid width to 0.06875 m and
retain the four original absolute timesteps. Both default
native and SM120 dry-scene tests pass.

| Absolute dt (s) | Native angle at 3 s | Radius (m) | Failed check |
|---:|---:|---:|---|
| 0.00187500 | 32.750738° | 0.161938 | Radius |
| 0.00156250 | 33.795786° | 0.160039 | Radius |
| 0.00093750 | 36.768764° | 0.159817 | Angle |
| 0.00078125 | 39.691141° | 0.156464 | Angle |

At each timestep, the midpoint angle falls between the
previous coarse and fine angles. The radii do not always
fall between their endpoint values. All four cases fail,
including a 0.039 mm radius miss at 0.00156250 s. The
temporary fixture edit was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-clearing-mid-width-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Reproduce grid phase and settling horizon from the test runner

The tracked compatible-floor diagnostic accepts the optional
arguments case, phase, and endpoint, in that order. Cases are
all, coarse, coarse-half, fine, and fine-half. Phases are
aligned, half-x, half-z, and half-xz. Endpoints are 1.5, 3,
and 6 seconds. The default remains the aligned 1.5 s matrix.

Both default dry-scene CTest targets pass. The runner rejects
invalid case, phase, and endpoint selectors with exit 2.
The native command below finishes with angle 19.558582° and
radius 0.146728 m, returning a geometry failure. This command
uses the tracked compatible-floor solver.

```powershell
$b = 'out/build/m5-main-d6e8c6e-release'
& "$b/fluidity_m5_dry_scene_tests.exe" --q-sand-compatible-floor-long-settle fine half-x 3
```

### Measure six-second grid alignment sensitivity on the tracked solver

Run the tracked compatible-floor diagnostic at 0.0625 m
and 0.0015625 s with the original release. Both native
cases complete and return geometry failure.

| Grid phase | Angle at 6 s | Radius (m) | Final 0.05 s angle drift |
|---|---:|---:|---:|
| Aligned | 18.236314° | 0.148503 | 0.069169° |
| Half-cell x | 11.621296° | 0.146728 | 0.149468° |

The shift leaves a 6.615° difference at six seconds. The
shifted angle has fallen 7.937° since the previous three-second
measurement while the radius matches at printed precision.
Both endpoints pass the short drift check. The late geometry
change therefore comes from declining height in the shifted
case. CUDA was not run for this six-second comparison.

```powershell
$b = 'out/build/m5-main-d6e8c6e-release'
foreach ($phase in 'aligned', 'half-x') {
  & "$b/fluidity_m5_dry_scene_tests.exe" --q-sand-compatible-floor-long-settle fine $phase 6
}
```

### Reset compatible-floor motion during the six-second trace

Temporarily enable the existing TestScene reset_column_velocity
argument in the tracked compatible-floor command. The diagnostic
zeros stored velocity and affine slopes at 0.75 s and retains
particle positions, elastic deformation, and plastic history.
It then advances to 6 s with the same gravity and wall settings.

| Native fine-grid phase | Kinetic energy removed (J) | Angle without reset | Angle after reset | Radius after reset (m) |
|---|---:|---:|---:|---:|
| Aligned | 0.00392945 | 18.236314° | 18.288757° | 0.148422 |
| Half-cell x | 0.00335059 | 11.621296° | 11.674413° | 0.146663 |

Both runs complete and fail the angle requirement. The reset
moves each final angle by less than 0.054° and leaves the
6.61° grid-alignment difference. Motion carried at 0.75 s
does not account for the late geometry in these cases. The
fixture was restored and the original native target rebuilt.

```powershell
$b = 'out/build/m5-main-d6e8c6e-release'
foreach ($phase in 'aligned', 'half-x') {
  & "$b/fluidity_m5_floor_reset_diagnostic.exe" --q-sand-compatible-floor-long-settle fine $phase 6
}
```

### Pair the tracked floor deformation gradient with stress scatter for six seconds

For the tracked compatible-floor solver, replace the completed
ghost gradient with the sum of compatible normalized weight
gradients times face velocity. Use that derivative in the
cut stress work budget and retain the existing affine fit
and swept wall contact. Repeat the nominal fine aligned
and half-cell x cases to 6 s on native.

| Phase | Angle with ghost gradient | Angle with paired gradient | Paired radius (m) | Paired cut stress work (J) |
|---|---:|---:|---:|---:|
| Aligned | 18.236314° | 10.952469° | 0.148108 | 1.39e-16 |
| Half-cell x | 11.621296° | 6.695033° | 0.146455 | −2.01e-17 |

The paired gradient closes the measured cut stress work
residual to roundoff. Both piles settle flatter and miss
the angle gate. This comparison isolates the work mismatch
from a sufficient explanation for late flattening. The
temporary solver edit was restored and the original native
target rebuilt. The six-second control ran on native only.

```powershell
$b = 'out/build/m5-main-d6e8c6e-release'
foreach ($phase in 'aligned', 'half-x') {
  & "$b/fluidity_m5_paired_6s_diagnostic.exe" --q-sand-compatible-floor-long-settle fine $phase 6
}
```

### Trace the transition to a floor layer

In the tracked compatible-floor nominal fine case, sample the
committed particle heights at the first step ending at or after
each integer second. Count a particle on the floor when its
height is at most 1e-9 m. There are 63 equal-mass particles.

| Time (s) | Aligned floor count | Aligned maximum (m) | Half-cell x floor count | Half-cell x maximum (m) |
|---:|---:|---:|---:|---:|
| 1 | 34 | 0.084971 | 30 | 0.078910 |
| 2 | 51 | 0.071561 | 49 | 0.062010 |
| 3 | 60 | 0.064925 | 58 | 0.052115 |
| 4 | 60 | 0.058321 | 61 | 0.044525 |
| 5 | 61 | 0.052881 | 61 | 0.037859 |
| 6 | 61 | 0.048930 | 62 | 0.030176 |

The aligned pile loses 6.6 mm of maximum height between the
3 and 4 s samples with the floor count fixed at 60. The shifted
pile loses 6.7 mm between 4 and 5 s with the count fixed at 61.
The final envelope is measured from a handful of elevated
particles while the bulk lies on the floor. Both native traces
reproduce the original six-second angles. The instrumentation
was restored and the original native target rebuilt.

### Compare upper pile retention on bounded tangent clearing

On 3008848, repeat the six-second height trace on native with
the same 1e-9 m floor tolerance. Both default native and SM120
dry-scene tests pass. The four extended geometry results match
the earlier clearing runs.

| Case | Floor count near 3 s | Floor count at 6 s | 90th-percentile height at 6 s (m) | Maximum height at 6 s (m) |
|---|---:|---:|---:|---:|
| Coarse | 38 | 58 | 0 | 0.092598 |
| Coarse half-step | 13 | 38 | 0.042772 | 0.111769 |
| Fine | 37 | 58 | 0 | 0.077173 |
| Fine half-step | 17 | 37 | 0.030998 | 0.093434 |

Both nominal cases have 58 of 63 particles on the floor at
6 s. Both half-step cases retain at least 25 elevated particles.
The nominal fine clearing case retains five elevated particles
at 6 s versus two on the tracked aligned compatible-floor case.
The time-step dependence therefore includes a change in how
much mass stays above the floor. The temporary fixture edit
was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-height-trace-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 6
}
```

### Extend the passing fine half-step to nine seconds

On 3008848, temporarily set the explicit endpoint of the
symmetric-affine-decay diagnostic to 9 s. Keep zero decay
and the original fine half-step duration and acceptance
checks. Both default native and SM120 dry-scene tests pass.
The extended native case returns a geometry failure.

| Time (s) | Angle | Radius (m) | Particles on floor |
|---:|---:|---:|---:|
| 6 | 30.869627° | 0.156304 | 37 of 63 |
| 9 | 27.988015° | 0.156304 | 51 of 63 |

The radius is stable to the reported precision while maximum
height falls from 0.093434 to 0.083067 m. The 9 s angle drift
over the last 0.05 s is 0.051583°, within the short-window
limit. The sole six-second geometry pass therefore does
not persist. The temporary fixture change was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-nine-second-bin'
& "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 fine-half all 9
```

### Report one-second drift in extended sand diagnostics

Extended sand cases now print the angle at the first
accepted step at or after one second before the endpoint,
the actual interval, and the absolute angle change. The
diagnostic does not split a step to take this sample.

Both default native and SM120 dry-scene tests pass. In the
native 1.5 s column matrix, the nominal coarse, nominal fine,
half-step coarse and half-step fine one-second drifts are
8.849252°, 6.428438°, 15.183327° and 11.922014°. Each passes
the original short-window drift check and fails the angle
check. The longer window is reported without changing
the existing acceptance thresholds.

```powershell
$b = 'out/build/m5-main-d6e8c6e-release'
& "$b/fluidity_m5_dry_scene_tests.exe" --q-sand-column-long-settle
```

### Reverse the order of the two wall traction sweeps

On 3008848, visit the same floor reaction points in reverse
order on the first sweep and forward order on the second.
Keep each sweep's half share of the Coulomb impulse cap.
Both default native and SM120 dry-scene tests pass.

| Native case at 3 s | Angle | Radius (m) |
|---|---:|---:|
| Coarse | 34.439206° | 0.168254 |
| Coarse half-step | 40.501578° | 0.156711 |
| Fine | 31.434087° | 0.159357 |
| Fine half-step | 34.645655° | 0.156346 |

Both coarse failures persist. The largest change is 0.110°
in angle and 0.200 mm in radius. Both fine cases still pass
at 3 s. The temporary source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-reverse-traction-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Divide the wall traction cap across eight sweeps

On 3008848, replace two alternating sweeps with eight,
each receiving one eighth of the original total Coulomb
impulse cap. Both default native and SM120 dry-scene tests pass.

| Native case at 3 s | Angle | Radius (m) |
|---|---:|---:|
| Coarse | 34.674466° | 0.166994 |
| Coarse half-step | 40.852974° | 0.155189 |
| Fine | 31.547053° | 0.158528 |
| Fine half-step | 34.788473° | 0.155500 |

All four radii decrease by 0.804–1.523 mm. The coarse nominal
radius remains above 0.16 m, and the coarse half-step angle
rises by 0.352°. Both fine cases still pass at 3 s. This
control changes the subdivision of the impulse cap, so it
does not test convergence of a coupled friction solve.
The temporary source was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-eight-traction-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Accumulate the Coulomb impulse on the clearing candidate

On 3008848, run eight alternating block-coordinate sweeps.
For each reaction point subtract its previous impulse
contribution from the local velocity, solve over the full
Coulomb disk and scatter the impulse difference. The impulse
stored at each point stays inside one disk across sweeps.
Both default native and SM120 dry-scene tests pass.

| Native case at 3 s | Angle | Radius (m) |
|---|---:|---:|
| Coarse | 34.755154° | 0.166721 |
| Coarse half-step | 40.852828° | 0.154997 |
| Fine | 31.387507° | 0.158873 |
| Fine half-step | 34.667450° | 0.155669 |

Compared with the eight equal-cap sweeps, each radius changes
by less than 0.35 mm. Both fine cases pass and both coarse
failures persist. The fixed sweep count supplies no convergence
certificate. The temporary source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-accumulated-traction-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Increase accumulated traction iterations from eight to 32

On 3008848, repeat the accumulated Coulomb control with 32
alternating sweeps. The reaction points and full per-point
Coulomb disk are the same as in the eight-sweep control.
Both default native and SM120 dry-scene tests pass.

| Native case at 3 s | Angle | Radius (m) |
|---|---:|---:|
| Coarse | 34.789543° | 0.166766 |
| Coarse half-step | 40.839192° | 0.155087 |
| Fine | 31.410407° | 0.158782 |
| Fine half-step | 34.652251° | 0.155721 |

Compared with eight accumulated sweeps, angles change by
less than 0.035° and radii by at most 0.091 mm. Both coarse
failures persist. These are geometry sensitivity measurements
rather than a residual bound on the friction solve. The
temporary source was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-accumulated-32-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Predict floor contact on the bounded tangent clearing candidate

Repeat the earlier predicted-contact mask on 3008848.
Activate the floor mask for current contacts and particles
whose height after one gravity-predicted step falls within
the contact tolerance. Keep the same paired stress transfer
and wall sweep. Both default native and SM120 dry-scene tests pass.

| Native case at 3 s | Angle | Radius (m) |
|---|---:|---:|
| Coarse | 34.312393° | 0.169355 |
| Coarse half-step | 40.558258° | 0.157815 |
| Fine | 31.300106° | 0.161715 |
| Fine half-step | 34.684115° | 0.156606 |

All four radii increase. The coarse half-step angle changes
by only 0.057°. The nominal fine loses its radius pass, and
both coarse failures persist. The temporary source was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-predict-contact-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Activate the floor mask within one quadratic support radius

On 3008848, activate the floor mask for particles below
1.5 cell widths plus contact tolerance. Keep the stress
transpose, traction and swept contact. Both default dry-scene
targets fail without an assertion label.

| Native case at 3 s | Angle | Radius (m) |
|---|---:|---:|
| Coarse | 38.409591° | 0.195349 |
| Coarse half-step | 43.557439° | 0.198574 |
| Fine | 33.778061° | 0.178510 |
| Fine half-step | 42.372202° | 0.181372 |

All four extended native runs finish and exceed the radius
limit. Particle sweep work falls in magnitude in every case.
Cut stress work residual remains below 8e-16 J. The mask
changes the transfer for particles above the contact plane
and does not resolve Q-SAND. The source was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-geometric-mask-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

The first failure under the geometric mask is
TestUnsupportedFloorHasNoReaction on both native and SM120.
The fixture starts a compressed particle 0.2 m above the
floor on a 1 m grid with zero incoming velocity. It requires
a nonzero stress deposit, zero particle contacts and less
than 1e-12 N·s of grid wall impulse. The broader mask
includes this airborne particle's stencil. The diagnostic
reported the first failing fixture without isolating a
single predicate. Both temporary changes were restored.

The airborne budget trace reports identical values on native
and SM120. The step returns OK, deposits 2.66725130815e-6 N·s
of proposed stress impulse in L1, and registers zero particle
wall contacts. It nevertheless records an upward grid wall
impulse of 6.66812827038e-7 N·s and gives the 1 kg particle an
upward velocity of 6.66812827038e-7 m/s. Both exceed the
isolated fixture tolerances. The geometric mask therefore
applies an early wall reaction through stress scatter.
The diagnostic source was restored after linking.

### Exclude zero-weight faces from the contact mask

On 3008848, mark a face in the floor contact patch only
when a particle already at the floor has positive quadratic
weight on that face. Both default dry-scene tests pass on
native and SM120. The four native extended logs match the
original clearing runs byte for byte. This rules out the
zero-weight entries as a source of geometry differences in
this matrix. The source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-positive-contact-mask-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Clear all affine feedback on the paired floor path

On 3008848, zero the stored affine fit after the wall sweep
for all compatible-floor particles. Keep the paired stress
transfer and the deformation derivative. Both default native
and SM120 dry-scene tests pass.

| Native case at 3 s | Angle | Radius (m) | Mean log plastic Jacobian |
|---|---:|---:|---:|
| Coarse | 53.887524° | 0.108952 | 0.002887 |
| Coarse half-step | 62.499203° | 0.092496 | 0.000676 |
| Fine | 45.499702° | 0.122369 | 0.016953 |
| Fine half-step | 58.423082° | 0.100748 | 0.001858 |

All four angles exceed 35°. Compared with component-wise
clearing, this control greatly reduces both the pile radius
and accumulated plastic volume. The effect is substantially
larger than the donor and traction controls. The temporary
source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-clear-all-compatible-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Clear only the normal affine row after the wall sweep

On 3008848, clear the normal row for every compatible-floor
particle after applying the wall sweep. Keep component-wise
tangent clearing and the deformation and stress operators.
Both default native and SM120 dry-scene tests pass.

| Native case at 3 s | Angle | Radius (m) | Mean log plastic Jacobian |
|---|---:|---:|---:|
| Coarse | 38.052574° | 0.155184 | 0.154000 |
| Coarse half-step | 52.708577° | 0.117199 | 0.106111 |
| Fine | 34.460965° | 0.160164 | 0.255365 |
| Fine half-step | 44.380313° | 0.138741 | 0.181887 |

The nominal coarse radius enters the allowed interval,
but its angle exceeds 35°. The other three cases also
miss at least one check. Normal feedback has a substantial
effect, especially at half-step. Source was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-clear-normal-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Clear the vertical derivative of the normal affine row

On 3008848, zero only the y derivative of stored normal
velocity after the wall sweep. Retain the two horizontal
derivatives and the existing tangent feedback. Both default
dry-scene tests pass on native and SM120.

| Native case at 3 s | Angle | Radius (m) |
|---|---:|---:|
| Coarse | 42.785261° | 0.148737 |
| Coarse half-step | 52.915989° | 0.124165 |
| Fine | 38.117050° | 0.146770 |
| Fine half-step | 47.031431° | 0.139084 |

All radii enter the allowed interval and all angles exceed
35°. Keeping the horizontal derivatives of normal velocity
makes the piles steeper than clearing the entire normal row
in the preceding control. The temporary source was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-clear-normal-diagonal-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Taper vertical affine feedback near the floor

On 3008848, multiply the stored normal velocity's y derivative
by the accepted height divided by cell width, clamped to [0, 1].
Apply this after the wall sweep. Both default dry-scene tests
pass on native and SM120.

| Native case at 3 s | Angle | Radius (m) |
|---|---:|---:|
| Coarse | 34.874953° | 0.165185 |
| Coarse half-step | 40.126370° | 0.154796 |
| Fine | 31.853856° | 0.154852 |
| Fine half-step | 35.007421° | 0.154754 |

All radii decrease. The coarse nominal radius still exceeds
0.16 m, and the coarse half-step still exceeds 35°. The fine
half-step misses the angle limit by 0.007421°. The source was
restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-taper-normal-diagonal-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Isolate the sign of vertical affine feedback

On 3008848, retain only negative or only positive values
of the stored normal velocity's vertical derivative after
the wall sweep. Keep all other derivatives and the paired
stress and deformation operators. Both default dry-scene
tests pass for both controls on native and SM120.

| Native case at 3 s | Compression-only angle / radius | Expansion-only angle / radius |
|---|---:|---:|
| Coarse | 35.869053° / 0.156974 m | 43.034526° / 0.149088 m |
| Coarse half-step | 41.208446° / 0.145862 m | 52.929172° / 0.124214 m |
| Fine | 31.542354° / 0.151578 m | 38.285957° / 0.146650 m |
| Fine half-step | 35.240760° / 0.151825 m | 47.064008° / 0.139058 m |

Expansion-only is within 0.25° of zero vertical feedback
across the matrix. Compression-only spreads the piles
farther and leaves only the nominal fine case passing.
The nominal coarse radius is in range but its angle is
0.869° high. Both temporary source controls were restored.

```powershell
foreach ($mode in 'compression', 'expansion') {
  $bin = "out/pr-m5-sand-affine/out/sand-affine-normal-$mode-bin"
  foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
    & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
  }
}
```

### Trace compressive vertical affine feedback by height

On 3008848, immediately before particle publication, accumulate
max(0, minus the vertical derivative of stored normal velocity).
Group by accepted height below one cell width or at least one
cell width. Means are over particle-step samples in each group.
The late window uses the last ceiling(0.05/dt) accepted steps.
Both default dry-scene tests pass. Removing the trace lines
recovers all four original native 3 s logs.

| Case | Full near mean (1/s) | Full upper mean (1/s) | Late near mean (1/s) | Late upper mean (1/s) | Near share of late sum |
|---|---:|---:|---:|---:|---:|
| Coarse | 0.3422 | 0.9937 | 0.0208 | 0.2442 | 55.7% |
| Coarse half-step | 0.2286 | 0.6557 | 0.0995 | 0.1235 | 77.4% |
| Fine | 0.2517 | 1.2035 | 0.0131 | 0.2211 | 46.7% |
| Fine half-step | 0.2791 | 0.7768 | 0.0184 | 0.1245 | 58.5% |

Upper particles have the larger mean compressive slope in
every case. The near-floor group still accounts for almost
half or more of the late sum because it contains more
particles. These slopes drive the next deposit. The trace
does not measure constitutive strain or stress. The
temporary instrumentation was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-normal-trace-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3 |
    Where-Object { $_ -like 'vertical feedback *' }
}
```

### Bound normal affine feedback by the local velocity range

On 3008848, scale the stored normal row after the wall sweep
so its centered affine correction plus accepted normal velocity
lies within the range of that velocity and the current
positive-weight compatible face samples. This applies to
the sampled stencil and does not certify the next deposit
after advection. Both default native and SM120 dry-scene
tests pass.

| Native case at 3 s | Angle | Radius (m) | Mean log plastic Jacobian |
|---|---:|---:|---:|
| Coarse | 36.212855° | 0.162125 | 0.106731 |
| Coarse half-step | 43.527380° | 0.140910 | 0.042938 |
| Fine | 31.918908° | 0.155877 | 0.095343 |
| Fine half-step | 38.844720° | 0.148874 | 0.038311 |

All four cases accumulate less plastic volume and spread
less than the original clearing runs. Only nominal fine
passes. The limiter makes both half-step piles steeper.
The temporary source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-normal-range-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Clip positive vertical affine feedback only after contact

On 3008848, discard the positive vertical derivative of
the stored normal velocity only for particles whose wall
sweep reports contact. The contact projection already
zeros the normal row for incoming downward motion. Both
default native and SM120 dry-scene tests pass. The four
extended native logs match the original clearing runs byte
for byte. The difference from applying the signed limiter
to every particle therefore comes from particles outside
this contact branch. The temporary source was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-contact-normal-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Clear vertical affine feedback on structurally cut normal stencils

On 3008848, zero the stored vertical derivative of normal
velocity after the wall sweep when its current normal stencil
contains a structurally excluded sample. Keep the derivative
for fully compatible stencils. Both default native and SM120
dry-scene tests pass.

| Native case at 3 s | Angle | Radius (m) |
|---|---:|---:|
| Coarse | 34.948569° | 0.165319 |
| Coarse half-step | 40.068228° | 0.155761 |
| Fine | 31.847958° | 0.156535 |
| Fine half-step | 34.980838° | 0.155103 |

All radii decrease and both fine cases pass at 3 s. The fine
half-step angle has only 0.019162° of margin. Both coarse
failures persist. The temporary solver edit was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-normal-supported-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Extend cut-stencil vertical clearing to six seconds

Repeat the cut-stencil control on 3008848 with the explicit
diagnostic endpoint changed to 6 s. Both default native and
SM120 dry-scene tests pass. The extended matrix ran on native.

| Case | Angle at 6 s | Radius (m) | Result |
|---|---:|---:|---|
| Coarse | 29.629389° | 0.165319 | Angle and radius |
| Coarse half-step | 34.306636° | 0.163234 | Radius |
| Fine | 26.083211° | 0.156535 | Angle |
| Fine half-step | 31.239504° | 0.155103 | Pass |

The nominal fine angle drops 5.765° after 3 s even though
its radius matches at printed precision. Only the fine
half-step passes, as in the six-second baseline without
cut-stencil clearing. The temporary edits were restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-normal-supported-six-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 6
}
```

### Requalification against upstream 6472b77

The milestone branch integrated upstream 6472b77 at b58e678.
The upstream changes include the graded sampled liquid runtime
and public band reseeding. They leave the dry step, M5 material
law and boundary transfer sources unchanged from d6e8c6e.

A clean Windows Release SM120 build passed all 11 M5 CTest
targets. The Blender 5.2.3 authored bake, frame-10 checkpoint
resume and playback passed on native and SM120. Both measured
a minimum snow volume ratio of 0.648350539, a minimum log
plastic Jacobian of −0.420240034 and a water gap of 0.631821917 m.
All tracked native sources passed clang-format 19.1.5.

Use the earlier clean-build and Blender commands with
out/build/m5-main-6472b77-release, out/authored-upstream-6472b77-cpu,
and out/authored-upstream-6472b77-cuda. The default dry-scene
targets do not include the extended sand matrix. Q-SAND remains open.

Verified upstream main at 6472b77 and the M6 qualification branch
at 996a34e. The latter is an ancestor of upstream main. Reran all
11 M5 CTest targets using the build above. All passed, and the
tracked native and Blender sources match b58e678.

```powershell
ctest --test-dir out/build/m5-main-6472b77-release -R '^M5' --no-tests=error --output-on-failure
```

### Clear horizontal derivatives of the normal affine row

On 3008848, zero the x and z derivatives of stored normal
velocity after the wall sweep. Keep its vertical derivative
and the existing tangent feedback. Both default native and
SM120 dry-scene tests pass.

| Native case at 3 s | Angle | Radius (m) | Failed check |
|---|---:|---:|---|
| Coarse | 31.759420° | 0.179910 | Radius |
| Coarse half-step | 39.827983° | 0.154348 | Angle |
| Fine | 28.378341° | 0.177818 | Angle and radius |
| Fine half-step | 35.494428° | 0.160308 | Angle and radius |

The nominal runs spread farther by 11.699 and 18.261 mm,
respectively. Both fine passes are lost. The temporary
source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-normal-no-cross-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Retain only skew wall-normal cross derivatives

On 3008848, replace the xy/yx and zy/yz pairs of stored
affine derivatives with their skew parts after wall contact
processing. This leaves diagonal derivatives and the paired
stress/deformation transfer in place. The operation can
reintroduce individual entries previously cleared by the
fallback or wall projection. Both default dry-scene tests
pass on native and SM120.

| Native case at 3 s | Angle | Radius (m) |
|---|---:|---:|
| Coarse | 28.767140° | 0.187589 |
| Coarse half-step | 36.478565° | 0.169735 |
| Fine | 25.848083° | 0.182295 |
| Fine half-step | 29.065959° | 0.175760 |

All four radii increase and exceed the radius limit. Both
fine passes are lost. The experimental source was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-normal-skew-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Apply wall-normal skew before the fallback mask

On 3008848, take the skew part of the xy/yx and zy/yz
derivative pairs immediately after the complete affine fit.
Then apply the existing row clearing and swept wall
projection. This ordering avoids adding new entries after
either projection. Both default native and SM120 dry-scene
tests pass.

| Native case at 3 s | Angle | Radius (m) |
|---|---:|---:|
| Coarse | 30.529058° | 0.183879 |
| Coarse half-step | 38.104872° | 0.164157 |
| Fine | 27.391408° | 0.178679 |
| Fine half-step | 31.001749° | 0.170541 |

Every radius exceeds 0.16 m. The radii are smaller than in
the post-projection skew control and larger than on the
clearing candidate. Both fine passes disappear. The temporary
source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-skew-before-mask-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Apply symmetric wall-normal feedback before the fallback mask

On 3008848, retain the symmetric part of the xy/yx and
zy/yz derivative pairs immediately after the affine fit.
Apply the original row clearing and swept wall projection
afterward. Both default native and SM120 dry-scene tests pass.

| Native case at 3 s | Angle | Radius (m) |
|---|---:|---:|
| Coarse | 38.512724° | 0.157267 |
| Coarse half-step | 45.050235° | 0.134542 |
| Fine | 34.152518° | 0.150483 |
| Fine half-step | 42.627867° | 0.134024 |

All radii decrease relative to clearing alone. Three angles
exceed 35°, leaving only the nominal fine pass. The
complementary skew control increased all four radii.
The temporary source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-symmetric-before-mask-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Check the local work identity for wall traction

On 3008848, compare the sum of midpoint face impulse work
for each wall reaction point with the analytic kinetic energy
change J dot u plus one half of J dot M J. Here J is the
point's tangent impulse, u is its pre-impulse gathered velocity,
and M is the diagonal mobility formed by summing squared
normalized weights divided by face mass. This checks the
actual traction gather and scatter, which use the same
eligible weights. Both default dry-scene tests pass.

| Native case through 3 s | Traction updates | Maximum absolute work residual (J) |
|---|---:|---:|
| Coarse | 59772 | 1.63e-19 |
| Coarse half-step | 95700 | 2.72e-20 |
| Fine | 78108 | 1.36e-19 |
| Fine half-step | 118478 | 2.72e-20 |

The largest positive measured work is below 1e-37 J.
The geometry reproduces the original clearing matrix.
This audit covers individual traction updates with frozen
normal reaction. It does not establish a coupled contact
solution or close Q-SAND. The instrumentation was restored.

### Prevent a downward normal affine deposit from reversing sign

On 3008848, limit the normal affine row of particles with
nonpositive vertical velocity immediately before deposit.
Use the centered offsets of the current eligible normal
stencil and scale the whole row by the largest fraction
that keeps its reconstructed samples nonpositive.
Both default dry-scene tests pass on native and SM120.

| Native case at 3 s | Angle | Radius (m) |
|---|---:|---:|
| Coarse | 34.836707° | 0.168481 |
| Coarse half-step | 42.871228° | 0.146503 |
| Fine | 31.353075° | 0.162246 |
| Fine half-step | 38.654455° | 0.151479 |

Both half-step piles become narrower and steeper. The
nominal radii exceed the bound. All four fail at least
one check. The temporary solver edit was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-normal-no-reversal-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Clear both tangent affine rows and retain normal feedback

On 3008848, zero the x and z rows of stored affine velocity
after the wall sweep. Keep the normal row and the paired
stress and deformation transfer. Both default dry-scene
tests pass on native and SM120.

| Native case at 3 s | Angle | Radius (m) | Mean log plastic Jacobian |
|---|---:|---:|---|
| Coarse | 48.332970° | 0.136706 | 0.291211 |
| Coarse half-step | 59.879126° | 0.102097 | 0.159363 |
| Fine | 41.319563° | 0.144111 | 0.562006 |
| Fine half-step | 54.767594° | 0.115443 | 0.213225 |

All four radii enter the allowed interval and all angles
exceed 35°. Nominal fine plastic volume accumulation
increases even though the pile spreads less. The temporary
source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-clear-tangent-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Retain only the vertical derivative of tangent affine velocity

On 3008848, clear the x and z derivatives in both tangent
rows after the wall sweep. Keep their vertical derivatives
and the full normal row. Both default native and SM120
dry-scene tests pass.

| Native case at 3 s | Angle | Radius (m) |
|---|---:|---:|
| Coarse | 45.251768° | 0.140956 |
| Coarse half-step | 57.918026° | 0.108565 |
| Fine | 39.675323° | 0.147197 |
| Fine half-step | 53.200363° | 0.119056 |

Keeping vertical shear reduces the angles by 1.57–3.08°
relative to clearing both tangent rows entirely. Every
radius is in range, and every angle still exceeds 35°.
The temporary source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-clear-tangent-plane-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Reduce in-plane tangent feedback by one quarter

On 3008848, multiply the four horizontal derivatives of
stored tangent velocity by 0.75 after the wall sweep.
Keep vertical shear and the normal row. Both default
native and SM120 dry-scene tests pass.

| Native case at 3 s | Angle | Radius (m) |
|---|---:|---:|
| Coarse | 36.891824° | 0.164596 |
| Coarse half-step | 45.879210° | 0.140118 |
| Fine | 32.539370° | 0.156057 |
| Fine half-step | 40.135741° | 0.151842 |

Each result falls between full retention and full in-plane
clearing in both angle and radius. Even this reduction
raises the half-step angles by more than 5°. Only the
nominal fine case passes. The source was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-damp-tangent-plane-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Scale in-plane tangent damping with the timestep

On 3008848, apply exp(−150 dt) to the four horizontal
derivatives of stored tangent velocity after the sweep.
The damping rate is 150 per second. This gives retention
near 0.75 at the coarse nominal step and weaker damping
per half-step. Both default dry-scene tests pass.

| Native case at 3 s | Angle | Radius (m) |
|---|---:|---:|
| Coarse | 36.713658° | 0.164619 |
| Coarse half-step | 43.098913° | 0.149296 |
| Fine | 32.554909° | 0.154768 |
| Fine half-step | 36.930262° | 0.155928 |

Both half-step angles fall relative to fixed retention of
0.75, but still exceed 35°. The nominal fine case is the
only pass. The temporary source was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-timed-tangent-plane-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Apply time-scaled damping only to tangent extension

On 3008848, apply exp(−150 dt) to the stored x/x and z/z
derivatives after the wall sweep. Keep both horizontal
cross derivatives. Both default dry-scene tests pass.

| Native case at 3 s | Angle | Radius (m) |
|---|---:|---:|
| Coarse | 36.741154° | 0.164712 |
| Coarse half-step | 43.101068° | 0.149279 |
| Fine | 32.539640° | 0.154698 |
| Fine half-step | 36.927869° | 0.155942 |

Relative to damping the entire in-plane block at the same
rate, the largest angle change is 0.028° and the largest
radius change is 0.093 mm. Damping the diagonal terms
alone reproduces nearly all of the geometry response.
Only nominal fine passes. The source was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-damp-tangent-diagonal-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Damp only positive horizontal divergence in stored affine velocity

On 3008848, damp the positive part of the horizontal trace
by exp(−150 dt) after the wall sweep. Subtract half the
removed trace from each horizontal diagonal entry. This
preserves the diagonal difference and both cross derivatives.
Both default native and SM120 dry-scene tests pass.

| Native case at 3 s | Angle | Radius (m) |
|---|---:|---:|
| Coarse | 36.765858° | 0.164527 |
| Coarse half-step | 43.124785° | 0.149189 |
| Fine | 32.413558° | 0.155175 |
| Fine half-step | 36.866956° | 0.156118 |

Relative to damping both horizontal diagonal entries,
angles differ by at most 0.127° and radii by 0.477 mm.
The response is largely reproduced by damping positive
trace alone. Only nominal fine passes. The temporary
source was restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-positive-plane-trace-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Apply horizontal expansion damping after initial collapse

On 3008848, apply the previous positive horizontal trace
damping only from 0.75 s onward. Use a temporary test-only
settings sentinel to select the control while retaining
PIC velocity storage. Both default dry-scene tests pass
on native and SM120.

| Native case at 3 s | Angle | Radius (m) |
|---|---:|---:|
| Coarse | 34.683850° | 0.168464 |
| Coarse half-step | 40.903148° | 0.155495 |
| Fine | 31.630945° | 0.159047 |
| Fine half-step | 35.062748° | 0.156131 |

Late-only damping changes angles by at most 0.402° from
the clearing baseline at 3 s. Damping from release
produces a larger change in all four cases. Both coarse
failures persist and fine half-step misses the upper
angle limit. The solver and fixture edits were restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-late-plane-trace-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Apply horizontal expansion damping only during initial collapse

On 3008848, use the same 150 per second positive horizontal
trace damping as in the late-only control, but enable it only
before 0.75 s. Both default dry-scene tests pass.

| Native case at 3 s | Angle | Radius (m) |
|---|---:|---:|
| Coarse | 35.705880° | 0.164674 |
| Coarse half-step | 41.427366° | 0.151870 |
| Fine | 32.342130° | 0.154659 |
| Fine half-step | 35.607926° | 0.158103 |

Early-only damping shifts the final angle more than late-only
damping in all four cases. It does not recover either coarse
case. Only nominal fine passes. The solver and fixture edits
were restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-early-plane-trace-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Remove the release gap on bounded tangent clearing

On 3008848, place the same 63-particle column on the
floor by setting the release gap to zero. Keep the
original solver, grid widths, timesteps and checks.
Both default native and SM120 dry-scene tests pass.

| Native case at 3 s | Angle | Radius (m) | Mean log plastic Jacobian |
|---|---:|---:|---|
| Coarse | 34.248555° | 0.160189 | 0.120513 |
| Coarse half-step | 40.108370° | 0.151751 | 0.044122 |
| Fine | 31.795111° | 0.150788 | 0.081606 |
| Fine half-step | 34.767398° | 0.152390 | 0.066415 |

Removing the drop reduces radius and plastic volume in
all four runs. The coarse nominal radius is 0.189 mm above
the limit, and the coarse half-step angle stays above 40°.
Both fine cases pass at 3 s. The original release was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-clearing-floor-release-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Run the floor-supported 45-degree cone on tangent clearing

On 3008848, select the existing 371-particle cone release
at zero gap and 0.025 m sampling. Use the original clearing
grid widths and timesteps. This release changes both shape
and total mass relative to the column. The cone diagnostic
tests angle, drift and kinetic energy. Both default native
and SM120 dry-scene tests pass.

| Native case at 3 s | Envelope | Quadrant slopes |
|---|---:|---:|
| Coarse | 37.335867° | 49.160345° to 52.933307° |
| Coarse half-step | 39.355429° | 47.552856° to 50.252795° |
| Fine | 36.087403° | 45.465249° to 49.817676° |
| Fine half-step | 38.517101° | 46.633052° to 47.084221° |

All four envelopes exceed 35°. Every radius is 0.160078 m
at printed precision, and mean log plastic Jacobian remains
below 0.00103. This release shows a different failure mode
from the spreading column. The temporary fixture edit was
restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-clearing-cone-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Measure late constitutive return activity in the cone

Enable the existing return counters for the same 371-particle
cone on 3008848 from 0.75 s through 3 s. The fractions weight
the number of affected particles by the accepted step duration.
Both default dry-scene tests pass and the four native cone
geometries reproduce the preceding run at printed precision.

| Case | Particle-time fraction with a projected state | Fraction with plastic volume change |
|---|---:|---:|
| Coarse | 76.36% | 0% |
| Coarse half-step | 69.62% | 0% |
| Fine | 66.83% | 0.320% |
| Fine half-step | 67.48% | 0.081% |

Constitutive return remains active despite the small change
in plastic volume. The two coarse cases record no late volume
changes at the existing reporting tolerance. These counters
measure frequency and do not quantify the size of each shear
correction. The temporary fixture change was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-clearing-cone-yield-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Measure elastic energy removed by granular return

On 3008848, repeat the same floor-supported 371-particle cone
on the native path. For each accepted step, decompose the trial
elastic map with the existing principal-frame routine and sum
`V0 * (W(trial) - W(returned))` over particles, where
`W = mu * sum(log(stretch)^2) + lambda/2 * sum(log(stretch))^2`.
Prorate a step's contribution by its overlap with each reported
time window. Final geometry reproduces the preceding cone run
at printed precision.

| Case | 0–0.75 s (J) | 0.75–3 s (J) | 2.95–3 s (J) |
|---|---:|---:|---:|
| Coarse | 0.0120293 | 0.0096710 | 0.0001987 |
| Coarse half-step | 0.0074824 | 0.0049145 | 0.0001186 |
| Fine | 0.0144204 | 0.0147929 | 0.0002676 |
| Fine half-step | 0.0085839 | 0.0070123 | 0.0002102 |

The half-step removes 51% and 47% of the nominal-step energy
after 0.75 s at the coarse and fine widths, respectively. The
measurement includes all return branches. It measures the change
across the local constitutive update only. All four runs exit
with the expected envelope-angle failure. The fixture and
instrumentation were restored after linking.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-cone-energy-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Locate the timestep-sensitive return energy by height

Repeat the native cone diagnostic and group each particle's energy
change by its height at the start of the accepted step. Report the
energy from 0.75 to 3 s in three bands. The first band covers one
initial particle spacing. The upper band starts one coarse grid
cell above the floor. Total energy and final geometry reproduce
the preceding runs.

| Case | Below 0.025 m (mJ) | 0.025 to 0.075 m (mJ) | At least 0.075 m (mJ) |
|---|---:|---:|---:|
| Coarse | 2.467 | 3.482 | 3.722 |
| Coarse half-step | 1.090 | 2.607 | 1.218 |
| Fine | 1.674 | 7.874 | 5.246 |
| Fine half-step | 0.974 | 4.030 | 2.007 |

The first band accounts for 29% of the energy reduction on timestep
halving at the coarse width and 9% at the fine width. The upper band
accounts for 53% and 42%, respectively. This locates where the return
removes energy. Boundary effects can propagate through the pile, so
these bands alone cannot locate the source of the sensitivity.
The temporary instrumentation and fixture changes were restored.
All four runs still fail the angle gate.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-cone-energy-height-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Unload the cone by removing gravity at 0.75 s

On 3008848, repeat the native elastic return energy diagnostic and
split the timestep exactly at 0.75 s. Set gravity to zero for the
remaining attempts. The cumulative elastic return loss before
unloading reproduces the loaded control at the reported precision.

| Case | Loaded loss from 0.75 to 2.25 s (mJ) | Unloaded loss over the same interval (mJ) | Last accepted unloaded time (s) |
|---|---:|---:|---:|
| Coarse | 6.514 | 20.661 | 2.645625 |
| Coarse half-step | 3.191 | 10.913 | 2.648438 |
| Fine | 10.609 | 17.848 | 2.251562 |
| Fine half-step | 4.325 | 9.415 | 2.292969 |

The unload excites a transient with more constitutive energy loss
than the loaded control. At both widths, the loss still drops by
roughly half with the timestep. Continued gravitational acceleration
is therefore unnecessary for this sensitivity in the unload transient.
All four runs terminate with FailedPrecondition after the comparison
interval. The release build reports an empty status message, confirmed
by printing the raw message in a separate fine-grid rerun. The temporary
edits were restored. The standard Q-SAND gate remains open.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-cone-energy-unloaded-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Resolve the fine unload rejection to missing destination support

Trace the fine nominal unload on 3008848. The first rejection comes
from the destination MAC stencil check at step 1441. The attempted
destination is approximately (0.001065, 0.656510, 0.000105) m on the
vertical face stencil. Expanding the allocation on every side exceeds
the bounded transfer capacity at initialization. Adding only one
vertical block layer permits all steps through 3 s. The cone then
reaches 0.902773 m maximum height, 79.761397° envelope angle and
0.009589 J kinetic energy. The widened diagnostic still exits on the
angle check. The unloaded top particle travels well beyond the loaded
cone height, so the late unload is a different physical regime from
the loaded Q-SAND settlement. Temporary edits were restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-cone-energy-unloaded-tall-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine all 3
```

### Check the loaded cone apex against its nearest neighbors

Repeat the loaded native cone cases on 3008848 and rank the final
particle positions by height. Measure the apex's Euclidean distance
to its nearest neighbor and its stored vertical speed. All four
cases reproduce the reported Q-SAND angle failures.

| Case | Apex height (mm) | Second height (mm) | Apex nearest-neighbor distance (mm) | Apex vertical speed (mm/s) |
|---|---:|---:|---:|---:|
| Coarse | 122.105 | 104.927 | 18.355 | −7.035 |
| Coarse half-step | 131.281 | 110.615 | 21.740 | −3.320 |
| Fine | 116.677 | 100.849 | 17.278 | −6.823 |
| Fine half-step | 127.410 | 108.148 | 20.693 | −5.028 |

The apex is descending in each case and its nearest neighbor is
closer than the initial 25 mm spacing. These distances give no
evidence of an isolated airborne apex in the loaded runs. The
maximum-height estimator remains sensitive to the apex. The
temporary fixture changes were restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-cone-top-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Gather deformation after a half stress kick

On 3008848, keep the ordinary post-contact field for PIC advection
and APIC storage. For deformation only, subtract half of the
applied stress acceleration times the step duration from the
post-contact donors and reconstruct the ghost rows from that
field. Gravity and wall traction enter the field with their full
increments. This is a split-update control, with the original
63-particle release and 0.10 m gap.

| Native case at 3 s | Envelope | Radius (m) | Mean log plastic Jacobian |
|---|---:|---:|---:|
| Coarse | 35.418809° | 0.197559 | 0.404900 |
| Coarse half-step | 39.977885° | 0.177459 | 0.190201 |
| Fine | 34.726713° | 0.175517 | 0.257545 |
| Fine half-step | 34.470502° | 0.169788 | 0.118874 |

All four radii exceed 0.16 m and the coarse half-step angle
still exceeds 35°. The experiment gathers deformation from a
different time level than the grid stress-work receipt, so its
cut work residual cannot qualify the paired operator. The
temporary solver change was restored. Q-SAND remains open.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-midpoint-stress-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Gather deformation before the wall traction correction

On 3008848, reconstruct the ghost velocities for the deformation
gradient from the field after stress and gravity, before the grid
applies tangential wall traction. Keep the fully corrected field
for PIC advection and APIC storage. Use the original 63-particle
column with a 0.10 m release gap.

| Native case at 3 s | Envelope | Radius (m) | Mean log plastic Jacobian |
|---|---:|---:|---:|
| Coarse | 35.544793° | 0.147847 | 0.221276 |
| Coarse half-step | 39.943048° | 0.139890 | 0.143500 |
| Fine | 31.512106° | 0.149593 | 0.212256 |
| Fine half-step | 35.415479° | 0.146974 | 0.118880 |

All four radii fall within the declared interval. The fine nominal
run passes; the other three exceed 35°. The pre-traction gradient
and post-traction stress work use different velocity fields, so
this diagnostic is insufficient to qualify their work identity.
The experimental change was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-pre-traction-gradient-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Vary applied grid traction while keeping the particle sweep

On 3008848, run the standard 63-particle column with either zero
grid tangential traction or half the solved face impulse. Keep
the particle Coulomb sweep and the matched stress transpose and
deformation gather. The full-strength particle sweep still acts
on both controls.

| Native case at 3 s | Zero grid traction angle / radius | Half grid traction angle / radius |
|---|---:|---:|
| Coarse | 28.433335° / 0.160939 m | 32.352707° / 0.164934 m |
| Coarse half-step | 33.326321° / 0.149923 m | 37.455386° / 0.151267 m |
| Fine | 24.949298° / 0.167438 m | 29.492357° / 0.160775 m |
| Fine half-step | 27.797890° / 0.161517 m | 32.554185° / 0.156271 m |

Increasing traction from zero to half raises every angle by
3.9–4.8°. The coarse half-step passes at zero and the fine
half-step passes at half strength. The paired cut stress work
residual stays within 3.6e-16 J across the eight runs. Both
controls leave the full gate open. The solver edits were restored.

```powershell
foreach ($variant in 'no-grid-traction', 'relaxed-grid-traction') {
  $bin = "out/pr-m5-sand-affine/out/sand-affine-$variant-bin"
  foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
    & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
  }
}
```

### Keep grid traction and remove particle tangential friction

On 3008848, keep the full grid traction solve and the particle
sweep's normal constraint. On the compatible floor path, set the
particle sweep and its affine derivative's friction coefficient
to zero. Use the original 63-particle column.

| Native case at 3 s | Envelope | Radius (m) |
|---|---:|---:|
| Coarse | 33.444340° | 0.172618 |
| Coarse half-step | 40.202194° | 0.155628 |
| Fine | 30.258110° | 0.166274 |
| Fine half-step | 34.478580° | 0.157770 |

The fine half-step passes. Both nominal runs exceed the radius
limit and the coarse half-step exceeds the angle limit. The
paired cut stress work residual stays within 2.6e-16 J. Removing
particle tangential friction leaves the timestep sensitivity
and the extended Q-SAND gate open. The solver edits were restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-grid-only-traction-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Transpose the wall reaction traversal

On 3008848, order the same floor reaction points by z then x
instead of x then z. Keep the forward and reverse sweeps and
their original impulse caps. Use the standard 63-particle column.

| Native case at 3 s | Envelope | Radius (m) |
|---|---:|---:|
| Coarse | 34.389107° | 0.168137 |
| Coarse half-step | 40.501222° | 0.156712 |
| Fine | 31.456976° | 0.159442 |
| Fine half-step | 34.696680° | 0.156300 |

The two fine cases pass and the two coarse cases fail, as in the
original traversal. The cut stress work residual stays within
2.9e-16 J. Traversal transposition leaves the coarse half-step
angle more than 5° above the limit. The experimental sort was
restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-transposed-traction-order-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Include wall-face stopping impulse in the grid friction cap

On 3008848, clamp incoming normal velocity to zero on grid faces
exactly on the floor. Add the resulting positive normal impulse
divided by the step duration to the local reconstructed reaction.
Keep the two tangential sweeps and the particle contact fallback.

| Native case at 3 s | Envelope | Radius (m) |
|---|---:|---:|
| Coarse | 34.162185° | 0.154494 |
| Coarse half-step | 37.624260° | 0.145693 |
| Fine | 27.781766° | 0.179817 |
| Fine half-step | 33.068293° | 0.175965 |

The coarse nominal case passes. The coarse half-step remains too
steep and both fine cases exceed the radius limit. The maximum
absolute paired cut stress work residual is 7.7e-17 J. Particle
sweep energy change falls below 0.000061 J in each run, indicating
that the grid clamp takes over almost all normal stopping work
in this fixture. The experiment was restored and Q-SAND stays open.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-wall-face-normal-cap-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Separate the wall-face normal clamp from its traction cap

Repeat the four native column cases on 3008848 with either the
wall-face normal clamp alone or the stopping impulse added to the
traction cap alone. Keep the particle sweep in both controls.

| Case | Clamp alone, first rejected step and stencil | Cap alone, envelope and radius at 3 s |
|---|---|---:|
| Coarse | 428, wall x traction | 35.052610° / 0.167978 m |
| Coarse half-step | 1465, wall x traction | 41.018846° / 0.160015 m |
| Fine | 501, wall z traction | 31.863409° / 0.161358 m |
| Fine half-step | 1382, destination | 35.290548° / 0.155641 m |

The clamp-only cases all reject a required MAC stencil. The
combined clamp and cap completed the four runs in the preceding
control. The cap-only cases complete and all miss the declared
angle or radius bound. Their paired cut stress work residual stays
within 6.5e-16 J. Both temporary solver changes were restored.

```powershell
foreach ($variant in 'wall-face-clamp-trace', 'wall-face-cap-only') {
  $bin = "out/pr-m5-sand-affine/out/sand-affine-$variant-bin"
  foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
    & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
  }
}
```

### Clamp normal velocity only for the compatible PIC gather

On 3008848, make a copy of the post-traction velocity field and
zero incoming normal velocity on occupied faces exactly on the
floor. Use the copy for compatible PIC advection and stored
translation. The unmodified post-traction field supplies the
deformation gradient and APIC fit. Account for the normal
stopping impulse in the wall momentum budget.

| Native case | Outcome |
|---|---|
| Coarse | 3.945917° envelope, 0.354413 m radius at 3 s |
| Coarse half-step | 6.265034° envelope, 0.362445 m radius at 3 s |
| Fine | FailedPrecondition at step 773 |
| Fine half-step | FailedPrecondition at step 2555 |

The two completed coarse runs spread to more than twice the
allowed radius. Their paired cut stress work residual stays
within 1.4e-16 J. Keeping the work identity alone therefore
does not prevent spreading when the two velocity fields diverge.
The temporary code was restored and Q-SAND stays open.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-normal-advection-clamp-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Measure low-mass stress omission and donor pruning separately

Run the four native column cases on unmodified 3008848 with
temporary tracing at donor removal. The existing impulse budget
measures force regularization after stress has been scattered.
The trace counts removal events across all steps and passes.

| Case | Proposed stress impulse L1 (N·s) | Omitted impulse at 1e-9 N·s precision | Donor removals |
|---|---:|---:|---:|
| Coarse | 81.872 | 0 | 570 |
| Coarse half-step | 97.020 | 0 | 0 |
| Fine | 75.667 | 0 | 485 |
| Fine half-step | 94.605 | 0 | 0 |

The measured omission after scattering is below reporting
precision. The donor-selection safeguard does change the
operator on both nominal runs, and neither half-step activates
that safeguard. In particular, the 40.50° coarse half-step
failure occurs without donor pruning. The trace was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-donor-pruning-receipt-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Partition late wall support between grid and particle contact

On 3008848, record the accepted grid wall impulse and swept particle
wall impulse for each step of the original native column matrix.
Sum the vertical components after the exact 0.75 s phase boundary.
For the final 0.05 s, prorate the step crossing 2.95 s.

| Case | Grid / particle, 0.75–3 s (N·s) | Grid / particle, 2.95–3 s (N·s) |
|---|---:|---:|
| Coarse | 33.306 / 18.942 | 0.483 / 0.675 |
| Coarse half-step | 42.999 / 9.161 | 0.875 / 0.285 |
| Fine | 29.533 / 22.744 | 0.490 / 0.670 |
| Fine half-step | 40.645 / 11.606 | 0.714 / 0.443 |

Halving the step shifts 9.7–11.1 N·s of support to the grid after
0.75 s while combined support remains close to 52.2 N·s in every
case. The effect persists in the final 0.05 s. The grid and particle
contact paths apply their friction impulses at different stages,
so this partition is a candidate for explaining timestep sensitivity.
Final angles and radii reproduce the preceding baseline. The trace
was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-support-phase-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Measure late tangential contact work by contact path

On 3008848, log the grid friction work and the horizontal
components of midpoint velocity dotted with the swept particle
wall impulse. Use the original native column matrix and sum the
accepted steps from 0.75 s to 3 s.

| Case | Grid tangential work (J) | Swept particle tangential work (J) |
|---|---:|---:|
| Coarse | −0.174694 | −0.000855 |
| Coarse half-step | −0.166236 | −0.000470 |
| Fine | −0.094254 | −0.001292 |
| Fine half-step | −0.123967 | −0.000613 |

The grid accounts for at least 98.6% of the measured tangential
work in every case. Despite the shift toward grid normal support
at both widths, halving the timestep reduces net tangential
dissipation on the coarse grid and increases it on the fine grid.
The final geometry reproduces the original matrix, including both
coarse failures. The temporary trace was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-wall-tangent-work-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Apply the full grid friction cap in one pass

On 3008848, replace the forward and reverse half-cap sweeps
with a single forward sweep using the full Coulomb cap. Keep
the same reaction points, particle sweep and stress work pair.

| Native case at 3 s | Envelope | Radius (m) |
|---|---:|---:|
| Coarse | 33.017268° | 0.171584 |
| Coarse half-step | 40.122559° | 0.159098 |
| Fine | 30.064790° | 0.160835 |
| Fine half-step | 34.431922° | 0.157467 |

Only the fine half-step passes. The coarse half-step stays above
40°, and the fine nominal radius crosses the 0.16 m limit. This
complements the previous accumulated-sweep tests by measuring the
full cap on the first visit alone. The paired cut stress work
residual stays within 3e-16 J. The temporary code was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-single-full-traction-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Extend the coarse half-step through 12 seconds

On 3008848, keep the coarse half-step solver and the original
63-particle column. Extend the endpoint to 12 s and sample
at exact multiples of 3 s by splitting steps at those times.

| Time (s) | Envelope | Maximum radius (m) |
|---|---:|---:|
| 3 | 40.500210° | 0.156714 |
| 6 | 34.203495° | 0.164440 |
| 9 | 31.747950° | 0.164440 |
| 12 | 29.115071° | 0.164440 |

The angle drops into the band at 6 and 9 s while the radius
exceeds the limit. By 12 s the angle has fallen below 30°.
Its drift over the final 0.05 s is only 0.038°, yet its
change from 9 to 12 s is 2.63°. The short drift window
therefore misses continued creep in this case. No sampled
time passes both bounds. The temporary fixture was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-coarse-long-horizon-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 coarse-half all 3
```

The final argument selects the temporarily extended diagnostic
mode. In the unmodified runner it selects 3 s.

### Check long-horizon persistence across the sand matrix

Extend the other three native column cases on 3008848 with the
same 12 s endpoint and exact 3 s sampling boundaries.

| Case | Angle at 3 s | 6 s | 9 s | 12 s | Radius at 3–12 s (m) |
|---|---:|---:|---:|---:|---:|
| Coarse | 34.328° | 28.831° | 25.319° | 21.948° | 0.168211 |
| Fine | 31.443° | 25.812° | 22.605° | 19.604° | 0.159557 |
| Fine half-step | 34.694° | 30.870° | 27.988° | 25.473° | 0.156304 |

Both fine cases pass the angle and radius checks at 3 s and
fall below 30° by 9 s. The outermost radius stays fixed to
the reported precision in these three runs while maximum
height falls throughout. With the preceding coarse half-step
result, all four angles are below 30° by 12 s. The result
points to late height loss as a common failure across the
matrix. The temporary fixture was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-coarse-long-horizon-bin'
foreach ($case in 'coarse', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Clear APIC feedback after the column first reaches 3 seconds

Repeat the native coarse and fine half-step runs on 3008848.
Starting at 3 s, clear the stored affine velocity slopes after
each accepted step. Keep stored translation and the material
state. The 3 s geometries match the loaded controls.

| Case | Angle at 6 s | 9 s | 12 s | Radius at 12 s (m) |
|---|---:|---:|---:|---:|
| Coarse half-step | 35.790785° | 33.772663° | 31.783084° | 0.164763 |
| Fine half-step | 32.181182° | 29.992081° | 27.553092° | 0.156304 |

Removing late affine feedback raises the final angle by 2.67°
on the coarse grid and 2.08° on the fine grid. Both piles
continue to lose height through 12 s. The coarse radius exceeds
0.16 m and the fine angle falls below 30°. The temporary
fixture change was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-late-apic-clear-bin'
foreach ($case in 'coarse-half', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Suppress late yielding as a diagnostic control

On 3008848, keep the original cohesionless column until 3 s.
Then move the granular yield surface away by setting translated
cone cohesion to 10 in log strain units. Keep the elastic
moduli, velocities, deformation and existing plastic history.
This switch probes the role of return in creep under load.

| Native case | Angle at 3 s | Angle at 12 s | Height at 12 s (m) | Radius at 12 s (m) |
|---|---:|---:|---:|---:|
| Coarse half-step | 40.500210° | 37.991655° | 0.135311 | 0.173242 |
| Fine half-step | 34.693822° | 33.516368° | 0.103520 | 0.156304 |

The coarse apex remains near its 3 s height. The fine apex
loses 4.7 mm compared with 33.7 mm in the cohesionless control.
Mean log plastic Jacobian stays at its 3 s value to printed
precision. The fine case passes the endpoint checks under the
changed material; the coarse case still exceeds both upper
bounds. The physical gate requires the original cohesionless
material. The temporary fixture change was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-late-elastic-bin'
foreach ($case in 'coarse-half', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Attribute late constitutive energy loss to the return branches

Repeat the native 12 s half-step column runs on 3008848.
For each accepted step, evaluate reference-volume-weighted
Hencky energy before and after granular return. Attribute the
loss to the tensile apex when the trial log stretch trace is
nonnegative. The remainder is dominated by the compressed
shear branch. Sample at the same 3 s boundaries.

| Case and interval | Total return loss (mJ) | Tensile apex loss (mJ) |
|---|---:|---:|
| Coarse half-step, 3–6 s | 314.198 | 2.232897 |
| Coarse half-step, 6–9 s | 17.102 | 0.000175 |
| Coarse half-step, 9–12 s | 6.762 | 0.000621 |
| Fine half-step, 3–6 s | 55.651 | 0.001472 |
| Fine half-step, 6–9 s | 14.636 | 0.000743 |
| Fine half-step, 9–12 s | 5.571 | 0.001191 |

The tensile apex accounts for at most 0.72% of return loss in
these late intervals. This identifies the dominant dissipation
branch while the pile creeps, leaving open which transfer or
contact effects repeatedly load that branch. Both trajectories
reproduce the previous results. Temporary edits were restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-late-return-branches-bin'
foreach ($case in 'coarse-half', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Reproduce long-horizon fine half-step creep on SM120

Run the same 12 s fixture on SM120 with exact 3 s sampling.
The envelope angles at 3, 6, 9 and 12 s are 34.693822°,
30.869627°, 27.988015° and 25.473317°. All four heights,
radii and angles match native at printed precision. The
final radius is 0.156304435 m. The hybrid CUDA transfer path
therefore reproduces the native loss of pile height and
fails the 30° lower bound. The temporary fixture was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-cuda-long-horizon-bin/fluidity_cuda_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 3
```

### Locate late return losses in the loaded fine column

Run the native fine half-step column on 3008848 to 12 s.
Sum the Hencky energy lost during return by particle height
at the beginning of each accepted step. The run reproduces
the baseline angles and radii at printed precision.

| Interval | Below 0.025 m (mJ) | 0.025 to 0.075 m (mJ) | At least 0.075 m (mJ) |
|---|---:|---:|---:|
| 3–6 s | 10.251 | 41.144 | 4.256 |
| 6–9 s | 6.697 | 4.918 | 3.021 |
| 9–12 s | 1.080 | 3.466 | 1.025 |

Particles above the lowest 25 mm account for 76% of the return
loss from 3 to 12 s. The upper band contains fewer particles as
the pile drops, so these totals measure each band's contribution
to total loss. They cannot identify a per-particle rate or the
source of repeated loading. The temporary trace was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-column-return-height-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 3
```

### Reset carried motion once at the late settlement boundary

On 3008848, run the native fine half-step column to 12 s.
At 3 s, clear stored translation and affine velocity slopes
once. Preserve particle positions and material history.
Account for the reset as an external impulse in the vertical
momentum check.

| Time (s) | Baseline envelope | Envelope after reset |
|---|---:|---:|
| 3 | 34.693822° | 34.693822° |
| 6 | 30.869627° | 30.873008° |
| 9 | 27.988015° | 27.990623° |
| 12 | 25.473317° | 25.474255° |

The final radius is unchanged at printed precision and the
final angle differs by less than 0.001°. The late height loss
persists when the state at 3 s starts from rest. The temporary
fixture change was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-late-motion-reset-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 3
```

### Freeze material history after the late motion reset

On 3008848, reset stored motion at 3 s and restore the accepted
elastic deformation and plastic Jacobian after each subsequent
step. The step still performs constitutive validation before
the fixture restores state. Gravity, transfer and wall contact
remain active. This freezes each particle's stress as it moves.

| Time (s) | Height (m) | Radius (m) | Envelope |
|---|---:|---:|---:|
| 3 | 0.108205 | 0.156304 | 34.693822° |
| 6 | 0.107948 | 0.156304 | 34.629900° |
| 9 | 0.111661 | 0.156304 | 35.541333° |
| 12 | 0.250637 | 0.374428 | 33.797831° |

The frozen control suppresses the initial late height loss,
then spreads beyond the radius limit and reaches 0.034064 J
of kinetic energy. Restoring stress after advection severs
the constitutive feedback and can inject energy, so the late
trajectory is diagnostic only. The temporary fixture was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-frozen-material-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 3
```

### Rebuild stress from the late pile geometry

On 3008848, reset stored motion and set the elastic deformation
to identity once at 3 s in the native fine half-step column.
Keep positions and log plastic Jacobians. Gravity and contact
remain active while stress rebuilds.

| Time (s) | Baseline envelope | Envelope after stress reset |
|---|---:|---:|
| 3 | 34.693822° | 34.693822° |
| 6 | 30.869627° | 31.049366° |
| 9 | 27.988015° | 28.050038° |
| 12 | 25.473317° | 25.853116° |

The final radius matches baseline at 0.156304 m. The stress
reset reduces the 12 s height loss by 1.27 mm, with 32.47 mm
of height loss remaining. The reset changes stored elastic
energy and the combined deformation Jacobian, so it is a
diagnostic intervention. The temporary fixture was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-late-stress-rebuild-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 3
```

### Clear elastic shear while retaining elastic volume

On 3008848, reset stored motion once at 3 s in the native fine
half-step column. Set the elastic map to the isotropic stretch
given by the cube root of its determinant. This preserves the
elastic Jacobian and hydrostatic Hencky stress while removing
deviatoric stress. Keep positions and plastic history.

| Time (s) | Baseline envelope | Envelope after shear reset |
|---|---:|---:|
| 3 | 34.693822° | 34.693822° |
| 6 | 30.869627° | 31.251379° |
| 9 | 27.988015° | 28.027629° |
| 12 | 25.473317° | 25.360606° |

The radius stays at 0.156304 m and the final angle differs
by 0.11°. The late height loss persists when the elastic
shear stress is rebuilt from the same geometry and elastic
volume. The temporary fixture was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-late-isochoric-reset-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 3
```

### Extend the fine quarter-step through 12 seconds

On 3008848, repeat the loaded fine column with a 0.0625 m grid
width and a 0.000390625 s timestep, half the preceding fine
half-step duration. Keep the solver and sample at exact
multiples of 3 s.

| Time (s) | Envelope | Radius (m) |
|---|---:|---:|
| 3 | 43.450751° | 0.143295 |
| 6 | 34.615322° | 0.156462 |
| 9 | 32.515366° | 0.156462 |
| 12 | 30.772573° | 0.156462 |

This case passes the fixture's checks at 12 s. Its angle
still drops by 1.74° between 9 and 12 s, with a final
0.05 s drift of only 0.029°. Timestep reduction delays
the height loss. The full matrix and long-term settlement
remain open. The temporary fixture was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-fine-quarter-horizon-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 3
```

The temporary runner maps `fine-half` to the quarter-step and
the final argument to 12 s.

### Check whether the fine quarter-step pass persists

Extend the native fine quarter-step diagnostic on 3008848 to
18 s. Keep the exact 3 s sampling boundaries.

| Time (s) | Envelope | Radius (m) |
|---|---:|---:|
| 12 | 30.772442° | 0.156462 |
| 15 | 29.223607° | 0.156462 |
| 18 | 27.813410° | 0.156462 |

The 12 s pass is transient. The angle drops below 30° by 15 s
while the radius stays constant to printed precision. The final
0.05 s drift is 0.027°, compared with 1.41° from 15 to 18 s.
The temporary fixture was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-fine-quarter-18-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 3
```

### Rebuild the extended sand diagnostic from source

Commit `255739e` on `codex/m5-sand-affine` exposes the fine
quarter-step and 12 s and 18 s horizons explicitly. Build the
native or CUDA dry-scene test executable from that commit and
run the command below. The native fine quarter-step reproduces
the 18 s angle of 27.813410°. The default native and SM120
dry-scene CTest targets both pass. The diagnostic leaves
the extended Q-SAND gate open.

```powershell
fluidity_m5_dry_scene_tests.exe --q-sand-symmetric-affine-decay 0 fine-quarter all 18
```

### Compare the APIC affine fit with the deformation derivative

The granular return on 3008848 uses the specified radial
deviator correction at fixed trace under compression. As a
transfer control, use the complete ghost-field APIC affine fit
for the trial deformation wherever that fit is enabled. Keep
the original stress scatter and its paired derivative for the
work receipt. Use the standard native column matrix.

| Case | Envelope at 3 s | Radius (m) |
|---|---:|---:|
| Coarse | 36.026817° | 0.167161 |
| Coarse half-step | 40.982734° | 0.157041 |
| Fine | 32.727131° | 0.158585 |
| Fine half-step | 35.964369° | 0.155640 |

Only the fine nominal case passes. The reported stress work
residual still measures the original derivative and its
transpose. Qualifying the affine fit as the deformation
operator would require its own paired stress scatter. The
temporary change was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-fit-deformation-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Pair the APIC deformation fit with its stress transpose

On 255739e, use the complete APIC fit for compatible-floor
deformation and scatter stress with its transpose. For a face
at displacement r with weight w, the slope coefficient is
4wr/h². Apply the same ghost transpose and leave particle
transport unchanged. Evaluate cut stress work using the fit.

| Native case | Envelope at 3 s | Radius (m) |
|---|---:|---:|
| Coarse | 35.297495° | 0.168524 |
| Coarse half-step | 40.197258° | 0.159639 |
| Fine | 32.832225° | 0.158354 |
| Fine half-step | 35.324804° | 0.155874 |

The paired work residual stays within 3.3e-16 J. Only the
fine nominal case passes the endpoint. The coarse half-step
remains more than 5° above the limit. The temporary solver
changes were restored and Q-SAND remains open.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-paired-affine-deformation-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Probe the discrete force balance from rest

On 255739e, copy the native fine half-step state at 3, 6 and
9 s. Clear translation and APIC slopes on each copy and
advance it one ordinary step with gravity and contact. Divide
the resulting stored velocity by step duration to measure the
effective acceleration. The copies leave the main trajectory
intact, and its 12 s angle reproduces the baseline.

| Time (s) | RMS (m/s²) | Maximum (m/s²) | Mean vertical (m/s²) |
|---|---:|---:|---:|
| 3 | 3.786 | 6.814 | −1.824 |
| 6 | 2.134 | 7.335 | −1.063 |
| 9 | 2.491 | 7.513 | −0.881 |

The low observed kinetic energy coexists with a substantial
discrete force imbalance in the from-rest probes. These values
include contact and transfer and measure their combined result.
The temporary fixture changes were restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-rest-force-probe-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 12
```

### Measure the force balance with gravity on and off for one step

On 255739e, copy the native fine half-step state at 3, 6 and
9 s. Clear motion and advance each copy once with standard
gravity or zero gravity. The main trajectory reproduces the
12 s baseline.

| Time (s) | RMS acceleration, loaded / unloaded (m/s²) | Mean vertical, loaded / unloaded (m/s²) | Grid support, both (mN·s) | Particle support, loaded / unloaded (mN·s) |
|---|---:|---:|---:|---:|
| 3 | 3.786 / 7.137 | −1.824 / 5.991 | 11.058 | 3.681 / 0 |
| 6 | 2.134 / 5.059 | −1.063 / 4.081 | 7.532 | 8.613 / 0 |
| 9 | 2.491 / 3.158 | −0.881 / 2.429 | 4.483 | 11.997 / 0 |

The reconstructed grid support at a fixed state is unchanged
in each pair. In the loaded probe, swept support increases
as reconstructed support falls over time. The unloaded probes
show upward acceleration and no swept wall impulse. The two
responses follow different contact branches and cannot be
combined as a linear force decomposition. The trace was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-rest-gravity-pair-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 12
```

### Match zero normal velocity on the active floor patch

On 255739e, set occupied normal grid faces to zero at y = 0
inside the existing contact patch. Add only the inward stopping
impulse to the grid friction cap. The deformation ghost field
already imposes zero there. Keep the particle sweep for any
remaining contacts.

| Native case | Envelope at 3 s | Radius (m) |
|---|---:|---:|
| Coarse | 36.272849° | 0.167545 |
| Coarse half-step | 41.662864° | 0.158736 |
| Fine | 31.721785° | 0.157737 |
| Fine half-step | 37.730761° | 0.156337 |

All four cases complete and the fine nominal case passes. The
absolute cut stress work residual stays below 3.9e-16 J. The
bilateral zeroing also stops outward motion on the active patch,
so it serves as a diagnostic rather than a unilateral contact
candidate. The temporary edit was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-active-wall-zero-normal-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Average particle translation for the wall sweep path

On 255739e, use the average of the old stored particle velocity
and the new PIC velocity to advect particles on the compatible
floor path. Keep the new PIC velocity as the incoming stored
velocity for contact and leave the deformation gather unchanged.

| Native case | Envelope at 3 s | Radius (m) |
|---|---:|---:|
| Coarse | 34.462340° | 0.167904 |
| Coarse half-step | 40.532172° | 0.156108 |
| Fine | 31.532431° | 0.159377 |
| Fine half-step | 34.680534° | 0.155862 |

The largest angle change from the original path is 0.133°.
Both fine cases pass the 3 s endpoint and both coarse failures
persist. The cut stress work residual stays within 3.5e-16 J.
The temporary edit was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-midpoint-advection-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Blend the quadratic derivative and APIC fit with matched scatter

On 255739e, average the quadratic shape derivative and complete
APIC fit in the compatible-floor deformation gather. Scatter
stress with the same equal-weight combination of their transposes.
Use the standard native column matrix.

| Case | Envelope at 3 s | Radius (m) |
|---|---:|---:|
| Coarse | 34.749673° | 0.168411 |
| Coarse half-step | 40.203489° | 0.158945 |
| Fine | 32.091002° | 0.159766 |
| Fine half-step | 34.940182° | 0.156519 |

The paired work residual stays within 5.2e-16 J. Both fine
cases pass the endpoint. The coarse radius and coarse
half-step angle still fail. The temporary code was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-paired-blended-deformation-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Check long-horizon creep with the paired blended gather

Run the native fine half-step blended control through 12 s.
The envelope is 34.940182° at 3 s, 31.483194° at 6 s,
28.789183° at 9 s and 26.147053° at 12 s. The radius stays
at 0.156519 m to printed precision. The final angle is
0.67° higher than baseline and still falls below the gate.
Paired cut stress work residual is −1.29e-16 J. The temporary
solver change was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-paired-blended-long-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 12
```

### Add the quadratic term to the frozen-gradient deformation increment

On 255739e, use (I + dt L + 0.5 dt² L²) F for compatible-floor
trial deformation, with L held at the gathered velocity
gradient. Keep transfer, contact and constitutive return intact.

| Native case | Envelope at 3 s | Radius (m) |
|---|---:|---:|
| Coarse | 34.405267° | 0.167779 |
| Coarse half-step | 40.522743° | 0.156163 |
| Fine | 31.474673° | 0.159112 |
| Fine half-step | 34.674671° | 0.156574 |

The largest angle change relative to the first-order control
is 0.076°. Both coarse failures persist. This checks the
frozen-gradient time integration term without correcting the
velocity field's time dependence. Temporary code was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-quadratic-deformation-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Check long-horizon creep with denser particle sampling

On 255739e, represent the fine half-step column with 504
particles at 0.025 m spacing. Keep the release dimensions,
total mass, grid and timestep. Run the native path to 12 s.

| Time (s) | Envelope | Radius (m) |
|---|---:|---:|
| 3 | 30.083739° | 0.204909 |
| 6 | 25.846978° | 0.212197 |
| 9 | 24.721483° | 0.212197 |
| 12 | 24.129897° | 0.212197 |

The denser column loses 23.65 mm of height after 3 s, compared
with 33.74 mm for 63 particles. It still creeps below 30°
and spreads beyond the radius limit. The temporary sampling
change was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-dense-long-horizon-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 12
```

### Include the affine normal velocity in floor patch prediction

On 255739e, activate the patch for existing contacts and when
a one-step prediction crosses the floor. Use the stored vertical
translation minus h times the norm of the vertical APIC slope
row, then add the gravity kick. This samples an incoming speed
bound over a ball of radius h around the particle. Keep the
original stress transpose, gather, and particle sweep.

| Native case | Envelope at 3 s | Radius (m) |
|---|---:|---:|
| Coarse | 34.179340° | 0.170465 |
| Coarse half-step | 40.574660° | 0.158481 |
| Fine | 31.187908° | 0.163386 |
| Fine half-step | 34.611556° | 0.158611 |

The fine half-step passes. Both nominal radii exceed 0.16 m
and the coarse half-step still exceeds 35°. All four radii
also increase relative to the translation-only prediction.
The cut work residual stays within 2.6e-16 J. The temporary
solver edit was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-affine-predicted-contact-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Exclude zero-weight faces from the predicted contact patch

On 255739e, combine the gravity and translation contact predictor
with a positive-weight requirement for each face in its patch.
All four native cases reproduce the preceding translation-only
prediction to printed precision. The coarse half-step remains
at 40.558258° and only the fine half-step passes. Zero-weight
stencil entries do not explain the predicted-contact outcomes
in this matrix. The temporary code was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-weighted-prediction-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Restrict the current floor patch to closing particles

On 255739e, keep the existing floor proximity test and exclude
a particle if its stored vertical velocity plus the gravity
kick points away from the floor. All four native outcomes match
the baseline at printed precision, including the cut work
residuals. The coarse half-step remains at 40.501234°. This
velocity criterion does not change the measured trajectories.
The temporary code was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-unilateral-patch-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Extend the coarse quarter-step through 18 seconds

On 255739e, halve the coarse half-step duration to 0.00046875 s
and sample the native trajectory every 3 s.

| Time (s) | Envelope | Radius (m) |
|---|---:|---:|
| 3 | 48.077570° | 0.131962 |
| 6 | 40.028898° | 0.154003 |
| 9 | 37.100592° | 0.161386 |
| 12 | 33.722170° | 0.161805 |
| 15 | 32.342969° | 0.161805 |
| 18 | 31.200871° | 0.161805 |

The coarse quarter-step reaches the angle band at the sampled
times from 12 s onward, with radius above 0.16 m throughout
that interval. Its angle falls 1.14° from 15 to 18 s. The
temporary fixture was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-coarse-quarter-18-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 coarse-quarter all 18
```

### Split compatible-floor deformation at swept impact

On 255739e, enable the existing pre-impact and post-impact
deformation composition for compatible-floor sand. Use the
gathered gradient before impact and its Coulomb projection
after impact. Keep the existing stress scatter and grid contact.

| Native case | Envelope at 3 s | Radius (m) |
|---|---:|---:|
| Coarse | 37.933422° | 0.168036 |
| Coarse half-step | 40.748841° | 0.154928 |
| Fine | 31.651993° | 0.161478 |
| Fine half-step | 35.020386° | 0.155106 |

All four cases fail at 3 s. The cut stress work receipt still
uses the pre-projection gather and cannot certify the split
deformation increment. The temporary edit was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-compatible-impact-split-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Restrict the impact split to steps starting above the floor

On 255739e, split compatible-floor deformation at the swept
impact only when the particle starts above contact tolerance.
Particles starting on the floor keep the ordinary deformation
gather. Compare with the preceding all-impact control.

| Native case | Envelope at 3 s | Radius (m) |
|---|---:|---|
| Coarse | 34.798496° | 0.168340 |
| Coarse half-step | 40.681377° | 0.157332 |
| Fine | 31.638407° | 0.159137 |
| Fine half-step | 34.962261° | 0.156023 |

Both fine cases pass again. The coarse nominal radius and
coarse half-step angle still fail. The stress work receipt
still uses the unsplit gather. The temporary edit was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-first-impact-split-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Disable grid traction after the initial pile forms

On 255739e, run the native fine half-step column with the
ordinary solver through 3 s. Disable only grid tangential
traction for the remaining accepted steps. Keep swept particle
contact and the existing deformation and stress operators.

| Time (s) | Baseline envelope | Grid traction off after 3 s |
|---|---:|---:|
| 3 | 34.693822° | 34.693822° |
| 6 | 30.869627° | 28.583616° |
| 9 | 27.988015° | 24.777995° |
| 12 | 25.473317° | 21.946015° |

The late control loses an additional 11.48 mm of height at
12 s with the same printed radius of 0.156304 m. The existing
grid traction resists late height loss in this case. The cut
stress work residual is −1.24e-16 J. Temporary edits were restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-late-grid-traction-off-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 12
```

### Measure Coulomb cap saturation during late creep

On 255739e, record the norm of each solved grid traction
impulse and its cap, for visits with positive cap and
nonzero tangent mobilities. Count a visit as saturated
within 1e-8 relative of the cap. Use the native fine
half-step trajectory.

| Interval | Sum of impulse norms / sum of caps | Saturated visits |
|---|---:|---:|
| 0–3 s | 89.9% | 85.5% |
| 3–6 s | 86.7% | 90.3% |
| 6–9 s | 89.5% | 90.2% |
| 9–12 s | 95.9% | 93.3% |

The late visits use nearly all the available scalar impulse
budget. This is consistent with the limited effect of more
sweeps. The sums are over visit magnitudes, so they do not
measure the net impulse vector or certify solver convergence.
The geometry reproduces baseline and the trace was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-traction-saturation-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 12
```

### Carry unused Coulomb cap into the reverse traction sweep

On 255739e, give each point the unused scalar allowance from its
first half-cap solve in addition to its usual second half-cap.
The sum of both impulse magnitudes remains bounded by the original
full cap. Keep the original wall model and native column matrix.

| Case | Envelope at 3 s | Radius (m) |
|---|---:|---:|
| Coarse | 34.329576° | 0.168211 |
| Coarse half-step | 40.501234° | 0.156712 |
| Fine | 31.428980° | 0.159587 |
| Fine half-step | 34.677697° | 0.156346 |

Both coarse geometries are unchanged at printed precision.
Fine angles change by at most 0.017°. The existing split of
the cap has little effect on these outcomes. The temporary
code was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-carry-traction-cap-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Compare a frozen-gradient midpoint deformation update

On 255739e, use four fixed-point iterations of the frozen-gradient
implicit midpoint equation for compatible-floor deformation.
Seed the iterations with the Euler trial. An exact midpoint
solve gives a Cayley rotation for a pure skew gradient. This
finite-iteration control does not certify exact orthogonality.

| Native case | Envelope at 3 s | Radius (m) |
|---|---:|---:|
| Coarse | 34.443076° | 0.167572 |
| Coarse half-step | 40.522921° | 0.156163 |
| Fine | 31.453697° | 0.159176 |
| Fine half-step | 34.674912° | 0.156575 |

The midpoint results are within 0.038° of the quadratic
frozen-gradient control. Both coarse failures persist.
The temporary solver edit was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-midpoint-frozen-rotation-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Test the energy blend with compatible grid history

On 255739e, use the compatible PIC weights for both pre-force
and post-contact grid velocity. Reuse the existing momentum-
corrected retained translation blend at energy 0.6. Keep the
existing support eligibility filter, particle advection and
paired stress transfer.

| Native case | Envelope at 3 s | Radius (m) |
|---|---:|---:|
| Coarse | 33.807934° | 0.166554 |
| Coarse half-step | 38.295968° | 0.159574 |
| Fine | 31.108506° | 0.156379 |
| Fine half-step | 33.609119° | 0.157328 |

Both fine cases pass. The coarse half-step angle is 2.21° lower
than baseline and remains above 35°. The coarse nominal radius
remains above 0.16 m. The absolute cut work residual is at most
1.3e-16 J. This tests retained translation in the existing
contact path. The temporary solver edits were restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-compatible-energy-blend-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Check long-horizon creep with compatible retained translation

Repeat the native fine half-step with the 0.6 energy blend
through 12 s. The radius stays at 0.157328 m from 3 s onward.

| Time (s) | Envelope |
|---|---:|
| 3 | 33.609119° |
| 6 | 30.140807° |
| 9 | 27.317809° |
| 12 | 25.046935° |

The pile loses 31.04 mm of height from 3 to 12 s, compared
with 33.74 mm at zero blend. It still falls below the angle
bound by 9 s. The paired cut work residual is 7.43e-17 J.
The temporary solver edits were restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-compatible-energy-long-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 12
```

### Compare carried material volume with late pile height

On 255739e, sum V0 det(F_E) exp(log J_pl) over the native fine
half-step particles and divide by the initial reference volume.
Also report the elastic-only volume ratio and the center height
of the equal-mass particles.

| Time (s) | Total material volume ratio | Elastic-only ratio | Center height (mm) |
|---|---:|---:|---:|
| 3 | 1.07537 | 0.92013 | 24.914 |
| 6 | 1.11490 | 0.94771 | 14.475 |
| 9 | 1.14359 | 0.96599 | 5.562 |
| 12 | 1.16538 | 0.97333 | 4.726 |

The carried material volume grows 8.37% after 3 s while the
center falls 20.19 mm and the printed outer radius stays
fixed. The Jacobian sum measures particle-carried volume.
Particle centers supply the independent geometric measurement.
The temporary trace was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-material-volume-horizon-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 12
```

### Measure particle-carried volume on the floor and above it

On 255739e, use the fixture's floor tolerance (1e-12 grid widths)
to group the native fine half-step particles. Sum total material
Jacobians in each group and report the mean within the group.

| Time (s) | Floor count / mean total Jacobian | Above-floor count / mean total Jacobian |
|---|---:|---:|
| 3 | 17 / 1.256 | 46 / 1.009 |
| 6 | 37 / 1.164 | 26 / 1.045 |
| 9 | 51 / 1.160 | 12 / 1.072 |
| 12 | 58 / 1.172 | 5 / 1.088 |

By 12 s, 58 of 63 particle centers lie on the floor and carry
92.6% of the reconstructed total volume. Group membership
changes over time, so this comparison cannot assign the volume
growth to particles that were already in contact. The temporary
trace was restored. Q-SAND stays open.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-material-volume-bands-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 12
```

### Attribute late volume change by contact transition

On 255739e, sum per-particle changes in total material Jacobian
over accepted steps from 3 to 12 s in the native fine half-step.
Classify each change by whether the particle is on the floor at
the start and end of the step.

| Contact transition | Sum of Jacobian changes |
|---|---:|
| Above to above | 1.741839 |
| Above to floor | 0.001029 |
| Floor to above | 0 |
| Floor to floor | 3.928191 |

Sustained floor contact accounts for 69.3% of the net gain.
The sum of 5.671059 reconstructs the increase in carried volume
reported earlier. The measurement isolates sustained floor
contact as the largest contributor and leaves the source of
its positive deformation trace open. The fixture was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-volume-transitions-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 12
```

### Resolve the sustained-contact deformation trace by axis

On 255739e, integrate dt times each diagonal component of the
gathered velocity gradient over accepted steps from 3 to 12 s.
Include particles whose centers begin and end the step on the
floor. On the native fine half-step the sums are 0.704347 in x,
1.934036 in y, and 0.637268 in z. The normal component accounts
for 59.0% of the summed first-order trace. The particle center
normal displacement is zero on these steps. The gradient trace
measures local deformation and differs from the finite-step
log determinant. The run reproduces the baseline and the
temporary trace was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-floor-diagonal-trace-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 12
```

### Remove the wall-normal diagonal in the paired operators

On 255739e, zero the deformation gradient's yy component for
particles that start on the floor. Omit the corresponding
stress term from the force scatter for those same particles.
The modification also changes the reconstructed normal
reaction, since it changes the stress scattered to the wall.

| Native case | Envelope at 3 s | Radius (m) | Mean log plastic Jacobian |
|---|---:|---:|---:|
| Coarse | 31.191050° | 0.167437 | 0.483013 |
| Coarse half-step | 39.504671° | 0.155209 | 0.221387 |
| Fine | 29.535828° | 0.162030 | 0.388308 |
| Fine half-step | 34.844536° | 0.155381 | 0.254342 |

Only the fine half-step passes. The cut work residual stays
within 3.5e-16 J, while mean log plastic Jacobian increases
in all four cases. Eliminating one positive trace contribution
does not remove the coupled pile failure. Temporary edits were restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-paired-floor-normal-diagonal-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Halve the wall-normal diagonal in both paired operators

Repeat the paired floor-normal control on 255739e with half
of the yy component retained in deformation and stress scatter.

| Native case | Envelope at 3 s | Radius (m) | Mean log plastic Jacobian |
|---|---:|---:|---:|
| Coarse | 32.900528° | 0.166899 | 0.355598 |
| Coarse half-step | 39.852304° | 0.154416 | 0.175731 |
| Fine | 31.122463° | 0.160001 | 0.290157 |
| Fine half-step | 34.773384° | 0.155880 | 0.192113 |

Mean log plastic Jacobian lies between baseline and full
removal in each case. The fine nominal misses the radius by
about 1 µm and the coarse half-step remains nearly 5° above
the angle limit. The paired work residual stays within 4.1e-16 J.
The temporary changes were restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-paired-half-normal-diagonal-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Check long-horizon creep with the paired half-normal term

On 255739e, extend the native fine half-step paired half-normal
control through 12 s. The envelope drops from 34.773384° at
3 s to 30.756089° at 6 s, 27.343667° at 9 s, and 24.483701°
at 12 s. The radius stays at 0.155880 m from 3 s onward.
The control loses 37.25 mm of height after 3 s, 3.51 mm
more than baseline. The paired cut work residual is
−1.67e-16 J. The temporary edits were restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-half-normal-long-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 12
```

### Increase the paired wall-normal diagonal by half

On 255739e, multiply the yy contribution by 1.5 for
particles starting on the floor in both stress scatter and
deformation gather. Run the native fine half-step to 12 s.
The angle is 35.851866° at 3 s, 31.591160° at 6 s,
28.328116° at 9 s and 26.114338° at 12 s. Radius stays at
0.157848 m from 3 s onward. The apex loses 36.68 mm after 3 s,
2.94 mm more than baseline despite the higher final angle.
The paired work residual is −2.41e-16 J. Scaling the paired
normal term in either direction leaves late creep unresolved.
The temporary edits were restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-strong-normal-long-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 12
```

### Suppress positive normal strain during sustained floor contact

On 255739e, keep the existing scatter and wall reaction.
For trial deformation only, cap the yy velocity-gradient
component at zero when the particle starts and ends on the
floor. On the native fine half-step the envelope falls from
35.497491° at 3 s to 32.505937° at 6 s, 30.767982° at 9 s,
and 27.060967° at 12 s. The radius stays at 0.156454 m.

The control delays the drop below 30° and still loses
31.66 mm of height after 3 s. The stress work receipt uses
the original derivative, so it cannot certify the capped
deformation. The temporary solver change was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-sustained-normal-extension-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 12
```

### Set the active below-floor normal ghost velocity to zero

On 255739e, retain the tangent continuation and replace reflected
below-floor normal ghost rows with empty rows. Use the same rows
for the stress transpose and deformation gather.

| Native case | Envelope at 3 s | Radius (m) |
|---|---:|---:|
| Coarse | 31.046973° | 0.164257 |
| Coarse half-step | 38.758295° | 0.149804 |
| Fine | 30.509111° | 0.157475 |
| Fine half-step | 34.180426° | 0.154918 |

The coarse half-step angle falls 1.74° and remains above 35°.
The coarse nominal radius still exceeds 0.16 m. Both fine
cases pass at 3 s and cut work residual stays within 3e-16 J.
The temporary solver edit was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-zero-normal-ghost-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Check long-horizon creep with zero normal ghost continuation

Run the same native fine half-step binary with zero normal ghost
continuation through 12 s. The envelope is 34.180426° at 3 s,
30.220503° at 6 s, 27.485918° at 9 s and 24.457285° at 12 s.
The radius stays at 0.154918 m. The apex loses 34.74 mm after
3 s, about 1 mm more than baseline. The paired cut work residual
is 8.26e-17 J. The early angle pass is temporary.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-zero-normal-ghost-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 12
```

### Track particle height quantiles during late settlement

On 255739e, sort the native fine half-step particle heights at
each sample and use index floor(p * (N - 1)) for each quantile.

| Time (s) | Median (mm) | 90th percentile (mm) | 95th percentile (mm) | Maximum (mm) |
|---|---:|---:|---:|---:|
| 3 | 9.289 | 60.528 | 63.520 | 108.205 |
| 6 | 0 | 30.998 | 53.389 | 93.434 |
| 9 | 0 | 1.678 | 44.700 | 83.067 |
| 12 | 0 | 0 | 34.076 | 74.464 |

The median reaches the floor by 6 s, and the 90th percentile
reaches it by 12 s. The 95th percentile and apex also descend.
The loss of height affects the full distribution. The original
angle estimator is used for the gate. Temporary tracing was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-height-distribution-long-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 12
```

### Check long-horizon creep under a half-cell horizontal grid shift

On 255739e, set the fine grid origin to (h/2, 0, h/2) and
run the native half-step through 12 s. Keep the release and
floor at their original world coordinates.

| Time (s) | Envelope | Radius (m) |
|---|---:|---:|
| 3 | 34.880324° | 0.171549 |
| 6 | 28.998285° | 0.177584 |
| 9 | 25.796791° | 0.177584 |
| 12 | 22.964356° | 0.177584 |

The shifted pile loses 44.34 mm of height after 3 s compared
with 33.74 mm at the original phase. The 12 s radius exceeds
0.16 m and the angle falls below 30°. Late height loss persists
under both grid alignments. The temporary fixture was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-half-phase-long-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 12
```

### Resolve the from-rest vertical acceleration by height

On 255739e, repeat the loaded one-step probe on copies of
the native fine half-step state. Group by initial height.
Report mean stored vertical acceleration and track the
highest particle in each snapshot separately.

| Time (s) | Floor (m/s²) | Above floor and below 50 mm (m/s²) | At least 50 mm (m/s²) | Apex (m/s²) |
|---|---:|---:|---:|---:|
| 3 | 0 | −4.234 | 1.089 | 0.643 |
| 6 | 0 | −3.330 | 0.598 | 0.824 |
| 9 | 0 | −6.800 | −0.276 | 0.077 |

The apex accelerates upward in all three probes. The lower
group accelerates downward even when the upper group is
supported on average. This localizes the net downward
acceleration in the from-rest test. The trajectory remains
unchanged and the temporary instrumentation was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-rest-support-bands-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 12
```

### Halve the timestep only after the pile forms

On 255739e, run the native fine half-step column to 3 s, then
switch from 0.00078125 s to 0.000390625 s for the remaining
steps. The 3 s state and geometry match the baseline.

| Time (s) | Baseline envelope | Envelope with smaller late steps |
|---|---:|---:|
| 3 | 34.693822° | 34.693822° |
| 6 | 30.869627° | 32.588625° |
| 9 | 27.988015° | 30.869518° |
| 12 | 25.473317° | 29.370816° |

The late height loss drops from 33.74 mm to 20.24 mm.
The outer radius remains 0.156304 m. The timestep affects
creep from the same formed pile, and the smaller step still
falls below 30° by 12 s. The temporary fixture was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-late-half-timestep-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 12
```

### Compare wall support at a fixed formed state across timesteps

On 255739e, copy the native fine half-step column at 3 s
and advance each copy once with its original timestep, half
of that timestep, or one quarter. Preserve carried motion and
material history. Divide each vertical wall impulse by its own
step duration.

| Step (ms) | Grid support (N) | Swept particle support (N) | Particle contacts |
|---|---:|---:|---:|
| 0.781250 | 14.154662 | 8.987635 | 17 |
| 0.390625 | 14.154662 | 13.263539 | 17 |
| 0.195313 | 14.154662 | 21.815347 | 17 |

The grid force is unchanged. Swept support increases in units
of force as the timestep shrinks, with the same contact count.
The impulse data fit a step-independent swept contribution of
about 3.34 mN·s plus a duration-scaled contribution. This test
isolates the immediate timestep response. The temporary probe
returned after the copied steps and bypassed the endpoint gate.
The fixture source was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-same-state-support-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 12
```

### Separate translation and affine motion in matched-state wall support

Repeat the one-step native support probe at 3 s on 255739e.
Reset carried particle translation and APIC slopes independently
on copies of the formed pile.

| Carried motion cleared | Swept support at 0.781250 ms (N) | 0.390625 ms (N) | 0.195313 ms (N) |
|---|---:|---:|---:|
| Neither | 8.988 | 13.264 | 21.815 |
| Translation | 4.182 | 3.652 | 2.591 |
| APIC slopes | 9.518 | 14.324 | 23.936 |
| Both | 4.712 | 4.712 | 4.712 |

The grid contribution stays at 14.155 N and all probes record
17 swept contacts. The carried translation supplies the
step-independent closing impulse in this snapshot. The affine
contribution partially offsets it. Clearing both yields a
duration-proportional swept impulse. The probe exits after
the copied steps and does not test settlement. The temporary
fixture was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-same-state-motion-probe-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 12
```

### Clear downward carried translation before each late step

On 255739e, use the native fine half-step through 3 s. Before
each subsequent attempt, zero negative stored vertical velocity
in one of two groups. Account for each reset as external
vertical impulse and leave the deformation and slopes intact.

| Group reset | Envelope at 6 s | 9 s | 12 s |
|---|---:|---:|---:|
| On floor | 30.869627° | 27.988015° | 25.473317° |
| Above floor | 31.939357° | 29.838500° | 27.992297° |

The floor-only reset reproduces baseline at printed precision.
The above-floor reset retains 8.62 mm more height by 12 s and
still falls below 30°. The outer radius stays at 0.156304 m
in both cases. The resets are diagnostic external impulses.
The temporary fixture edits were restored.

```powershell
foreach ($variant in 'late-floor-vertical-reset', 'late-above-floor-reset') {
  & "out/pr-m5-sand-affine/out/sand-affine-$variant-bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 fine-half all 12
}
```

### Shift the vertical grid phase through late settlement

On 255739e, set the fine grid origin to (0, h/2, 0) while
keeping the world-space floor, release and half-step duration.

| Time (s) | Envelope | Radius (m) |
|---|---:|---:|
| 3 | 33.904352° | 0.165444 |
| 6 | 30.013632° | 0.166312 |
| 9 | 27.514416° | 0.168046 |
| 12 | 25.801822° | 0.169707 |

The shifted apex loses 29.15 mm after 3 s, compared with
33.74 mm in the aligned case. The radius continues growing
and the endpoint misses both bounds. Late creep persists
when the floor lies between normal velocity faces. The
temporary fixture was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-vertical-half-phase-long-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 12
```

### Double the timestep only after the pile forms

On 255739e, start from the same native fine half-step state
at 3 s and double the timestep to 0.0015625 s. Compare the
three schedules that share the original 3 s state.

| Late timestep (ms) | Angle at 6 s | 9 s | 12 s | Height loss after 3 s (mm) |
|---|---:|---:|---:|---:|
| 0.390625 | 32.589° | 30.870° | 29.371° | 20.24 |
| 0.781250 | 30.870° | 27.988° | 25.473° | 33.74 |
| 1.562500 | 27.944° | 23.566° | 20.430° | 49.98 |

Height loss increases across all three late step sizes with
the same initial geometry and unchanged printed radius of
0.156304 m. Even the smallest tested late step misses the
12 s angle bound. The temporary fixture was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-late-double-timestep-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 12
```

### Substep deformation and return on a frozen grid gradient

On 255739e, run the native fine half-step through 12 s. Split
each compatible-floor constitutive update into two equal
Euler increments using the same gathered gradient. Apply
granular return after each increment. Grid transfer, wall
contact and advection still advance at the original timestep.

The envelope angles at 3, 6, 9 and 12 s are 34.683069°,
30.849602°, 27.959452° and 25.451087°. The largest difference
from baseline is 0.029°. The 12 s radius is 0.156627 m and
late height loss is 33.84 mm. Splitting return on the frozen
gradient leaves the observed creep nearly intact. The
temporary solver change was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-constitutive-substeps-long-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 12
```

### Alternate late step lengths at the same mean timestep

On 255739e, form the native fine half-step pile through 3 s.
Then alternate 0.390625 ms and 1.171875 ms steps. The mean is
the original 0.78125 ms, so the number of transfer and contact
updates per pair of steps is unchanged.

| Time (s) | Constant timestep angle | Alternating timestep angle |
|---|---:|---:|
| 3 | 34.693822° | 34.693822° |
| 6 | 30.869627° | 30.279380° |
| 9 | 27.988015° | 26.906364° |
| 12 | 25.473317° | 24.331743° |

The alternating schedule loses 37.53 mm of height after 3 s,
3.79 mm more than the constant schedule. The final radius
matches at 0.156304 m. The timestep distribution affects
creep at the same mean transfer frequency. The temporary
fixture change was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-late-alternating-timestep-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 12
```

### Reverse the phase of the alternating late timestep

Repeat the fine half-step alternating schedule on 255739e with
the long and short steps exchanged. At 6, 9 and 12 s the
angles are 30.278973°, 26.906027° and 24.331512°. All differ
from the original alternating schedule by less than 0.0005°.
The radius matches at 0.156304 m. For this schedule the increased
creep persists under phase reversal. The temporary fixture was
restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-late-reversed-timestep-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 12
```

### Keep the unprojected APIC slope after swept contact

On c58b4ba, keep the compatible-floor fitted slope when a
particle hits the floor. Preserve the swept velocity update,
deformation gather and stress transpose. Run the native
fine half-step to 12 s.

| Time (s) | Envelope | Radius (m) |
|---|---:|---:|
| 3 | 34.501934° | 0.161718 |
| 6 | 30.693454° | 0.161718 |
| 9 | 27.664889° | 0.161718 |
| 12 | 25.355121° | 0.161718 |

The apex loses 34.52 mm after 3 s, compared with 33.74 mm
with projection. The radius exceeds the limit and the angle
continues to fall below 30°. The temporary solver edit was
restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-no-affine-wall-projection-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 12
```

### Probe wall support with one compressed particle

On c58b4ba, place a stationary 1 kg granular particle on the
floor of a unit-width grid. Use reference volume 0.1 m³ and
isotropic elastic stretch 0.95. Advance independent copies for
1 ms and 0.5 ms. With gravity −1 m/s², both steps produce
0.0666813 N of reconstructed grid support and 0.933319 N of
swept support, with zero final vertical velocity. With gravity
zero, the swept support vanishes and the same grid reaction
accelerates the particle upward at 0.0666813 m/s².

This minimal native case reproduces the two support paths and
exhibits duration-proportional impulse from rest. It supplies a
reference for testing changes to wall transfer before rerunning
the pile. The temporary fixture was restored. Q-SAND remains open.

### Measure deformation drift for the stationary wall particle

Extend the compressed single-particle control on c58b4ba to
one second under gravity −1 m/s². Keep the same initial stress.
In each run the final particle height and normal velocity are zero.

| Timestep (ms) | Final elastic yy entry | Final elastic Jacobian |
|---|---:|---:|
| 1 | 0.949494657882 | 0.857043687601 |
| 0.5 | 0.949747022563 | 0.857209061712 |
| 0.25 | 0.949873434515 | 0.857291960170 |
| 0.125 | 0.949936698045 | 0.857333462395 |

The initial yy entry is 0.95 and the initial Jacobian is
0.857375. Halving the step approximately halves the drift
in each quantity at fixed physical time. The static particle
therefore gives a small reproducible strain-drift case even
with exact zero normal translation at accepted endpoints.
The temporary source was restored.

```powershell
foreach ($variant in 'single-strain', 'single-strain-refined') {
  & "out/pr-m5-sand-affine/out/sand-affine-$variant-bin/fluidity_m5_dry_scene_tests.exe" --q-sand-single-particle-strain
}
```

### Clamp the wall-face normal velocity in the stationary strain control

Repeat the one-second stationary particle probe with incoming normal
velocity clamped on grid faces exactly at the floor. At 1 ms, grid
support rises from 0.067010 N to 0.924153 N and swept support falls
from 0.932990 N to 0.075847 N. Their sum remains 1 N. The final
elastic yy entry and Jacobian reproduce the unclamped values to
12 significant digits at all four timesteps. The clamp changes
the transport and impulse partition while the deformation
reconstruction already assigns zero velocity to the wall face.
The temporary solver and fixture edits were restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-single-strain-clamp-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-single-particle-strain
```

### Compare zero ghost continuation in the stationary strain control

Repeat the four native refinements on c58b4ba with zero
below-floor normal ghost continuation and its paired transpose.

| Timestep (ms) | Final elastic yy entry | Final elastic Jacobian |
|---|---:|---:|
| 1 | 0.949636113758 | 0.857171328294 |
| 0.5 | 0.949817992901 | 0.857273106617 |
| 0.25 | 0.949908980437 | 0.857324038906 |
| 0.125 | 0.949954486213 | 0.857349515852 |

Normal ghost reflection contributes to the strain drift.
Zero continuation reduces the yy drift by about 28% at each
timestep, with a first-order residual. The accepted normal
position and velocity remain zero and total support remains
1 N. The temporary edits were restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-single-zero-ghost-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-single-particle-strain
```

### Vary the load in the stationary particle strain probe

Keep the initial isotropic stretch at 0.95 and step the
single particle for one second at 1 ms on c58b4ba. All four
loads maintain zero final normal position and velocity.

| Gravity magnitude (m/s²) | Final elastic yy entry | Final elastic Jacobian |
|---|---:|---:|
| 0.25 | 0.950205304844 | 0.857684929967 |
| 0.5 | 0.949968363503 | 0.857471129273 |
| 1 | 0.949494657882 | 0.857043687601 |
| 2 | 0.948547954460 | 0.856189442739 |

With the initial stress held fixed across runs, the normal
strain drift changes sign between 0.25 and 0.5 m/s². The
change in the final yy entry is approximately linear in
the applied gravity over the tested range. The sweep balances
the remainder of the load at each endpoint. The temporary
fixture was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-single-load-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-single-particle-strain
```

### Exclude the uniform gravity kick from the deformation donor field

On 5ff4866, subtract the uniform gravity increment from the
live normal donors before constructing the compatible floor
deformation field. Leave particle transport and stress work
accounting on their original fields.

| Timestep (ms) | Final elastic yy entry | Final elastic Jacobian |
|---|---:|---:|
| 1 | 0.950442305223 | 0.857898783916 |
| 0.5 | 0.950221433598 | 0.857637178762 |
| 0.25 | 0.950110787173 | 0.857506161210 |
| 0.125 | 0.950055411196 | 0.857440598579 |

Excluding gravity changes the drift from compression to
extension and leaves first-order convergence. The stress
kick and wall reconstruction can therefore generate strain
in this fixture without the gravity term in the gradient.
The two fields make this a diagnostic only. The temporary
solver change was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-stationary-gravity-free-gradient-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-single-particle-strain
```

### Exclude the gravity kick from deformation in the sand matrix

Apply the stationary probe's gravity-free donor field on 5ff4866
to the standard native column. Transport still receives gravity.

| Case | Envelope at 3 s | Radius (m) | Cut work difference (J) |
|---|---:|---:|---:|
| Coarse | 30.471411° | 0.159529 | −0.686541 |
| Coarse half-step | 39.298344° | 0.148766 | −0.442620 |
| Fine | 26.529011° | 0.156566 | −0.521062 |
| Fine half-step | 34.087972° | 0.151180 | −0.365688 |

The coarse nominal and fine half-step pass the endpoint checks.
The other angles fail in opposite directions. The different
velocity fields used for force work and deformation produce
a substantial work difference. The temporary edit was restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-sand-gravity-free-gradient-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Check long-horizon settlement with the gravity-free donor field

On 5ff4866, extend the native fine half-step gravity-free
deformation diagnostic to 12 s. Its angle drops from 34.087972°
at 3 s to 25.760826° at 6 s, 17.307059° at 9 s, and 8.472996°
at 12 s. The apex loses 79.77 mm after 3 s, versus 33.74 mm
with the original gradient. The final radius is 0.151327 m
and the cut work difference is −0.584012 J. The apparent
3 s pass does not persist. The temporary edit was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-long-gravity-free-gradient-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-symmetric-affine-decay 0 fine-half all 12
```

### Use the pre-force normal field for deformation

On 5ff4866, reconstruct normal deformation donors from deposited
momentum divided by mass, before gravity and stress acceleration.
Keep tangential deformation, transport and stress scatter on their
ordinary paths. The stationary particle retains its initial yy
stretch of 0.95 to printed precision at all four timesteps.
The total elastic Jacobian still drifts, reaching 0.857499676038
at 1 ms against 0.857375 initially.

The native fine half-step column reaches 29.619063° at 12 s and
loses 29.02 mm of height after 3 s. Its radius exceeds the limit
at 0.175142 m. The cut work difference grows to 2.657 J because
the deformation and work fields differ. The temporary solver
change was restored.

```powershell
$exe = 'out/pr-m5-sand-affine/out/sand-affine-pre-force-normal-gradient-bin/fluidity_m5_dry_scene_tests.exe'
& $exe --q-sand-single-particle-strain
& $exe --q-sand-symmetric-affine-decay 0 fine-half all 12
```

### Use the full pre-force field for deformation

On 5ff4866, build all three deformation donor components
from deposited momentum divided by mass, before the force
and wall kicks. The stationary native particle retains yy
stretch 0.95 and Jacobian 0.857375 at all four tested steps,
to printed precision. The grid and swept support stay at
0.066681283 N and 0.933318717 N respectively.

In the native fine half-step pile, this control reaches
35.231817° and a 0.187719 m radius at 3 s. The difference
between deformation and force-work fields accumulates
1.250275 J. The stationary result identifies the post-force
field as necessary for the observed drift in that probe.
The unmatched update fails both pile bounds. The temporary
solver change was restored.

### Remove the mean normal kick on the active floor patch

On 5ff4866, compute the mass-weighted post-force minus deposited
normal velocity over occupied faces in the active floor patch.
Subtract that common increment from those deformation donors
before ghost reconstruction. Transport and the original force
work still use the post-force field.

The stationary particle ends at yy stretch 0.950379238675 at
1 ms and 0.950047497192 at 0.125 ms. The extension persists
and decreases approximately in proportion to timestep. The
native fine half-step pile reaches 34.557628° and a radius of
0.156856 m at 3 s, with a cut work difference of −0.073219 J.
This field subtraction preserves relative increments within
the patch but changes the work pairing. Temporary edits were
restored and Q-SAND remains open.

### Pair floor-patch mean removal with its force transpose

On 5ff4866, subtract the mass-weighted mean normal velocity
on the occupied floor-patch faces before applying the ghost
gather. After ghost force scatter, subtract the summed patch
normal force distributed in proportion to face mass. Record
the resulting net force change in the wall impulse budget.
This projects the velocity itself, including carried motion.

| Native case | Envelope at 3 s | Radius (m) |
|---|---:|---:|
| Coarse | 29.936016° | 0.180357 |
| Coarse half-step | 36.765210° | 0.180493 |
| Fine | 27.682320° | 0.164679 |
| Fine half-step | 34.693725° | 0.163743 |

The maximum absolute cut work residual is 3.82e-16 J.
All four piles exceed the radius limit. The patch-wide
projection preserves the work pairing and changes the
stress-driven transport. The temporary solver changes were
restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-paired-patch-projection-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Project mean normal motion within each floor-patch column

On 5ff4866, group occupied normal faces in the floor patch by
horizontal anchor. Subtract the mass-weighted velocity mean in
each column before ghost reconstruction. Apply the transpose to
the scattered normal force column by column and record the
net force change in the wall impulse budget.

| Native case | Envelope at 3 s | Radius (m) |
|---|---:|---:|
| Coarse | 31.679662° | 0.181636 |
| Coarse half-step | 37.547069° | 0.180934 |
| Fine | 30.247284° | 0.165754 |
| Fine half-step | 35.398363° | 0.162887 |

The paired work residual stays within 7.4e-16 J. All four radii
exceed 0.16 m, as with the patch-wide projection. The temporary
solver changes were restored.

```powershell
$bin = 'out/pr-m5-sand-affine/out/sand-affine-paired-column-projection-bin'
foreach ($case in 'coarse', 'coarse-half', 'fine', 'fine-half') {
  & "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 $case all 3
}
```

### Balance the initial normal strain increment in the stationary probe

On 43ec90c, interpolate the one-step change in elastic yy
stretch between loads of 0.25 and 0.5 m/s². The zero of that
response is at gravity −0.466768979359 m/s². Hold that load
fixed for four one-second native refinements.

| Timestep (ms) | Final elastic yy entry | Final elastic Jacobian |
|---|---:|---:|
| 1 | 0.949999855312 | 0.857499545466 |
| 0.5 | 0.949999963774 | 0.857437320415 |
| 0.25 | 0.949999990937 | 0.857406172152 |
| 0.125 | 0.949999997734 | 0.857390589067 |

The residual normal drift decreases by about fourfold on
each halving, while total volume drift from 0.857375 halves.
The load cancels the leading normal contribution only.
Position and vertical velocity remain zero at the endpoints.
The temporary fixture was restored.

```powershell
& 'out/pr-m5-sand-affine/out/sand-affine-balanced-stationary-load-bin/fluidity_m5_dry_scene_tests.exe' --q-sand-balanced-load
```

### Reproduce the sand timestep and stationary wall probes from source

Build both dry-scene executables from `43ec90c` on
`codex/m5-sand-affine`. Use the same Windows Release SM120
configuration as the main qualification build. In the commands
below, set `$bin` to the directory containing those executables.

```powershell
& "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-single-particle-strain
& "$bin/fluidity_cuda_m5_dry_scene_tests.exe" --q-sand-single-particle-strain
& "$bin/fluidity_m5_dry_scene_tests.exe" --q-sand-symmetric-affine-decay 0 fine-half all 12 alternating-reversed
```

The stationary command checks exact floor position and vertical
velocity at every step, relative wall impulse error of at most
1e-9, and the refinement bound for the one-second yy strain drift.
Native and SM120 pass all four timestep refinements and agree
to printed precision. The alternating command reproduces
24.331512° at 12 s and exits with status 1 on the angle check.
The `half`, `double`, and `alternating` options apply their named
schedules after 3 s. Omit the schedule to use constant steps.

### Shift the horizontal phase of the stationary particle

On 43ec90c, run the stationary native probe at 1 ms for one
second. Change x and z together and allocate two blocks along
each horizontal axis so every phase has full stencil support.

| x and z (m) | Final elastic yy entry | Final elastic Jacobian |
|---|---:|---:|
| 2.000 | 0.949494669937 | 0.857033307788 |
| 2.125 | 0.949494694172 | 0.857035761616 |
| 2.250 | 0.949494657882 | 0.857043687601 |
| 2.375 | 0.949494639337 | 0.857059654457 |
| 2.500 | 0.949494228338 | 0.857413422584 |

Normal compression is similar across all phases. Total volume
drift changes sign at the half-cell phase, where the tangential
strain more than offsets normal compression. The smaller
allocation rejects that phase, and the wider allocation gives
the same result at the other four phases to printed precision.
The temporary fixture was restored.

### Resolve stationary half-cell strain into its diagonal components

At horizontal position (2.5, 2.5) m on the unit-width grid,
repeat the stationary native probe through one second.
Both horizontal elastic stretch entries agree to printed precision.

| Timestep (ms) | Horizontal stretch | Vertical stretch | Elastic Jacobian |
|---|---:|---:|---:|
| 1 | 0.950274279032 | 0.949494228338 | 0.857413422584 |
| 0.5 | 0.950137195880 | 0.949746914952 | 0.857394181213 |
| 0.25 | 0.950068612043 | 0.949873407584 | 0.857384583051 |
| 0.125 | 0.950034309549 | 0.949936691308 | 0.857379789632 |

The two horizontal extensions together outweigh vertical
compression at every refinement. Both contributions are
approximately first order in the timestep. The temporary
fixture was restored.

### Use pre-force tangential donors for deformation

On 43ec90c, reconstruct x and z deformation donors from
deposited momentum divided by mass. Keep the post-force normal
field and leave the original transport and stress work paths.

At 1 ms the stationary particle ends at yy stretch 0.949494802608
and Jacobian 0.856919059353. The original Jacobian was
0.857043687601 at the same step. The near-identical normal
compression produces more total compression once the horizontal
post-force response is removed.

The native fine half-step ends at 35.059702° and 0.181033 m
at 3 s, with a cut work difference of 4.115359 J. The
deformation-only change fails both bounds and was restored.

### Project resting particle normal velocity on the grid

On 43ec90c, perform four projected Gauss–Seidel sweeps on the
normal velocity at particles starting on the floor. Use the
compatible PIC stencil for the constraint and its transpose
for the face impulse. Keep impulses unilateral and record
them in the grid wall budget. The particle sweep still resolves
remaining contact. The tangential cap still uses its original
stress-derived reaction.

| Native case | Envelope at 3 s | Radius (m) |
|---|---:|---:|
| Coarse | 33.250446° | 0.170759 |
| Coarse half-step | 41.238694° | 0.153238 |
| Fine | 31.264093° | 0.168854 |
| Fine half-step | 36.414652° | 0.157928 |

All four cases complete and miss at least one bound. The paired
cut stress work residual stays within 1.6e-16 J. This control
adds grid support before deformation while retaining the original
friction cap. The temporary solver changes were restored.

### Include projected normal support in the grid friction cap

On 43ec90c, retain four projected normal sweeps. Scatter
the final per-particle normal impulse into the reaction map
using the compatible normal stencil, in addition to the
existing stress-derived reaction. Keep the particle sweep
for any remaining contact.

| Native case | Envelope at 3 s | Radius (m) |
|---|---:|---:|
| Coarse | 33.912205° | 0.170983 |
| Coarse half-step | 41.440564° | 0.158930 |
| Fine | 31.876072° | 0.165500 |
| Fine half-step | 36.073858° | 0.160435 |

All four cases miss at least one bound. The absolute cut work
residual stays below 1.9e-16 J. This control uses the updated
normal reaction in a subsequent tangential solve, with no
coupled iteration between them. The temporary edits were restored.

### Apply grid friction at the projected particle contacts

On 43ec90c, use the compatible tangent stencils to gather and
scatter friction at each particle after the four normal sweeps.
The Coulomb cap uses that particle's projected normal impulse.
This replaces the stress-reaction tangential solve for the trial.

| Native case | Envelope at 3 s | Radius (m) |
|---|---:|---|
| Coarse | 37.038025° | 0.150993 |
| Coarse half-step | 40.657314° | 0.143107 |
| Fine | 31.283617° | 0.161880 |
| Fine half-step | 37.217962° | 0.148860 |

Three cases exceed the angle limit and the fine nominal exceeds
the radius limit. Compared with the reaction-map control, all
four radii decrease. The cut stress work residual stays within
1.2e-16 J. The temporary edits were restored.

### Run projected normal contact on the stationary refinement probe

On 43ec90c, apply the four normal grid sweeps to the stationary
particle. The exact-zero check rejects step zero. A temporary
1e-12 position and velocity tolerance accepts all steps and
retains the original impulse balance bound. Under that tolerance
the original strain bound still fails at 1 ms.

| Timestep (ms) | Final elastic yy entry | Final elastic Jacobian |
|---|---:|---:|
| 1 | 0.952442859155 | 0.859703998878 |
| 0.5 | 0.951177532112 | 0.858499990218 |
| 0.25 | 0.950646483309 | 0.857989605033 |
| 0.125 | 0.950331207800 | 0.857689499379 |

All support moves to the grid at 1 N. The normal strain drift
changes to extension and is larger in magnitude at every
refinement. Support at the particle's interpolated velocity
does not constrain the local deformation gradient. Temporary
diagnostic tolerances and solver edits were restored.

### Project the wall-normal strain rate in the continued field

On 43ec90c, visit particles starting on the floor once and
project each normal velocity stencil against its vertical
shape derivative. Use the modified continued field for the
gather and keep the original force scatter and transport.
Overlapping stencils can change previously visited constraints.

The single-particle probe retains yy stretch 0.95 at all four
timesteps. Its total volume still drifts through tangential
strain. The fine half-step pile reaches 29.417241° with a
0.191044 m radius at 3 s, and its stress work difference is
0.974122 J. The unpaired projection is a diagnostic and fails
the pile bounds. The temporary code was restored.

### Measure stress-kick self-work during creep

On 43ec90c, sum half the squared scattered stress impulse
divided by force mass on occupied faces. This is the quadratic
term in the grid kinetic-energy change from the stress kick.
Compare it with stress impulse dotted into the final grid
velocity for the native fine half-step.

| Interval | Stress-kick quadratic term (J) | Stress work at final velocity (J) |
|---|---:|---:|
| 0–3 s | 0.544661 | −2.065886 |
| 3–6 s | 0.219911 | −0.008212 |
| 6–9 s | 0.159695 | 0.003841 |
| 9–12 s | 0.069781 | −0.000444 |

The quadratic term is much larger than the net final-velocity
work in the late intervals. The latter includes the effects
of gravity and wall traction on the velocity used in the
receipt. This comparison does not close the full energy
budget. The temporary trace was restored.

### Measure grid kinetic work at the stress kick

On 43ec90c, multiply each applied stress impulse by its
face velocity averaged immediately before and after the
stress kick. Sum the result before gravity and wall traction.
This measures the actual grid kinetic-energy change across
that operation in the native fine half-step.

| Interval | Midpoint stress work (J) | Stress work at final velocity (J) |
|---|---:|---:|
| 0–3 s | −1.981555 | −2.065886 |
| 3–6 s | 0.054758 | −0.008212 |
| 6–9 s | 0.051397 | 0.003841 |
| 9–12 s | 0.053892 | −0.000444 |

The stress kick adds 0.160047 J of grid kinetic energy from
3 to 12 s. The end-velocity receipt sums to −0.004815 J over
the same interval. The kick can release elastic energy, so a
full balance needs the material energy change and transfer
losses as well. The trace was restored.

### Account for accepted elastic energy and plastic return loss

On 43ec90c, evaluate the reference-volume-weighted Hencky
energy at the accepted start, trial and returned state on
each native fine half-step. The trial change is the accepted
change plus the energy removed by return.

| Interval | Accepted elastic energy change (J) | Return loss (J) | Trial elastic energy change (J) |
|---|---:|---:|---:|
| 0–3 s | 0.087729 | 2.043434 | 2.131164 |
| 3–6 s | −0.047380 | 0.055651 | 0.008271 |
| 6–9 s | −0.018459 | 0.014636 | −0.003823 |
| 9–12 s | −0.005119 | 0.005571 | 0.000452 |

From 3 to 12 s, return removes 0.075858 J, while accepted
elastic energy falls 0.070958 J. The net trial gain is only
0.004900 J. Over the same interval the final-velocity stress
work is −0.004815 J, close to the negative of the trial gain.
The separate midpoint grid work remains 0.160047 J. A full
energy balance must account for the work evaluated at different
stages and for transfers. The temporary trace was restored.

### Separate gravity and traction contributions to the stress work receipt

On 43ec90c, dot the applied stress impulse with each successive
normal and tangent face velocity in the native fine half-step.
The midpoint column uses the average immediately across the
stress kick. The other columns use the velocity at the named stage.

| Interval | Stress midpoint (J) | Stress endpoint (J) | After gravity (J) | After grid traction (J) |
|---|---:|---:|---:|---:|
| 0–3 s | −1.981555 | −1.436895 | −1.840663 | −2.065886 |
| 3–6 s | 0.054758 | 0.274670 | 0.018456 | −0.008212 |
| 6–9 s | 0.051397 | 0.211092 | 0.024016 | 0.003841 |
| 9–12 s | 0.053892 | 0.123673 | 0.008923 | −0.000444 |

From 3 to 12 s, the stress endpoint adds 0.449387 J relative
to the midpoint evaluation. Gravity shifts that receipt down by
0.558040 J and grid traction by another 0.056201 J. These
differences are cross terms with the stress impulse. The
trace was restored and the trajectory reproduces baseline.

### Measure transfer-stage kinetic energy during late settlement

On 43ec90c, trace occupied-grid kinetic energy just after APIC
deposit and just before PIC gather on each accepted native fine
half-step. Compare with particle translational kinetic energy
at step start and just before the particle wall sweep,
respectively. The stored-velocity blend is zero in this case.

| Interval | Grid after deposit minus starting particle translation (J) | Gathered particle translation minus grid before gather (J) |
|---|---:|---:|
| 0–3 s | 12.072329 | −14.833539 |
| 3–6 s | −0.002789 | −0.137145 |
| 6–9 s | −0.044268 | −0.100440 |
| 9–12 s | −0.000945 | −0.038819 |

The gather-stage change sums to −0.276403 J from 3 to 12 s,
compared with 0.160047 J added by the stress kick over that
interval. Deposit uses APIC affine velocity, while the particle
reference includes translational energy only. The deposit
column therefore measures a change between energy representations.

Run the instrumented native scene binary with
`--q-sand-symmetric-affine-decay 0 fine-half all 12`.
The run reproduced the 25.473317° endpoint and exited 1.
The temporary instrumentation was restored.

### Resolve PIC kinetic energy loss by stencil and direction

On 43ec90c, sum the weighted squared deviation of each occupied
face velocity from its particle's compatible PIC velocity,
multiplied by half the particle mass. Group each component
stencil by whether it excludes a structurally incompatible
sample. Use the same native fine half-step and zero retained
velocity as the transfer-stage trace.

| Interval | Full x (J) | Full y (J) | Full z (J) | Cut x (J) | Cut y (J) | Cut z (J) |
|---|---:|---:|---:|---:|---:|---:|
| 0–3 s | 0.649055 | 12.135621 | 0.649141 | 0.123398 | 1.152777 | 0.123547 |
| 3–6 s | 0.001668 | 0.064990 | 0.001684 | 0.007858 | 0.053120 | 0.007825 |
| 6–9 s | 0.000772 | 0.004275 | 0.000784 | 0.001049 | 0.092506 | 0.001053 |
| 9–12 s | 0.000380 | 0.003305 | 0.000389 | 0.000179 | 0.034381 | 0.000185 |

From 3 to 12 s, the six components sum to 0.276403 J,
matching the independently traced gather energy loss to
six decimal places. The cut normal component contributes
0.180007 J, or 65%. All normal components together contribute
0.252578 J, or 91%. This localizes the loss to vertical
velocity variation, with the largest share in floor-cut
stencils. The variance does not identify which grid operation
created that variation.

Run the instrumented native scene binary with
`--q-sand-symmetric-affine-decay 0 fine-half all 12`.
It reproduced 25.473317° and exited 1. The isolated solver
source was restored and its worktree was clean.

### Attribute vertical PIC variance to the stress kick

On 43ec90c, center the post-deposit vertical velocity and the
vertical stress-induced velocity increment separately within each
compatible PIC stencil. Sum half the particle mass times the
weighted square of each centered field and twice their product.
The three terms add to the post-stress variance. The final
pre-gather measurement includes gravity and grid wall traction.

| Interval | Stencil | After deposit (J) | Stress increment alone (J) | Cross term (J) | After stress and before gather (J) |
|---|---|---:|---:|---:|---:|
| 0–3 s | Full | 12.111978 | 0.062035 | −0.038392 | 12.135621 |
| 0–3 s | Cut | 1.196594 | 0.040184 | −0.084000 | 1.152777 |
| 3–6 s | Full | 0.029180 | 0.037597 | −0.001787 | 0.064990 |
| 3–6 s | Cut | 0.008877 | 0.059547 | −0.015304 | 0.053120 |
| 6–9 s | Full | 0.001803 | 0.002026 | 0.000447 | 0.004275 |
| 6–9 s | Cut | 0.017751 | 0.093439 | −0.018684 | 0.092506 |
| 9–12 s | Full | 0.001309 | 0.002669 | −0.000673 | 0.003305 |
| 9–12 s | Cut | 0.000379 | 0.036678 | −0.002676 | 0.034381 |

The stress kick accounts for 0.153001 J of the 0.180007 J
of late cut-normal variance after including its cross term
with deposited motion. The post-stress and pre-gather values
agree to the shown precision. Gravity shifts the vertical
face velocities uniformly and the grid wall traction changes
only the tangent components. The measured stress contribution
combines the force distribution and the face mass denominator.

The instrumented native scene ran with
`--q-sand-symmetric-affine-decay 0 fine-half all 12`.
The angle reproduced 25.473317° and the command exited 1.
The temporary solver instrumentation was restored.

### Separate the original vertical stress force from floor reconstruction

On 43ec90c, divide the original force and the change made by
the floor ghost transpose by the same production face mass
denominator and multiply by the accepted duration. Center
both increments over each compatible vertical PIC stencil.
The entries are half the particle mass times the weighted
variance or twice the weighted covariance, summed by interval.

| Interval | Stencil | Original force (J) | Reconstruction change (J) | Cross term (J) | Combined kick (J) |
|---|---|---:|---:|---:|---:|
| 0–3 s | Full | 0.140204 | 0.038012 | −0.116181 | 0.062035 |
| 0–3 s | Cut | 0.113387 | 0.032541 | −0.105744 | 0.040184 |
| 3–6 s | Full | 0.051298 | 0.004051 | −0.017752 | 0.037597 |
| 3–6 s | Cut | 0.067697 | 0.010436 | −0.018587 | 0.059547 |
| 6–9 s | Full | 0.001718 | 0.001151 | −0.000843 | 0.002026 |
| 6–9 s | Cut | 0.054424 | 0.010719 | 0.028297 | 0.093439 |
| 9–12 s | Full | 0.001919 | 0.000778 | −0.000028 | 0.002669 |
| 9–12 s | Cut | 0.013062 | 0.008914 | 0.014702 | 0.036678 |

The original cut-normal force produces 0.135183 J of variance
from 3 to 12 s at the recorded states. The reconstruction
change and its cross term add a net 0.054482 J. Reconstruction
reduces the kick variance through 6 s and increases it afterward.
The measured kick on faces below the force mass floor contributes
zero variance at 1e-9 J precision. These are contributions at
the recorded states, so they leave the trajectory response to
a different force operator open.

The instrumented native run used
`--q-sand-symmetric-affine-decay 0 fine-half all 12`.
It reproduced 25.473317° and exited 1. The source was restored.

### Resolve the original vertical kick into normal and shear terms

On 43ec90c, split the original force on y-faces into the
normal-stress term −V₀ τyy ∂yN and the remaining shear terms
−V₀ (τyx ∂xN + τyz ∂zN). Center their velocity increments
and the floor reconstruction increment over each compatible
vertical PIC stencil using the same face mass denominator.

| Interval | Cut normal stress (J) | Cut shear stress (J) | Normal–shear cross (J) | Normal–reconstruction cross (J) | Shear–reconstruction cross (J) |
|---|---:|---:|---:|---:|---:|
| 0–3 s | 0.123776 | 0.003111 | −0.013500 | −0.108623 | 0.002879 |
| 3–6 s | 0.069032 | 0.002039 | −0.003373 | −0.016627 | −0.001959 |
| 6–9 s | 0.053589 | 0.001258 | −0.000424 | 0.029860 | −0.001563 |
| 9–12 s | 0.011375 | 0.000580 | 0.001107 | 0.016053 | −0.001351 |

From 3 to 12 s, normal stress alone accounts for 0.133996 J
of cut-stencil variance. Shear and its cross term with the
normal stress contribute a net 0.001186 J. The change in
sign of the reconstruction cross term is also concentrated
in its interaction with normal stress. This points to the
normal load discretization as the next operator to test.

Run the instrumented native scene with
`--q-sand-symmetric-affine-decay 0 fine-half all 12`.
The run reproduced 25.473317° and exited 1. The isolated
source was restored and its worktree was clean.

### Fit the reflected normal field through the stationary wall

On 43ec90c, fit the vertical velocity above the wall as a
linear function of height with zero intercept. Use donor
mass divided by one plus squared distance in grid units as
the least-squares weight. Evaluate the fit at the reflected
target height with the odd sign. Use the same coefficients
for continuation and the stress transpose. Keep tangential
rows and compatible PIC transport as in the baseline.

Using the fit only after the exact reflected donor shortcut
reproduces the baseline three-second matrix and stationary
probe at printed precision. Replacing the shortcut with the
fit gives the following results.

| Native case | Envelope at 3 s | Radius (m) |
|---|---:|---:|
| Coarse | 39.764264° | 0.165163 |
| Coarse half-step | 42.761533° | 0.152076 |
| Fine | 35.865342° | 0.157070 |
| Fine half-step | 37.296256° | 0.154708 |

The stationary strain probe passes all four refinements.
The absolute stress-work residual is at most 3.5e-16 J.
The native fine half-step passes the 12 s endpoint, reaching
32.725723° with radius 0.154708 m. It loses 18.42 mm of height
from 3 to 12 s, compared with 33.74 mm on the original
operator. The 6 s and 9 s angles are 34.970927° and
34.692032°. The replacement improves late retention and
pushes all four three-second angles above 35°.

Run the experimental native scene binary with
`--q-sand-single-particle-strain` and
`--q-sand-symmetric-affine-decay 0 <case> all <seconds>`.
The 12 s fine half-step exits 0 and the four three-second
cases with the shortcut replaced exit 1. The isolated
source was restored after each variant.

### Blend the wall-anchored fit with exact reflection

On 43ec90c, use the arithmetic mean of the exact reflected
donor row and the wall-anchored fit where the exact donor is
available. Keep the baseline fallback elsewhere. Use the
same blended row for the continuation and force transpose.

| Native case | Envelope at 3 s | Radius (m) |
|---|---:|---:|
| Coarse | 36.554786° | 0.167061 |
| Coarse half-step | 41.435573° | 0.154798 |
| Fine | 33.303573° | 0.159124 |
| Fine half-step | 35.837405° | 0.155348 |

The stationary strain probe passes all four refinements.
The stress-work residual is below 1.5e-16 J. The fine nominal
case passes at 3 s. The fine half-step misses the upper angle
bound at 3 s and reaches 28.284089° at 12 s, missing the lower
bound. Its height decreases by 28.60 mm from 3 to 12 s,
between the exact reflection and full fit results. A fixed
50% blend therefore does not resolve the two endpoint failures.

Run the isolated native scene with
`--q-sand-single-particle-strain` and
`--q-sand-symmetric-affine-decay 0 <case> all <seconds>`.
The temporary solver source was restored.

### Extend the wall-anchored fit across the sand matrix

Build the full wall-anchored fit on 43ec90c with the fit ahead
of the exact reflection shortcut. Extend the four native
column cases to 18 s with unchanged estimator, material
parameters and acceptance bounds.

| Case | Angle at 6 s | 9 s | 12 s | 15 s | 18 s | Radius at 18 s (m) |
|---|---:|---:|---:|---:|---:|---:|
| Coarse | 36.962° | 33.925° | 31.968° | 30.200° | 28.496° | 0.165163 |
| Coarse half-step | 38.680° | 37.246° | 36.358° | 34.979° | 33.922° | 0.162393 |
| Fine | 32.804° | 30.192° | 28.437° | 27.003° | 25.725° | 0.157070 |
| Fine half-step | 34.971° | 34.692° | 32.726° | 31.320° | 30.216° | 0.154708 |

Only the fine half-step passes at 18 s. It loses 2.510°
from 12 to 18 s despite drifting just 0.016752° during
the final 0.05 s. The coarse half-step meets the angle
band at 18 s and exceeds the radius bound by 2.39 mm.
The stress-work residual remains below 3.4e-16 J. The
fit improves retention while leaving long-horizon creep
and grid and timestep sensitivity unresolved.

Run the experimental native scene binary with
`--q-sand-symmetric-affine-decay 0 <case> all 18`.
The fine half-step exits 0 and the other cases exit 1.
The experimental source was restored before the runs.

### Check persistence of the wall-anchored fine half-step

Extend the diagnostic horizon parser to 24 s on 43ec90c
and rebuild the full wall-anchored fit. The material,
transport, angle estimator and acceptance bounds are unchanged.
Run the native diagnostic with
`--q-sand-symmetric-affine-decay 0 fine-half all 24`.

| Time | Envelope | Height (m) | Radius (m) |
|---|---:|---:|---:|
| 18 s | 30.215362° | 0.090097 | 0.154708 |
| 21 s | 29.268099° | 0.086705 | 0.154708 |
| 24 s | 28.418463° | 0.083715 | 0.154708 |

The run exits 1. The fit delays the lower-bound crossing
until after 18 s. The pile loses another 6.38 mm of height
between 18 and 24 s at fixed radius, despite drifting only
0.013509° over the final 0.05 s. The paired stress-work
residual is −8.30e-17 J. Both temporary sources were restored
before running the rebuilt executable.

### Refine the wall-anchored timestep through 24 seconds

Use the same native binary and fixture bounds as the preceding
24 s run. At the fine width of 0.0625 m, halve the timestep again
from 0.00078125 s to 0.000390625 s. Extend the coarse half-step
to the same horizon.

| Case | 3 s | 12 s | 18 s | 21 s | 24 s | Radius at 24 s (m) |
|---|---:|---:|---:|---:|---:|---:|
| Fine quarter-step | 45.340° | 35.073° | 34.196° | 31.457° | 29.786° | 0.155289 |
| Coarse half-step | 42.761° | 36.358° | 33.922° | 33.037° | 32.276° | 0.162393 |

Both cases exit 1. Fine timestep refinement delays the angle
crossing and leaves the angle below 30° by 24 s. The fine
quarter-step loses 16.63 mm of height from 18 to 24 s at fixed
radius, compared with 6.38 mm at the fine half-step. The
coarse half-step still exceeds the radius bound by 2.39 mm.
The absolute paired stress-work residual stays below 3.4e-16 J.

Run the experimental native binary from the preceding check
with `--q-sand-symmetric-affine-decay 0 fine-quarter all 24`
and repeat with `coarse-half`. The run uses the existing binary
and leaves the isolated source unchanged.

### Refine the timestep from the same formed wall-anchored pile

Use the full wall-anchored fit and the fine half-step through
3 s, then halve the timestep to 0.000390625 s. The pile matches
the constant-step run at 3 s in height, radius and angle.
Keep the material, operator and diagnostic bounds fixed.

| Time | Constant fine half-step | Quarter-step after 3 s |
|---|---:|---:|
| 3 s | 37.296256° | 37.296256° |
| 6 s | 34.970927° | 35.926382° |
| 12 s | 32.725580° | 34.527499° |
| 18 s | 30.215362° | 33.713488° |
| 24 s | 28.418463° | 31.969522° |

The switched run passes the 24 s endpoint at 31.969522°
and 0.154708 m radius. It still loses 6.67 mm of height
from 18 to 24 s, compared with 6.38 mm at constant timestep.
The finer late timestep improves retention from the same
formed pile and leaves measurable creep. The absolute
paired stress-work residual is 8.39e-17 J or less.

Run the same experimental native binary with
`--q-sand-symmetric-affine-decay 0 fine-half all 24 half`.
The isolated tracked sources are clean.

### Compare three late timesteps from one wall-anchored pile

Use the same native wall-anchored binary and the same formed
pile at 3 s. Double the late step and compare with the
constant and halved runs. The radius is 0.154708 m in all
three runs from 3 to 24 s.

| Late timestep (ms) | Envelope at 24 s | Height loss from 3 to 24 s (mm) |
|---:|---:|---:|
| 0.390625 | 31.969522° | 21.282 |
| 0.781250 | 28.418463° | 34.125 |
| 1.562500 | 22.311591° | 54.353 |

Each doubling increases cumulative height loss by about 60%
over the same elapsed time. The larger step takes fewer
updates and produces more creep, ruling out a constant loss
per update for these runs. The result alone leaves the
small-timestep limit undetermined.

Run `--q-sand-symmetric-affine-decay 0 fine-half all 24 double`
with the existing experimental executable. It exits 1 on
the angle bound and gives a stress-work residual of −1.03e-16 J.

### Alternate late timesteps with the wall-anchored fit

Keep the native fine half-step through 3 s. Alternate between
0.390625 and 1.171875 ms afterward, in both orders. Each pair
has the same duration as two constant 0.781250 ms steps.

| Late schedule | Angle at 12 s | Angle at 18 s | Angle at 24 s | Height loss from 3 to 24 s (mm) |
|---|---:|---:|---:|---:|
| Constant | 32.725580° | 30.215362° | 28.418463° | 34.125 |
| Alternating | 31.801372° | 29.195923° | 27.268000° | 38.098 |
| Alternating reversed | 31.801199° | 29.195780° | 27.267835° | 38.099 |

The alternating runs lose about 12% more height than the
constant run. Reversing the order changes the final angle
by 0.000165°. Their common mean timestep therefore hides
an effect of the step-size variation. The radius remains
0.154708 m and the absolute stress-work residual stays
below 8.5e-17 J.

Use the existing wall-anchored native diagnostic with
`--q-sand-symmetric-affine-decay 0 fine-half all 24 alternating`
and repeat with `alternating-reversed`. Both exit 1.

### Sample the vertical stress kick at midpoint for deformation

On 43ec90c with exact reflection, subtract half the vertical
stress kick from the deformation donor field. Pass this
offset through the same ghost rows before applying the
quadratic derivative. Retain the final grid field for
transport, stored APIC velocity and wall traction. Evaluate
the vertical contribution to the stress-work receipt at
the same field used for deformation.

| Time | Original fine half-step | Vertical midpoint |
|---|---:|---:|
| 3 s | 34.693822° | 35.061270° |
| 6 s | 30.869627° | 33.660867° |
| 9 s | 27.988015° | 30.608857° |
| 12 s | 25.473317° | 29.672977° |

The radius is 0.159789 m from 3 s onward. Height loss from
3 to 12 s falls from 33.74 mm to 21.10 mm. The paired work
residual is 1.58e-16 J. The native run exits 1 on the angle
bound. This isolates vertical stress time centering while
leaving horizontal deformation and all advection at their
original time levels.

Run the isolated native diagnostic with
`--q-sand-symmetric-affine-decay 0 fine-half all 12`.
The temporary solver source was restored.

### Check vertical midpoint sampling across the native sand matrix

Use the vertical midpoint diagnostic above with exact
reflection and unchanged transport. Run each case to 18 s.

| Case | Angle at 3 s | 12 s | 18 s | Radius at 18 s (m) |
|---|---:|---:|---:|---:|
| Coarse | 36.127° | 30.163° | 25.221° | 0.196341 |
| Coarse half-step | 41.216° | 33.767° | 33.161° | 0.168535 |
| Fine | 33.888° | 27.779° | 27.756° | 0.167741 |
| Fine half-step | 35.061° | 29.673° | 28.982° | 0.159789 |

All four exit 1. The coarse half-step changes by only 0.015°
from 15 to 18 s, but its radius exceeds 0.16 m. The fine
half-step stays inside the radius bound and falls below 30°.
The paired work residual is at most 6.2e-16 J. The stationary
strain probe passes all four refinements with exact zero
position and velocity. Its final elastic yy stretch at the
1 ms timestep is 0.949272811491, versus 0.949494657882 in the
original probe. The midpoint sample preserves the existing
refinement bound and leaves a larger strain error at 1 ms.

Run the experimental native executable with
`--q-sand-symmetric-affine-decay 0 <case> all 18` and
`--q-sand-single-particle-strain`. The isolated source is clean.

### Center gravity with the vertical stress kick

On 43ec90c, subtract half the vertical stress and gravity
increments from the final deformation donor field. Apply
the existing ghost rows to the offset and use the resulting
vertical deformation field in the stress-work receipt.
PIC transport, affine storage and contact use the final field.

| Time | Vertical stress midpoint | Vertical stress and gravity midpoint |
|---|---:|---:|
| 3 s | 35.061270° | 34.665918° |
| 6 s | 33.660867° | 31.284990° |
| 9 s | 30.608857° | 28.930263° |
| 12 s | 29.672977° | 26.392718° |

The fine half-step loses 30.83 mm in height from 3 to 12 s.
The radius stays at 0.157836 m. Centering gravity removes
most of the retention gain from centering stress alone.
The stationary strain probe passes all four refinements.
Its 1 ms final elastic yy stretch is 0.949746986210,
closer to the initial 0.95 than either earlier variant.
The cut stress-work residual is −6.74e-17 J.

Run the isolated native binary with
`--q-sand-single-particle-strain` and
`--q-sand-symmetric-affine-decay 0 fine-half all 12`.
The first exits 0 and the second exits 1. The source was restored.

### Halve the late timestep with vertical midpoint sampling

Use the existing paired vertical midpoint executable, with
the fine half-step through 3 s and a quarter-step afterward.
The two runs have identical recorded height and radius at
3 s. Neither the material nor the stress sampling changes.

| Time | Constant fine half-step | Quarter-step after 3 s |
|---|---:|---:|
| 3 s | 35.061270° | 35.061270° |
| 6 s | 33.660867° | 34.175474° |
| 12 s | 29.672579° | 32.152035° |
| 18 s | 28.982269° | 29.958344° |

The smaller late step improves the 18 s angle by 0.976° and
still falls below 30°. The radius stays at 0.159789 m. The
height drops by 8.34 mm from 12 to 18 s with late refinement.
The stress-work residual is 1.62e-16 J. The native diagnostic
exits 1 with
`--q-sand-symmetric-affine-decay 0 fine-half all 18 half`.

### Combine wall-anchored reconstruction and vertical stress midpoint

On 43ec90c, use the full wall-anchored reflected row in the
continuation and its force transpose. Apply the vertical
stress midpoint offset through those same rows for deformation
and its work receipt. PIC, APIC storage and wall contact use
the final field as before.

| Native case | Angle at 3 s | 12 s | 18 s | Radius at 18 s (m) |
|---|---:|---:|---:|---:|
| Coarse | 41.384° | 36.064° | 29.343° | 0.212261 |
| Coarse half-step | 43.809° | 39.459° | 39.382° | 0.166836 |
| Fine | 38.269° | 32.465° | 28.122° | 0.176331 |
| Fine half-step | 38.167° | 34.594° | 33.900° | 0.157910 |

The fine half-step exits 0. It loses 0.50 mm in height from
15 to 18 s, compared with 4.04 mm using the wall fit alone.
The other three cases exit 1. The coarse half-step changes
by less than 0.10° from 12 to 18 s and exceeds both the
angle and radius bounds. Both nominal-step cases spread
late and end below 30°. The stationary strain probe passes
all four refinements and the maximum absolute paired work
residual is 1.61e-16 J.

Run the experimental native binary with
`--q-sand-single-particle-strain` and
`--q-sand-symmetric-affine-decay 0 <case> all 18`.
Both temporary solver edits were restored before the runs.

### Refine the combined wall fit and midpoint operator

Use the combined native executable above at width 0.0625 m.
Compare the 0.00078125 s step with 0.000390625 s applied
either at release or after the matched 3 s formed state.

| Schedule | Angle at 3 s | 12 s | 18 s | Radius at 18 s (m) |
|---|---:|---:|---:|---:|
| Constant half-step | 38.167104° | 34.593606° | 33.900254° | 0.157910 |
| Quarter-step after 3 s | 38.167104° | 35.985957° | 34.804493° | 0.157910 |
| Constant quarter-step | 45.759652° | 37.208375° | 35.442095° | 0.157809 |

The late-refined run passes the 18 s endpoint and loses
1.51 mm of height from 15 to 18 s, compared with 0.50 mm
at constant half-step. The fully refined run exceeds 35°
at 18 s and loses 2.92 mm in the same interval. The apparent
plateau in the half-step is sensitive to timestep and release
history. Both new runs preserve the stress-work receipt
within 1.9e-16 J.

Run `--q-sand-symmetric-affine-decay 0 fine-half all 18 half`
and `--q-sand-symmetric-affine-decay 0 fine-quarter all 18`.
The first exits 0 and the second exits 1.

### Check independent surface slopes with the combined operator

On 43ec90c, rebuild the combined wall fit and stress midpoint
with the independent annular surface fit enabled for both
particle samplings. Use the same native fine half-step through
18 s. The dense case doubles sampling per axis while keeping
grid width, timestep, release geometry and material parameters.

| Sampling | Envelope | Radius (m) | Signed quadrant surface slopes |
|---|---:|---:|---:|
| Original | 33.900254° | 0.157910 | −34.225006° to 0.000000° |
| Doubled per axis | 26.312249° | 0.245209 | 8.302677° to 11.160215° |

All four quadrant fits resolve at each sampling. Negative
angles mean the fitted heights rise outward. The original
sampling passes the envelope bounds and has no quadrant
surface slope in the target range. The denser release spreads
farther, fails the envelope and radius bounds, and has
shallow outward-descending slopes. The paired work residuals
are below 1.5e-16 J. The passing envelope is sensitive to
particle sampling and leaves the free-surface angle unqualified.

Run the experimental native binary with
`--q-sand-symmetric-affine-decay 0 fine-half all 18`
and with `fine-half-dense` for the temporary denser selection.
The two exits are 0 and 1 respectively. Both solver files
and the fixture source were restored before the runs.

### Measure bulk radial spread in the denser combined run

Repeat the doubled-per-axis native fine half-step with the
combined wall fit and stress midpoint through 18 s. At each
3 s boundary, sort the 504 particle radii and report the
lower-index empirical quantiles. Sum the mass beyond the
0.16 m fixture radius separately.

| Time | Median radius (m) | 90th percentile (m) | 95th percentile (m) | Mass outside 0.16 m |
|---|---:|---:|---:|---:|
| 3 s | 0.096799 | 0.161239 | 0.171733 | 10.52% |
| 6 s | 0.105113 | 0.172103 | 0.182525 | 16.87% |
| 9 s | 0.114188 | 0.178746 | 0.187972 | 23.21% |
| 12 s | 0.118268 | 0.186816 | 0.197432 | 27.38% |
| 15 s | 0.124559 | 0.195189 | 0.208446 | 30.56% |
| 18 s | 0.132646 | 0.208789 | 0.224442 | 34.72% |

The median moves outward by 35.85 mm and more than a third of
the mass ends beyond the radius bound. This is bulk spread.
The radial trace reproduces the 26.312249° envelope and exits 1.
Run the instrumented native fixture with
`--q-sand-symmetric-affine-decay 0 fine-half-dense all 18`.

### Recheck the floor candidate from PR 111 on merged main

Cherry-pick the isolated floor change 98e2107 onto e8c4a45
and build Windows Release native and SM120 dry scene tests.
Both default dry scene tests pass. Both backends reproduce
the same failure pattern in the explicit sand commands.

| Command | Native angles in fixture order | Exit on both backends |
|---|---|---:|
| --q-sand-refined | 30.052977° | 0 |
| --q-sand-half-step | 37.729429°, 33.013884° | 1 |
| --q-sand-column-long-settle | 29.113507°, 26.761332°, 29.006384°, 27.033753° | 1 |

The CUDA refined angle differs by 0.000001°. The standalone
candidate commit applies cleanly. The stacked PR includes
earlier history and conflicts with merged main. The candidate
was reverted on the evidence branch after qualification.

### Run the sand settlement gate in CTest

Configure the existing native or CUDA build with
`-DFLUIDITY_ENABLE_Q_SAND=ON` and build the dry-scene targets.
Run `ctest --test-dir out/build -R '^QSand\.' --no-tests=error --output-on-failure`.
Each test runs the original-density 1.5 s column at both grid
widths and timesteps with the existing physical acceptance
checks. Independent surface fits and longer experimental
runs are separate diagnostics.

On merged main 5b175ec, both default dry-scene tests pass.
Both QSand tests fail with the same final envelope angles
to six decimal places, in fixture order: 29.113507°,
26.761332°, 29.006384°, and 27.033753°. The option is off
by default while this gate remains open.

### Isolate the swept deformation gradient on merged main

On 72cdcda, run the opt-in Q-SAND matrix with two temporary
post-impact deformation controls. In the first, remove the
normal velocity gradient row for incoming motion and retain
the tangential rows. In the second, use the original Coulomb
gradient projection and restore only the normal derivative
of normal velocity. Leave particle advection and wall impulse
unchanged in both controls.

| Operator | Coarse | Fine | Coarse half-step | Fine half-step |
|---|---:|---:|---:|---:|
| Baseline | 29.113507° | 26.761332° | 29.006384° | 27.033753° |
| Normal-only projection | 22.283879° | 22.241051° | 28.477639° | 27.128322° |
| Restore normal derivative | 24.691677° | 23.755083° | 30.375928° | 29.212625° |

Native and SM120 agree to within 0.000001°. Both controls pass
the default dry-scene tests and fail the Q-SAND tests. Removing
tangential projection worsens both nominal-step angles.
Restoring the normal derivative raises both half-step angles
and lowers both nominal-step angles. This leaves timestep
sensitivity within the swept deformation treatment.

Run each variant with `FLUIDITY_ENABLE_Q_SAND=ON` and
`ctest --test-dir out/build -R '^(QSand|M5DryScenes)\.' --output-on-failure`.
The temporary solver source was restored after each build.

### Separate first impact and sustained floor contact

On 72cdcda, restrict the existing swept deformation split to
particles starting above the floor in the first control, then
to particles starting on the floor in the second. Keep the
position sweep and Coulomb impulse in both controls.

| Deformation split | Coarse | Fine | Coarse half-step | Fine half-step |
|---|---:|---:|---:|---:|
| First impact only | 20.804356° | 20.578893° | 28.040286° | 26.585617° |
| Sustained contact only | 29.004946° | 26.721486° | 29.004619° | 27.031849° |

The sustained-contact control is within 0.11° of baseline in
all four cases. Restricting the split to first impact lowers
both nominal-step angles by more than 6°. The baseline effect
therefore depends on treating repeated floor contact. Both
controls pass default dry-scene tests and fail the opt-in
Q-SAND matrix on native and SM120, with cross-backend angle
agreement within 0.000001°. The solver source was restored.

### Preserve the Coulomb-projected affine fit after contact

On 72cdcda, apply the existing swept Coulomb gradient map
to the stored APIC fit when a granular particle hits the
stationary floor. Keep the baseline deformation and sweep.
Run the native and SM120 Q-SAND and default dry-scene tests.

| Case | Baseline | Projected affine storage |
|---|---:|---:|
| Coarse | 29.113507° | 29.083229° |
| Fine | 26.761332° | 26.790663° |
| Coarse half-step | 29.006384° | 28.883736° |
| Fine half-step | 27.033753° | 26.915365° |

Both backends pass the default dry-scene tests and fail the
four-case Q-SAND matrix. The largest angle change is 0.123°.
This rules out clearing the post-contact affine fit as the
dominant effect in the tested merged solver. The temporary
solver edit was restored.

### Separate accumulated granular plastic volume from the elastic return

On 72cdcda, temporarily reset the granular log plastic Jacobian
to zero after each accepted update. Keep the returned elastic
deformation and all solver operations. This control deliberately
breaks total volume bookkeeping and tests feedback from the
stored history only.

All four native and SM120 Q-SAND cases reproduce the baseline
angles, heights, radii and kinetic energies at printed precision
with mean log plastic Jacobian zero. Both gate invocations fail.
The result rules out feedback from accumulated plastic volume in
this fixture. It leaves the effect of the elastic return to the
tensile apex open. The temporary solver source was restored.

### Locate late tensile-apex returns in the merged sand solver

On 72cdcda, classify accepted granular updates by whether the
sweep hits the floor and whether the particle starts at the
floor. Count a tensile-apex return when the log plastic
Jacobian changes beyond the existing material-state tolerance.
Aggregate after the 0.75 s sample through 1.5 s.

| Case | No floor hit | Arriving floor hit | Sustained floor hit | Share of apex returns with no hit |
|---|---:|---:|---:|---:|
| Coarse | 10468 / 11535 | 13 / 13 | 0 / 13715 | 99.88% |
| Fine | 11081 / 12436 | 20 / 20 | 0 / 17784 | 99.82% |
| Coarse half-step | 9226 / 32265 | 0 / 12 | 0 / 18186 | 100% |
| Fine half-step | 12668 / 36439 | 8 / 13 | 0 / 24028 | 99.94% |

The native run reproduced all four baseline envelope angles
and exited 1. Repeated floor hits contribute no measured
late tensile-apex events. The next useful control is to
resolve the no-hit events by height and measure whether
their stress reset changes load support near the floor.
The temporary solver trace was restored.

### Measure incoming compression of late tensile-apex particles

On 72cdcda, split the accepted native return trace by starting
height. Sum the positive part of compressive Kirchhoff stress
in the vertical direction, weighted by particle reference
volume, for all updates and for apex events alone. This proxy
measures incoming stress carried by those particles before
the return. Aggregate after the 0.75 s sample through 1.5 s.

| Case | Apex at floor | Apex above floor through h/2 | Apex above h/2 | Share of incoming compression on apex events |
|---|---:|---:|---:|---:|
| Coarse | 0 | 7602 | 2879 | 0.000267% |
| Fine | 0 | 8733 | 2368 | 0.000143% |
| Coarse half-step | 0 | 5100 | 4126 | 0.0000244% |
| Fine half-step | 0 | 10024 | 2652 | 0.0000348% |

Most returns occur below half a cell above the floor. The
particles undergoing them carry less than 0.0003% of the
incoming compression summed over the window in every case.
This bounds direct removal of incoming compressive stress
at the return events. The native fixture reproduces the
baseline angles and exits 1. The temporary trace was restored.

### Compare the compatible-floor path on both backends

On 72cdcda, enable the existing compatible floor path in the
four Q-SAND column cases. Keep the 1.5 s horizon, material,
sampling and gate criteria. Native and SM120 agree at the
reported precision.

| Case | Final angle | Integrated cut stress-work residual (J) |
|---|---:|---:|
| Coarse | 29.962967° | −0.097516 |
| Fine | 27.424648° | −0.111296 |
| Coarse half-step | 32.042153° | 0.020711 |
| Fine half-step | 29.277572° | −0.026814 |

Only the coarse half-step reaches the angle band. Both CTest
entries fail. The compatible stress scatter uses normalized
weight derivatives while deformation gathers the complete
ghost field. The measured work residual motivates testing a
deformation gather paired with the normalized scatter.
The temporary fixture edit was restored.

### Pair normalized stress scatter with the deformation gather

On d27b5b4, enable the compatible floor in all four native
Q-SAND cases. Gather the deformation gradient using the
normalized weight derivatives already used by the stress
scatter. Keep the complete ghost gather for the APIC fit
and leave PIC translation and the swept contact unchanged.

| Case | Complete ghost gradient | Paired normalized gradient |
|---|---:|---:|
| Coarse | 29.962967° | 28.407709° |
| Fine | 27.424648° | 26.082740° |
| Coarse half-step | 32.042153° | 31.915729° |
| Fine half-step | 29.277572° | 29.126156° |

The integrated work residual drops from at most 0.111296 J
to below 7.3e-17 J. All four angles decrease and the matrix
still fails. The native default dry scene passes. This
isolates work consistency from angle retention for the
normalized derivative. Both temporary edits were restored.

### Pair the complete floor continuation with its stress transpose

On 4300d54, port the weighted floor ghost rows from 43ec90c
as a temporary native control. Scatter complete quadratic
stress through the row transpose and gather the complete
deformation gradient from the same rows. Keep the merged
solver's swept deformation and compatible PIC translation.

| Case | Normalized derivative | Complete ghost transpose |
|---|---:|---:|
| Coarse | 28.407709° | 36.571036° |
| Fine | 26.082740° | 32.610886° |
| Coarse half-step | 31.915729° | 37.590067° |
| Fine half-step | 29.126156° | 35.079876° |

The integrated cut stress-work residual stays below 3e-16 J.
All four angles increase, and only the fine nominal case
is within the band. The default native dry-scene executable
also exits 1 before printing scene results. The experimental
sources were restored. The control has no CUDA result.

#### Identify the default fixture regression

Run all nine default native checks individually with the
same complete ghost transpose. Only the compatible
single-particle wall impulse fixture fails. It returns OK
with one hit, a −0.5 N·s tangential impulse, a 1 N·s normal
impulse, −0.875 J of particle wall kinetic work and zero grid
wall impulse, cut stress work and plastic volume events.
Those fields match the fixture. Its granular projection
count is one, versus the fixture requirement of zero.
The ordinary floor variant reports zero projections.
The failure is in the deformation path of a stress-free
initial particle. The impulse accounting still passes.
Temporary diagnostics and solver edits were restored.

#### Use the complete continuation for the full deformation step

On 4300d54, keep the complete ghost transpose control and
skip the additional swept deformation split on that path.
Keep the swept position update, wall impulse and return
mapping. The native default dry-scene test passes, including
the compatible wall oracle that failed with both corrections.

| Case | With swept split | Without swept split |
|---|---:|---:|
| Coarse | 36.571036° | 34.892353° |
| Fine | 32.610886° | 29.430121° |
| Coarse half-step | 37.590067° | 37.222976° |
| Fine half-step | 35.079876° | 34.001150° |

The native matrix still exits 1. The absolute cut stress-work
residual stays below 4.2e-16 J. Removing the second deformation
correction resolves the single-particle regression and leaves
the sand result sensitive to grid width and timestep.
The temporary solver and fixture edits were restored.

### Compare paired blends of normalized and complete derivatives

On 5fd3ae8, blend the normalized derivative with the complete
ghost derivative, applying the same weights to the stress
transpose. Use the blended gradient for the entire deformation
step and preserve the swept position and impulse update.

| Case | Half complete | Three quarters complete |
|---|---:|---:|
| Coarse | 29.541652° | 32.867041° |
| Fine | 28.543189° | 29.961114° |
| Coarse half-step | 33.852203° | 35.974314° |
| Fine half-step | 31.796255° | 33.619922° |

Both variants pass the default native dry-scene test.
The half blend passes both half-step cases and misses both
nominal cases. The three-quarter blend passes the coarse
nominal case, misses the fine nominal and coarse half-step
angle bounds, and exceeds the fine half-step radius bound
by 1.60 mm. The absolute work residual is below 3.2e-16 J.
The temporary edits were restored. Neither blend was tested
on CUDA.

### Refine both grid widths with the half-complete derivative

On d220fa4, rerun the equal paired blend with quarter and
eighth nominal timesteps. Keep the 1.5 s measurement horizon
and original particle sampling. The prior half-step is
included for comparison.

| Grid width (m) | Half-step | Quarter-step | Eighth-step |
|---|---:|---:|---:|
| 0.075 | 33.852203° | 39.880769° | 49.489569° |
| 0.0625 | 31.796255° | 35.920061° | 44.069917° |

All four new native cases exceed 35°. The difference between
grid widths grows from 2.06° to 5.42° over this step sequence.
The eighth-step radius shrinks to 0.112709 m on the coarse grid
and 0.131221 m on the fine grid. The integrated cut work
residual stays below 2.4e-16 J. This finite-time result leaves
the smaller-step settlement behavior unqualified. The
temporary sources were restored. CUDA was untested.

### Extend the refined paired blend through three seconds

On 64db21c, rerun the equal paired blend with the same quarter
and eighth timesteps and a 3 s endpoint. Keep the material,
particle sampling and acceptance criteria.

| Case | Angle at 1.5 s | Angle at 3 s | Radius at 3 s (m) |
|---|---:|---:|---:|
| Coarse quarter-step | 39.880769° | 33.384357° | 0.142157 |
| Fine quarter-step | 35.920061° | 31.251002° | 0.163865 |
| Coarse eighth-step | 49.489569° | 39.228247° | 0.128871 |
| Fine eighth-step | 44.069917° | 35.200336° | 0.151239 |

Angles fall by 4.67° to 10.26° after the original endpoint.
Only the coarse quarter-step satisfies both angle and radius
bounds at 3 s. The fine quarter-step enters the angle band
and spreads beyond the radius limit. Both eighth-steps remain
above 35°. The maximum cut stress-work residual is 2.34e-16 J.
The native CTest matrix fails. The temporary solver and test
edits were restored. CUDA was untested for this extension.

### Measure wall friction across the refined paired blend

On 0f9e7cb, keep the equal paired blend at the fine grid width
of 0.0625 m. Sum individual magnitudes of tangential wall
impulse for contacts starting at the floor from 0.75 to 1.5 s.
Use the sum of their normal impulses as the denominator.
Count impulses within 1e-9 relative of the Coulomb cap.

| Step (ms) | Tangential impulse (N·s) | Normal impulse (N·s) | Ratio | At cap / contacts |
|---|---:|---:|---:|---:|
| 1.562500 | 2.141890 | 14.449638 | 0.1482 | 0 / 15360 |
| 0.781250 | 3.291302 | 13.550246 | 0.2429 | 0 / 19123 |
| 0.390625 | 5.038844 | 13.175313 | 0.3824 | 0 / 22132 |
| 0.1953125 | 7.088584 | 13.189027 | 0.5375 | 1752 / 34560 |

Finer steps increase tangential impulse at sustained contact
while the normal impulse stays within 9% of the nominal run.
The ratio approaches the 0.7 cap and the eighth-step reaches
it in 5.1% of contacts. The native diagnostic reproduces the
previously measured angles and exits 1. The trace measures
association with timestep and does not isolate a causal wall
friction change. The temporary solver and fixture were restored.

### Isolate floor friction in the refined paired blend

On cb16ffd, keep the equal paired derivative blend and the
normal floor sweep. Set only the floor friction coefficient
to zero in the native fine grid runs through 1.5 s.

| Step (ms) | Angle with friction 0.7 | Angle with friction 0 | Frictionless radius (m) |
|---|---:|---:|---:|
| 1.562500 | 28.543189° | 6.568419° | 0.181194 |
| 0.781250 | 31.796255° | 14.764613° | 0.172055 |
| 0.390625 | 35.920061° | 24.161425° | 0.158571 |
| 0.1953125 | 44.069917° | 36.547119° | 0.136994 |

Removing friction reduces the angle at every timestep.
The angle span across timesteps increases from 15.53° to
29.98°, so the trend persists without tangential wall
impulse. The absolute work residual stays below 2.1e-16 J.
The native diagnostic exits 1. Temporary sources were
restored and CUDA was untested for this control.

### Use deposited mass for the paired stress acceleration

On 2bc9d8b, remove the acceleration mass floor on the equal
paired compatible-floor blend. Keep the ordinary floor path,
force scatter, deformation gather and acceptance checks.

| Case | Regularized acceleration | Actual deposited mass |
|---|---:|---:|
| Coarse | 29.541652° | 29.539514° |
| Fine | 28.543189° | Face motion bound at step 460 |
| Coarse half-step | 33.852203° | 33.847722° |
| Fine half-step | 31.796255° | 31.796472° |

All three completed cases move by less than 0.005°. The
fine nominal run rejects before reaching its endpoint.
The default native dry-scene test passes. The Q-SAND
matrix fails and the absolute cut work residual remains
below 3e-16 J on completed runs. The temporary sources
were restored. CUDA was untested for this control.

#### Retry the fine nominal face-motion rejection

On the same unregularized paired blend, retry only the
duration-bound failures, halving the attempted step and
resuming the nominal schedule after each accepted retry.
The fine nominal run needed three halvings across two
steps near 0.719 s. It reached 1.5 s with an envelope of
28.514871° and radius 0.154059 m. The regularized run
reached 28.543189° and 0.154080 m. The other three native
cases reproduced the preceding control. The face-motion
rejection was recoverable and the angle gate still failed.
The temporary solver and fixture edits were restored.

#### Measure the face that triggers the motion retry

On 6104805, repeat the unregularized native run and log the
first face-motion rejection. At step 460, the vertical face
at (−0.03125, 0.18750, −0.03125) m carries 5.923e-10 kg, or
1.579e-8 of one particle mass. Its stress force is 9.213 µN.
The deposited velocity is −0.07985 m/s. Stress and gravity
take it to 24.20842 m/s, above the 18 m/s bound. Substituting
the existing mass floor into that same update gives
−0.09134 m/s. The face lies three fine cells above the floor.
This locates the retry in thin support above the pile.
The other three angles reproduce the preceding control.
Temporary sources were restored.

### Limit thin-face stress acceleration by the motion bound

On 6104805, use actual deposited mass for granular faces
below the existing acceleration mass floor if the gravity
and deposited motion stay inside 99% of the face speed
limit. Bound the stress increment to the remaining speed
interval and use the resulting effective mass in the work
and omitted impulse receipts. Run the ordinary native
Q-SAND matrix.

| Case | Existing mass floor | Motion-limited stress |
|---|---:|---:|
| Coarse | 29.113507° | 29.113649° |
| Fine | 26.761332° | 26.754162° |
| Coarse half-step | 29.006384° | 29.002684° |
| Fine half-step | 27.033753° | 27.032133° |

All cases reach the endpoint. Reported omitted stress impulse
is at most 2e-9 N·s and the angles move by less than 0.008°.
The default native dry-scene test passes. Q-SAND still fails.
The temporary source was restored. CUDA was untested.

### Integrate the merged deformation step to second order

On 6104805, add the quadratic term (dt²/2) L²F to each
frozen-gradient increment, including both intervals of the
swept floor split. Keep the grid velocities and constitutive
return timing unchanged.

| Case | Euler | Quadratic |
|---|---:|---:|
| Coarse | 29.113507° | 29.102928° |
| Fine | 26.761332° | 26.847840° |
| Coarse half-step | 29.006384° | 29.009818° |
| Fine half-step | 27.033753° | 27.045365° |

The largest change is 0.087°. The default native dry-scene
test passes and Q-SAND fails. The integration order at a
fixed velocity gradient has little effect on the measured
angles. The temporary source was restored. CUDA was untested.

### Center the ordinary-floor deformation gradient in time

On 6104805, gather the ordinary granular gradient from
the pre-force field and the final grid field. Use their
arithmetic mean for both intervals of the existing swept
deformation split. Transport and affine storage still use
the final field. Snow and the compatible-floor path keep
their original deformation update.

| Case | Final-field gradient | Midpoint gradient |
|---|---:|---:|
| Coarse | 29.113507° | 31.045043° |
| Fine | 26.761332° | 27.864590° |
| Coarse half-step | 29.006384° | 29.712519° |
| Fine half-step | 27.033753° | 27.634628° |

All four angles improve. The coarse nominal case passes.
The other angles remain below 30°, and the fine half-step
also exceeds the radius bound by 0.247 mm. The default
native dry-scene test passes and the Q-SAND matrix fails.
The temporary solver edit was restored. CUDA was untested.

#### Separate vertical and horizontal midpoint gradients

Repeat the ordinary-floor midpoint test on 6104805, averaging
only the vertical row in the first variant and only the two
horizontal rows in the second. Keep both sweep intervals.

| Case | Baseline | Vertical midpoint | Horizontal midpoint |
|---|---:|---:|---:|
| Coarse | 29.113507° | 30.494929° | 29.719217° |
| Fine | 26.761332° | 27.783781° | 26.962533° |
| Coarse half-step | 29.006384° | 29.710618° | 29.015467° |
| Fine half-step | 27.033753° | 27.652184° | 27.019219° |

The vertical row produces most of the angle increase seen
with all rows centered. Only the coarse case passes with
vertical centering. Both variants pass the default native
dry-scene test and fail Q-SAND. The temporary sources were
restored. CUDA was untested for these controls.

#### Center only the vertical normal stress increment

On 6104805, assemble the vertical force from the yy stress
component separately. Subtract half its velocity increment
from the final vertical field for the ordinary granular
deformation gradient. Keep shear, gravity, transport, affine
storage and contact at their previous time levels.

| Case | Full vertical midpoint | yy stress midpoint |
|---|---:|---:|
| Coarse | 30.494929° | 30.499204° |
| Fine | 27.783781° | 27.761428° |
| Coarse half-step | 29.710618° | 29.729921° |
| Fine half-step | 27.652184° | 27.633602° |

The two results agree within 0.023° across the native matrix.
The default dry-scene test passes. The coarse case reaches
the angle band and the other three remain below it. The
normal stress increment accounts for nearly all the effect
of centering the vertical field in this control. Temporary
sources were restored. CUDA was untested.

#### Sample deformation before the normal stress kick

On 04b21df, subtract the entire yy stress velocity increment
from the deformation sample. Keep the complete kick for
transport, affine storage and contact as in the midpoint test.

| Case | Normal midpoint | Before normal kick |
|---|---:|---:|
| Coarse | 30.499204° | 32.330788° |
| Fine | 27.761428° | 28.336343° |
| Coarse half-step | 29.729921° | 30.660133° |
| Fine half-step | 27.633602° | 28.179787° |

Both coarse cases pass at 1.5 s. Both fine cases remain below
30°, and the fine half-step exceeds the radius limit by
0.818 mm. The default native dry-scene test passes and
Q-SAND fails. Earlier sampling improves all four angles
and leaves the grid-width discrepancy. Temporary source
was restored. CUDA was untested for this control.

#### Use the pre-kick vertical field for translation as well

On 04b21df, additionally subtract the yy stress increment
from the PIC translation and its grid velocity increment.
The wall sweep receives that translation and the stored
affine fit still comes from the final field.

| Case | Pre-kick deformation only | Pre-kick deformation and translation |
|---|---:|---:|
| Coarse | 32.330788° | 21.657478° |
| Fine | 28.336343° | 19.671279° |
| Coarse half-step | 30.660133° | 23.278828° |
| Fine half-step | 28.179787° | 21.037452° |

The default native dry-scene and Q-SAND tests fail. Removing
the normal stress kick from the particle translation also
removes its impulse from the advancing particle velocity.
This control shows that the previous improvement depends on
keeping that impulse in transport. The temporary source was
restored. CUDA was untested.

#### Center displacement while retaining the stored velocity kick

On 04b21df, average the pre-force and final PIC velocities
for the ordinary granular wall sweep path. Keep the final
PIC velocity as the stored velocity entering contact and
keep the original deformation gradient. The wall sweep
therefore sees a different path and impact time.

| Case | Baseline | Midpoint displacement |
|---|---:|---:|
| Coarse | 29.113507° | 27.032886° |
| Fine | 26.761332° | 24.422122° |
| Coarse half-step | 29.006384° | 28.555911° |
| Fine half-step | 27.033753° | 26.757779° |

The default native dry-scene test passes. All four sand
angles fall and Q-SAND still fails. This control isolates
the time level of advection and the associated contact
timing from the stored velocity update. The temporary
source was restored. CUDA was untested.

#### Isolate the vertical midpoint displacement

On 146f854, center the vertical displacement alone or the
two horizontal components alone in the ordinary granular
sweep. Keep final stored velocity and the final deformation
field in both variants.

| Case | Baseline | Vertical midpoint displacement | Horizontal midpoint displacement |
|---|---:|---:|---:|
| Coarse | 29.113507° | 26.931189° | 29.211256° |
| Fine | 26.761332° | 24.302544° | 26.932741° |
| Coarse half-step | 29.006384° | 28.443352° | 29.138165° |
| Fine half-step | 27.033753° | 26.713218° | 27.096094° |

The vertical change accounts for the angle reduction in the
full midpoint displacement control. Centering the horizontal
components increases angles by at most 0.172°. Both native
variants pass the default dry-scene test and fail Q-SAND.
The temporary source was restored. CUDA was untested.

#### Separate above-floor and floor-start vertical displacement

On 146f854, apply the vertical midpoint displacement only to
updates starting above the floor, then only to updates starting
on the floor. Keep final stored velocity and deformation.

| Case | Above floor only | At floor only |
|---|---:|---:|
| Coarse | 26.961980° | 29.048986° |
| Fine | 24.308982° | 26.731736° |
| Coarse half-step | 28.442358° | 29.013248° |
| Fine half-step | 26.731460° | 27.028059° |

The above-floor variant reproduces the full vertical midpoint
effect within 0.031°. The floor-start variant is within 0.065°
of baseline. This attributes the difference to above-floor
transport, which includes both flight and arrival steps.
Both variants pass the default native dry-scene test and fail
Q-SAND. The temporary source was restored. CUDA was untested.

#### Separate floor arrival from transport above the floor

On 146f854, classify steps starting above the floor using
both the ordinary and midpoint swept endpoints. Apply the
vertical midpoint in the arrival control when either path
reaches the floor, and in the above-floor control when
neither path reaches it.

| Case | Arrival only | Above-floor transport only |
|---|---:|---:|
| Coarse | 29.086256° | 26.984932° |
| Fine | 26.710930° | 24.313435° |
| Coarse half-step | 29.006001° | 28.442677° |
| Fine half-step | 27.033504° | 26.729852° |

Arrival-only angles stay within 0.051° of baseline.
Above-floor transport reproduces the all-above-start control
within 0.023°. The effect is therefore mainly in steps
where neither displacement reaches the floor. Both native
default tests pass and Q-SAND fails. Temporary source was
restored. CUDA was untested.

#### Separate gravity and stress in the above-floor midpoint

On a638560, subtract half of the uniform gravity increment
from the final vertical PIC velocity in one control. In
the other control, subtract half of the gathered grid
increment minus the uniform gravity increment. Apply each
change only when both its path and the baseline stay
above the floor. The residual includes the small difference
between uniform gravity and the gathered occupied-face kick.

| Case | Gravity midpoint | Residual midpoint |
|---|---:|---:|
| Coarse | 30.401200° | 25.337593° |
| Fine | 27.517651° | 23.207458° |
| Coarse half-step | 29.398101° | 28.012009° |
| Fine half-step | 27.429057° | 26.256851° |

Gravity centering raises all four angles. Centering the
residual lowers all four, accounting for the sign of the
combined midpoint result. Both variants pass default native
dry scenes and fail Q-SAND. Temporary source was restored.
CUDA was untested for these controls.

#### Combine gravity-centered displacement and pre-kick deformation

On 8777acf, combine the above-floor gravity midpoint for
displacement with the pre-kick yy stress deformation sample.
The final PIC velocity still carries the full impulse into
contact and storage. The existing swept deformation split
uses the modified deformation gradient.

| Case | Pre-kick deformation only | Combined timing |
|---|---:|---:|
| Coarse | 32.330788° | 32.625579° |
| Fine | 28.336343° | 28.705882° |
| Coarse half-step | 30.660133° | 30.998102° |
| Fine half-step | 28.179787° | 28.526939° |

The combined change raises each angle by 0.29° to 0.37°.
Both coarse cases pass. Both fine cases remain below 30°
and the fine half-step radius is 0.162090 m. The default
native dry-scene test passes and Q-SAND fails. Temporary
source was restored. CUDA was untested.

#### Extend the combined timing control to three seconds

Use the same native combined timing operator and four
nominal and half-step cases on 8777acf. Extend only the
endpoint from 1.5 to 3 s.

| Case | Angle at 1.5 s | Angle at 3 s | Radius at 3 s (m) |
|---|---:|---:|---:|
| Coarse | 32.625579° | 34.231526° | 0.153019 |
| Fine | 28.705882° | 26.184369° | 0.158969 |
| Coarse half-step | 30.998102° | 27.815078° | 0.156228 |
| Fine half-step | 28.526939° | 24.097968° | 0.164120 |

Only the coarse nominal case passes at 3 s. The coarse
half-step loses 3.18° after its earlier pass and the two
fine cases flatten further. The native matrix exits 1.
The temporary solver and fixture were restored.

#### Check persistence of the combined timing control

On b06dea6, extend the same combined timing runs to 6 s
with the original angle, radius and short drift bounds.

| Case | Angle at 3 s | Angle at 6 s | Radius at 6 s (m) |
|---|---:|---:|---:|
| Coarse | 34.231526° | 33.098793° | 0.153019 |
| Fine | 26.184369° | 24.760041° | 0.158969 |
| Coarse half-step | 27.815078° | 27.422196° | 0.158282 |
| Fine half-step | 24.097968° | 19.515452° | 0.164120 |

The coarse nominal case still passes the endpoint. It
loses 4.37 mm of height from 3 to 6 s at fixed radius.
The fine half-step loses 15.24 mm of height at fixed
radius. The native matrix exits 1. The sources were
restored and CUDA was untested for this extension.

#### Localize pre-kick deformation by height

On b06dea6, keep the above-floor gravity midpoint displacement.
Apply the pre-kick normal stress sample only within one grid
cell of the floor, then repeat above that layer only.

| Case | Full pre-kick sample | Within one cell | Above one cell |
|---|---:|---:|---:|
| Coarse | 32.625579° | 33.637073° | 30.179197° |
| Fine | 28.705882° | 29.742537° | 27.063117° |
| Coarse half-step | 30.998102° | 31.096001° | 29.424344° |
| Fine half-step | 28.526939° | 28.918862° | 27.331978° |

The near-floor variant raises every angle relative to the
whole-pile correction. Both fine cases still miss 30° and
the fine half-step radius is 0.160757 m. Both default native
dry-scene tests pass and both Q-SAND matrices fail. CUDA was
untested. The temporary solver source was restored.

#### Smooth the pre-kick correction across the bottom cell

On fa9c791, replace the sharp one-cell cutoff with a cubic
smoothstep falloff from full correction at the floor to
zero at one grid cell. Keep gravity-centered displacement
and the existing gate limits.

| Case | Sharp cutoff | Smooth falloff | Smooth radius (m) |
|---|---:|---:|---:|
| Coarse | 33.637073° | 33.284429° | 0.144036 |
| Fine | 29.742537° | 30.473579° | 0.153564 |
| Coarse half-step | 31.096001° | 30.728948° | 0.143619 |
| Fine half-step | 28.918862° | 28.685323° | 0.157818 |

The smooth control passes three of the four native 1.5 s
cases and the default dry-scene test. The fine half-step
meets the radius bound and misses the angle bound. Q-SAND
still fails. CUDA and later horizons were untested. The
experimental solver source was restored.

#### Check persistence of the smooth bottom-cell correction

On fa9c791, extend the smooth control to 6 s with unchanged
material, sampling and acceptance bounds.

| Case | Angle at 1.5 s | Angle at 6 s | Radius at 6 s (m) |
|---|---:|---:|---:|
| Coarse | 33.284429° | 34.653684° | 0.150013 |
| Fine | 30.473579° | 27.250852° | 0.154368 |
| Coarse half-step | 30.728948° | 28.398747° | 0.153679 |
| Fine half-step | 28.685323° | 20.674222° | 0.160561 |

The coarse nominal case meets the endpoint bounds and falls
0.998° in the final second. The two earlier passes on fine
nominal and coarse half-step fall below 30°. The native
matrix exits 1. Temporary sources were restored. CUDA was
untested for this extension.

### Compare the merged sand solver at matched absolute timesteps

On f0354f1, keep the original material and 1.5 s endpoint.
Run both grid widths with the fine nominal and half-step
durations on native and SM120.

| Step (s) | Coarse angle | Fine angle |
|---|---:|---:|
| 0.00156250 | 26.727865° | 26.761332° |
| 0.00078125 | 30.385098° | 27.033753° |

Both backends agree to six decimal places. At the larger
matched step the grid-width difference is 0.033467°.
At the smaller step it grows to 3.351345°. Both matrices
fail, with the coarse smaller-step case alone meeting
all fixture bounds. Angle is the only failing criterion
in the other three cases. The temporary fixture was restored.

#### Shift both horizontal grid coordinates at matched timesteps

On f0354f1, shift the grid origin by half a cell in x and z.
Keep particle coordinates, wall and material fixed.

| Step (s) | Coarse unshifted | Coarse shifted | Fine unshifted | Fine shifted |
|---|---:|---:|---:|---:|
| 0.00156250 | 26.727865° | 25.317653° | 26.761332° | 24.650853° |
| 0.00078125 | 30.385098° | 30.027867° | 27.033753° | 25.059682° |

The coarse smaller-step case still passes, with 0.028°
margin above the lower bound. At that step the grid-width
gap increases from 3.351° to 4.968°. The other three cases
fail angle alone. Native and SM120 agree to six decimal
places. The temporary fixture was restored.

#### Double particle sampling at matched timesteps

On 48de18d, use the original grid origin and double sampling
per axis at both widths. Run the same two absolute timesteps
through 1.5 s on native. The annular estimator resolves all
four quadrants in each case.

| Grid width (m) | Step (ms) | Envelope | Radius (m) | Quadrant slope |
|---|---:|---:|---:|---:|
| 0.075 | 1.5625 | 25.763030° | 0.168944 | 11.791403° |
| 0.0625 | 1.5625 | 25.002454° | 0.182075 | 21.925421° |
| 0.075 | 0.78125 | 27.342640° | 0.166654 | 4.997874° |
| 0.0625 | 0.78125 | 23.529643° | 0.183927 | 13.925098° |

All four cases fail the angle and radius checks and pass the
short drift and kinetic energy checks. The smaller-step
fine-grid deficit in envelope angle grows from 3.35° at the
original sampling to 3.81°. The independent slope is shallower
than the envelope in every case. The fixture exits 1.
The temporary fixture was restored. CUDA was untested.

SM120 reproduces the four dense matched-step failures on
48de18d. Envelope angles and fitted quadrant slopes agree
with native within 0.000002°. Radii agree to six decimal
places. All cases fail angle and radius and pass kinetic
energy and short drift. The CUDA fixture exits 1 with
`--q-sand-dense-column-long-settle` after matching the two
absolute timesteps. The temporary fixture was restored.

#### Track bulk height at the smaller matched timestep

On 0d1cd2d, trace mass-weighted height and vertical velocity
for the original particle sampling at both widths and the
same 0.78125 ms step. The mass fraction within half a cell
uses each grid's own width. Heights are measured from the
same physical floor.

| Time (s) | Coarse mean height (mm) | Fine mean height (mm) |
|---|---:|---:|
| 0.1961 | 111.371 | 109.257 |
| 0.3914 | 64.690 | 53.551 |
| 0.7820 | 37.481 | 24.057 |
| 1.5000 | 19.387 | 11.794 |

At 1.5 s the fine mean height is 7.59 mm lower and the
maximum height is only 2.25 mm lower. Mean downward
velocity remains 18.4 mm/s on coarse and 10.9 mm/s on
fine. The integrated vertical momentum residuals are
−1.10e-4 and 1.48e-5 N·s. The fine case's greater bulk
settlement is already visible by 0.3914 s. These global
receipts leave local load distribution open. The native
diagnostic reproduced the matched-step angle results,
and the temporary fixture was restored.

#### Trace early floor support at the smaller matched timestep

On 0d1cd2d, use the 0.78125 ms step at both widths and sum
accepted vertical wall impulses over successive 0.1 s
intervals. Both runs begin with zero vertical momentum.

| Interval (s) | Coarse floor impulse (N·s) | Fine floor impulse (N·s) |
|---|---:|---:|
| 0–0.1 | 0 | 0 |
| 0.1–0.2 | 2.488292 | 2.318574 |
| 0.2–0.3 | 4.466719 | 4.564617 |
| 0.3–0.4 | 2.074988 | 2.054150 |
| 0.4–0.5 | 2.309966 | 2.444664 |

The fine pile receives 0.169718 N·s less initial support
by 0.2 s despite the same contact count (477). Its momentum
is lower by the same amount. By 0.5 s cumulative fine
support exceeds coarse by 0.042040 N·s, after the fine
pile has already settled lower. This locates the first
global impulse difference during collapse and leaves
the local force distribution unresolved. The native
fixture reproduces the matched angles and was restored.

#### Separate arrival and resident support during first collapse

On 0d1cd2d, classify the swept normal impulse by starting
height at the matched 0.78125 ms step. In the interval
0.1–0.2 s both widths record nine arrivals supplying
0.527671 N·s and 468 floor-start contacts. The floor-start
impulse is 1.960622 N·s on coarse and 1.790903 N·s on fine.
This accounts for the entire initial 0.169718 N·s support
difference within trace precision. The 0.2–0.4 s windows
contain only floor-start contacts in both runs. The
native fixture reproduces the previous matched angles.
The temporary solver and fixture were restored.

#### Decompose the first resident support difference

On 0d1cd2d, gather the deposited vertical velocity, gravity
kick and stress kick separately at ordinary floor-start
contacts. Negate each mass-weighted contribution so the sum
gives the impulse needed to stop incoming vertical motion.

| Grid width (m) | Deposit (N·s) | Gravity (N·s) | Stress (N·s) | Measured wall (N·s) |
|---|---:|---:|---:|---:|
| 0.075 | 1.683442 | 0.134504 | 0.142675 | 1.960622 |
| 0.0625 | 1.472552 | 0.134504 | 0.183847 | 1.790903 |

The terms close to roundoff over 0.1–0.2 s. The fine grid
gathers 0.210891 N·s less downward momentum from the
deposit stage and 0.041172 N·s more from stress. Gravity
matches. This points to the pre-force transfer and the
carried velocity distribution for the next comparison.
The native trace reproduces the matched angles. Temporary
sources were restored.

### Disable vertical affine deposition at matched timesteps

On 48302e2, set only the vertical APIC row to zero in
ordinary granular deposits. Keep horizontal affine
deposition, translation, grid stress and the wall sweep.

| Step (ms) | Grid width (m) | Original angle | No vertical affine deposit |
|---|---:|---:|---:|
| 1.56250 | 0.075 | 26.727865° | 37.566203° |
| 1.56250 | 0.0625 | 26.761332° | 30.768585° |
| 0.78125 | 0.075 | 30.385098° | 52.469166° |
| 0.78125 | 0.0625 | 27.033753° | 43.640523° |

The angle increases in every case. The smaller-step
grid-width gap grows to 8.83°. Only the fine nominal
case passes. The default native dry-scene test also
fails, with its coarse pile reaching 49.046609° at
0.75 s. Vertical affine deposition strongly affects
settlement and removing it does not qualify the gate.
Temporary solver and fixture sources were restored.
CUDA was untested.

#### Remove only the vertical normal affine term

On 48302e2, repeat the matched timestep control with only
the yy APIC entry set to zero at deposit. Keep the two
cross derivatives of vertical velocity.

| Step (ms) | Width (m) | No vertical row | No yy entry |
|---|---:|---:|---:|
| 1.56250 | 0.075 | 37.566203° | 40.222214° |
| 1.56250 | 0.0625 | 30.768585° | 35.961672° |
| 0.78125 | 0.075 | 52.469166° | 53.173398° |
| 0.78125 | 0.0625 | 43.640523° | 46.948907° |

Removing yy alone raises every angle more than removing
the full row. All four cases pass radius, short drift and
kinetic energy and fail the upper angle bound. The default
native sand case also fails at 50.248767°. The result
isolates the dominant vertical feedback to yy in this
control. Temporary sources were restored. CUDA was untested.

#### Separate compressive and extensional vertical affine feedback

On 48302e2, run the registered native matrix while dropping
the yy deposit term only when it is negative or only when it
is positive. Use the original timesteps for this comparison.

| Case | Baseline | Drop compression | Drop extension |
|---|---:|---:|---:|
| Coarse | 29.113507° | 37.099832° | 27.619309° |
| Fine | 26.761332° | 36.036959° | 25.824766° |
| Coarse half-step | 29.006384° | 49.762374° | 28.756002° |
| Fine half-step | 27.033753° | 46.949070° | 26.741414° |

Dropping compression raises every angle above 35° and fails
the default native dry-scene test. Dropping extension lowers
every angle and passes the default test. Both matrices fail.
The influence of the yy term is concentrated in its negative
values in this control. Temporary source was restored.
CUDA was untested.

#### Retain part of the compressive vertical affine feedback

On 302f85e, scale only negative yy at the ordinary granular
deposit. Keep the registered timesteps and acceptance bounds.

| Case | Baseline | 75% retention | 50% retention |
|---|---:|---:|---:|
| Coarse | 29.113507° | 29.398108° | 31.949424° |
| Fine | 26.761332° | 28.912552° | 31.497797° |
| Coarse half-step | 29.006384° | 35.972185° | 41.928780° |
| Fine half-step | 27.033753° | 33.386845° | 39.020718° |

The 75% control passes the fine half-step only. The 50%
control passes both nominal cases and overshoots both
half-steps. The default native sand case fails at 37.89°
and 43.13° respectively. A fixed retention fraction leaves
a large timestep dependence. Temporary source was restored.
CUDA was untested for these diagnostic controls.

#### Scale compressive retention by physical time

On 302f85e, multiply negative yy by exp(−400 dt), with
dt in seconds, at ordinary granular deposit. At the
registered steps the factor ranges from 0.472 to 0.732.

| Case | Final angle |
|---|---:|
| Coarse | 32.238003° |
| Fine | 31.145172° |
| Coarse half-step | 37.620263° |
| Fine half-step | 33.832469° |

Three native cases pass, and the coarse half-step exceeds
35°. The default native sand case also fails at 43.616929°.
This reduces the half-step overshoot of constant 50%
retention, while leaving the short-horizon regression and
Q-SAND failure. Temporary source was restored. CUDA was untested.

#### Localize time-scaled compression by stencil support

On 53e7662, apply the same exp(−400 dt) multiplier only
when a nonzero vertical weight crosses below the floor,
then apply it only to stencils wholly on or above the floor.

| Case | Floor-crossing only | Uncut only |
|---|---:|---:|
| Coarse | 25.001785° | 32.888475° |
| Fine | 24.109886° | 32.192299° |
| Coarse half-step | 29.380047° | 36.841241° |
| Fine half-step | 27.136347° | 33.641611° |

The floor-crossing control passes the native default test
and fails all four extended angles. The uncut control passes
three extended cases and fails the default coarse case at
42.895836°. The angle improvement from time-scaled retention
occurs on uncut stencils in this comparison. Both native
Q-SAND matrices fail. The temporary source was restored.
CUDA was untested.

#### Trace the coarse collapse with uncut compression retention

On 68ae076, sample the native coarse nominal angle at the
first accepted step at or after each 0.1 s mark. Compare
the original solver with the uncut exp(−400 dt) control.

| Time (s) | Baseline | Uncut control |
|---|---:|---:|
| 0.10125 | 79.287674° | 79.287674° |
| 0.200625 | 72.108231° | 72.702763° |
| 0.30000 | 47.080203° | 60.609532° |
| 0.500625 | 37.962759° | 50.694881° |
| 0.800625 | 32.226688° | 41.792279° |
| 1.50000 | 29.113507° | 32.888475° |

The largest sampled gap is 13.529329° at 0.3 s, during
the initial collapse. By 1.5 s the gap falls to 3.774968°
and the control has lost 5.17° over the final half-second.
The short-horizon overshoot begins during collapse, and
its later angle pass does not establish settlement.
Both native matrices exit 1. Temporary sources were restored.

#### Start uncut retention after the initial collapse

On 68ae076, enable uncut exp(−400 dt) compression retention
only after the 0.75 s accepted sample. All four angles at
the switch match baseline. Use the original 1.5 s gate.

| Case | Baseline | Late-only retention | Retention from release |
|---|---:|---:|---:|
| Coarse | 29.113507° | 29.225909° | 32.888475° |
| Fine | 26.761332° | 27.022180° | 32.192299° |
| Coarse half-step | 29.006384° | 29.758772° | 36.841241° |
| Fine half-step | 27.033753° | 27.565543° | 33.641611° |

Late-only retention raises the final angles by 0.11° to
0.75° and all four remain below 30°. It passes the default
native test, which ends at the switch time. The larger
changes seen when retention starts at release depend on
earlier history. The temporary solver and fixture were
restored. CUDA was untested for this control.

#### Retain compression correction during collapse only

On 87e9687, apply the uncut exp(−400 dt) control from release
through the accepted 0.75 s sample. Return to the ordinary
solver afterward. The native 1.5 s Q-SAND CTest entry passes
all four cases. Repeat at 3 s with the same switch and bounds.

| Case | Angle at 0.75 s | Angle at 1.5 s | Angle at 3 s |
|---|---:|---:|---:|
| Coarse | 42.895836° | 31.741264° | 29.018562° |
| Fine | 39.173403° | 31.001955° | 27.851871° |
| Coarse half-step | 49.152848° | 34.547276° | 26.687394° |
| Fine half-step | 43.759664° | 31.809489° | 25.493112° |

The 1.5 s pass is transient. Every case loses another
2.72° or more by 3 s and falls below the angle band. The
short default coarse endpoint also exceeds 35°. CUDA was
untested for this diagnostic. Both sources were restored.


#### Compare the unchanged solver at three seconds

On 0240092, change the four end times from 1.5 to 3.0 s in
the shared `--q-sand-column-long-settle` fixture. Keep the
solver, material, sampling, and acceptance bounds fixed.
Build both dry-scene executables and run the matrix.

```powershell
cmake --build out/requalified --target fluidity_m5_dry_scene_tests fluidity_cuda_m5_dry_scene_tests
ctest --test-dir out/requalified -R '^QSand\.' --output-on-failure
```

| Case | Baseline angle, native and SM120 | Baseline 0.05 s drift | Native collapse-only angle |
|---|---:|---:|---:|
| Coarse | 30.519543° | 0.074589° | 29.018562° |
| Fine | 23.904249° | 0.055416° | 27.851871° |
| Coarse half-step | 22.853454° | 0.137065° | 26.687394° |
| Fine half-step | 20.811628° | 0.171848° | 25.493112° |

Both baseline matrices fail. All four cases meet the radius,
kinetic energy, and short drift bounds. The correction keeps
three cases steeper by 3.83–4.68° at 3 s, but all three remain
below 30°. The coarse case loses 1.50° with the correction.
The fine half-step baseline still loses 3.83° in the last
second even though its final kinetic energy is 0.000037 J.
The fixture edit was restored.


#### Bound the effect of late floor friction

On 283e2ce, repeat the native three-second column control.
In the fixture, keep the floor coefficient at 0.7 until the
0.75 s sample, then change only the coefficient to 1e6 or 0
for the remaining accepted steps. Use the same native build
target and QSand.native CTest entry as the baseline above.

| Case | Coefficient 0.7 | Coefficient 1e6 after 0.75 s | Coefficient 0 after 0.75 s |
|---|---:|---:|---:|
| Coarse | 30.519543° | 30.519543° | 10.806232° |
| Fine | 23.904249° | 23.904249° | 8.134887° |
| Coarse half-step | 22.853454° | 22.853454° | 17.119293° |
| Fine half-step | 20.811628° | 20.811628° | 15.842373° |

The large coefficient leaves every reported angle, drift,
radius and kinetic energy unchanged at printed precision.
The zero coefficient reduces the angle in every case and
both fine cases also exceed the radius bound. Late basal
friction therefore supports the pile, but raising the
existing Coulomb cap does not arrest the residual flattening.
Both native matrices fail. CUDA was untested for the two
controls. The fixture edit was restored.


#### Compare bulk settlement with the envelope

On 006be35, repeat the unchanged native three-second matrix.
At the first accepted endpoint at or after 1.5 s and at 3 s,
report arithmetic mean height and radius over all 63 equal-mass
particles. Count particles within 1e-8 m of the floor. Keep
the original envelope and acceptance checks.

| Case | Mean height near 1.5 s → 3 s (mm) | Mean radius near 1.5 s → 3 s (mm) | Floor count near 1.5 s → 3 s |
|---|---:|---:|---:|
| Coarse | 8.702 → 3.676 | 82.109 → 83.281 | 39 → 56 |
| Fine | 6.295 → 2.535 | 87.672 → 88.290 | 46 → 59 |
| Coarse half-step | 16.210 → 5.777 | 83.580 → 85.146 | 30 → 43 |
| Fine half-step | 11.786 → 3.377 | 90.275 → 91.572 | 30 → 50 |

The fine samples start at 1.501562500 s and 1.500781250 s
respectively. Mean height falls by 60–71% in the three
cases that miss the final angle band. Their mean radii
grow by at most 1.57 mm. The passing coarse case also loses
58% of its mean height while its highest particle rises
from 75.240 to 80.950 mm. Its passing envelope therefore
coexists with substantial bulk settlement. The native
matrix reproduces the baseline angles and fails. The
temporary fixture edit was restored. CUDA was untested
for the bulk trace.


#### Separate compression from extension in post-contact deformation

On e611eb7, repeat the native three-second matrix and
replace only the yy component of the post-impact deformation
gradient with the gathered yy rate, its compressive part
min(0, yy), or half that compressive rate. The swept wall
impulse, transport and APIC storage use their original paths.

| Case | Baseline | Signed yy | Full compression | Half compression |
|---|---:|---:|---:|---:|
| Coarse | 30.519543° | 12.916662° | 39.909968° | 38.235365° |
| Fine | 23.904249° | 12.742297° | 32.235648° | 30.910923° |
| Coarse half-step | 22.853454° | 23.649621° | 31.686193° | 27.489086° |
| Fine half-step | 20.811628° | 22.741347° | 31.861087° | 26.817953° |

Full compression puts three cases in band at 3 s, while
coarse overshoots and the default native dry-scene test
fails at 35.674372°. The signed and half-compression controls
pass the default native test. All three extended matrices
fail. The opposing response to signed and compression-only
updates shows that extension matters to the contact split.
These were native controls. The fixture and solver edits
were restored.


#### Restrict post-contact deformation to zero-fraction hits

On b8d3a1d, apply the same yy control only when the swept
contact fraction equals zero. Keep the original projected
gradient for positive-time hits. Compare compression
min(0, yy) and extension max(0, yy) at 3 s on native.

| Case | Baseline | Compression at zero fraction | Extension at zero fraction |
|---|---:|---:|---:|
| Coarse | 30.519543° | 39.909968° | 12.957675° |
| Fine | 23.904249° | 32.236242° | 11.354341° |
| Coarse half-step | 22.853454° | 31.685760° | 21.186607° |
| Fine half-step | 20.811628° | 31.860759° | 19.436887° |

Compression agrees within 0.0006° with the all-hit control
in every case and still fails the default dry scene at
35.674372°. Restricting it to zero-fraction hits does not
remove the coarse overshoot. The extension-only control
passes the default native dry scene and lowers all four
extended angles relative to baseline. Both QSand.native
runs fail. CUDA was untested and both edits were restored.


#### Extend zero-fraction compression on native and CUDA

On 08797c1, repeat the zero-fraction compression diagnostic
above and set the four shared column end times to 6 s.
Build both dry-scene targets and run both QSand CTest entries.
Native and SM120 match to six decimals for the geometry
and drift measures below.

| Case | Angle at 3 s | Angle at 6 s | Absolute drift over last ≈1 s | Radius at 6 s (m) |
|---|---:|---:|---:|---:|
| Coarse | 39.909968° | 36.442893° | 0.418467° | 0.142033 |
| Fine | 32.236242° | 30.159032° | 0.335024° | 0.147397 |
| Coarse half-step | 31.685760° | 34.609817° | 0.974711° | 0.142443 |
| Fine half-step | 31.860759° | 30.112880° | 0.511944° | 0.152230 |

Both matrices fail on the coarse angle. Both fine cases
remain above 30° at 6 s with less than 0.16° of margin
and are still flattening over the last second. Every case
meets the radius, kinetic energy, and short drift limits.
These results leave Q-SAND open. Temporary sources were
restored.


#### Bound resident compression by incoming normal speed

On c524856, repeat the native three-second matrix with the
zero-fraction control. Clamp the retained yy rate to the
interval from min(0, incoming PIC y velocity / cell width)
to zero. This bounds the per-step added normal strain by
the incoming normal travel divided by the cell width.

| Case | Full resident compression | Speed-bounded compression |
|---|---:|---:|
| Coarse | 39.909968° | 39.013679° |
| Fine | 32.236242° | 31.662281° |
| Coarse half-step | 31.685760° | 28.385289° |
| Fine half-step | 31.860759° | 27.723006° |

The bound reduces the coarse overshoot by 0.90° but
reduces both half-step angles by more than 3°. The default
native dry scene also fails at 35.114976°. Both native CTest
entries fail. CUDA was untested and both temporary edits
were restored.


### Recheck production snow–water after the upstream liquid changes

On 2c18ed5, rebuild the production simulator, Blender bridge,
and both M5PlanarCoupling executables with Windows Release
and SM120 enabled. Both planar coupling CTest entries pass.
Run the authored Blender 5.2.3 bake and reopen command above
once per backend with fresh output directories. Both backends
report the same minimum snow volume ratio (0.648350539),
minimum log plastic Jacobian (−0.420240034), and emitted
water gap (0.631821917 m). These match the earlier merged
qualification. Q-SAND remains open.

```powershell
cmake --build out/requalified --target fluidity_simulate fluidity_blender_bridge fluidity_m5_planar_coupling_tests fluidity_cuda_m5_planar_coupling_tests
ctest --test-dir out/requalified -R '^M5PlanarCoupling\.' --no-tests=error --output-on-failure
$tree = (Get-Location).Path
$bin = (Resolve-Path out/requalified).Path
$blender = 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe'
foreach ($backend in 'cpu', 'cuda') {
  & $blender --background --factory-startup --python-exit-code 1 --python "$tree/tools/blender/verify_m5_snow_water.py" -- $tree $bin "$tree/out/authored-recheck-2c18ed5-$backend" $backend
}
```


#### Separate collapse and late resident-floor compression

On 9bc9c8f, apply the full zero-fraction compression diagnostic
from release to the accepted 0.75 s sample, or only on steps
after that sample. Run the native column matrix through 3 s
using the same CTest entry and bounds as above.

| Case | Baseline | Collapse only | Late only | Full resident compression |
|---|---:|---:|---:|---:|
| Coarse | 30.519543° | 39.909968° | 30.519588° | 39.909968° |
| Fine | 23.904249° | 32.236242° | 23.904249° | 32.236242° |
| Coarse half-step | 22.853454° | 31.646974° | 23.299272° | 31.685760° |
| Fine half-step | 20.811628° | 31.845897° | 20.941489° | 31.860759° |

Collapse-only compression reproduces the full diagnostic
within 0.039°. Late-only compression improves the final
angles by at most 0.446° and passes the default native dry
scene. Both extended matrices fail. The three-second
geometry effect comes predominantly from the first 0.75 s.
CUDA was untested. Both temporary sources were restored.


#### Attenuate resident compression over the acoustic timescale

On d5c90e2, multiply min(0, gathered yy) at zero-fraction
floor hits by exp(−dt c / (h C)), where c is the particle's
accepted acoustic speed bound, h is the cell width and C
is the dry acoustic CFL setting. Keep all four cases and
acceptance bounds. Rebuild the native dry-scene target
and run M5DryScenes.native and QSand.native. Repeat
QSand.native with the endpoint extended from 3 to 6 s.

| Case | Angle at 3 s | Angle at 6 s |
|---|---:|---:|
| Coarse | 39.005939° | 36.943464° |
| Fine | 31.659782° | 29.643051° |
| Coarse half-step | 30.493135° | 32.887773° |
| Fine half-step | 30.868832° | 29.326368° |

The default native dry scene passes. Both extended matrices
fail. Both fine cases cross below 30° between 3 and 6 s,
and the coarse nominal case remains above 35°. This native
control was not run on CUDA. Temporary edits were restored.


#### Measure single-particle sensitivity of the envelope

On 9db511b, run the unchanged native three-second column
matrix. At the final accepted state, omit each of the 63
particles in turn and recompute both maxima and the same
atan2 envelope. Record the minimum and maximum of the 63
estimates. This measures sensitivity to a single sampled
particle in a fixed state. It does not measure variation
between particle samplings of the simulation.

| Case | Full sample | Leave-one-out range |
|---|---:|---:|
| Coarse | 30.519543° | 22.897593–30.519543° |
| Fine | 23.904249° | 19.776255–23.904249° |
| Coarse half-step | 22.853454° | 17.244575–22.853454° |
| Fine half-step | 20.811628° | 17.042990–20.811628° |

The coarse pass has a 0.52° margin over the lower bound
and its estimate falls by 7.62° on a single omission.
The three other cases remain below the band for every
omission. The full-sample checks are unchanged and the
native matrix fails. CUDA was untested for this trace.
The temporary fixture edit was restored.


#### Compare the two particle samplings at three seconds

On 89e7511, change only the four column end times from
1.5 to 3 s and run the native executable with
`--q-sand-dense-column-long-settle`. The dense sampling
uses 504 particles at 25 mm spacing. The existing baseline
uses 63 particles at 50 mm spacing. Both represent the same
released column dimensions, total mass and material.

| Case | Sparse angle | Dense angle | Sparse radius (m) | Dense radius (m) |
|---|---:|---:|---:|---:|
| Coarse | 30.519543° | 29.703637° | 0.137319 | 0.170290 |
| Fine | 23.904249° | 26.954622° | 0.146767 | 0.182075 |
| Coarse half-step | 22.853454° | 21.853351° | 0.143316 | 0.172975 |
| Fine half-step | 20.811628° | 20.623778° | 0.155423 | 0.185666 |

Resampling changes the angle by up to 3.05° and increases
the radius by 29.7–35.3 mm. The dense matrix exits 1.
All four dense cases miss both angle and radius bounds,
while passing short drift and kinetic energy. The sparse
coarse angle pass is sensitive to initial sampling as well
as to omission from the final state. CUDA was untested for
the dense endpoint. The temporary fixture edit was restored.


#### Locate the radial spread in the dense baseline

On 24147a5, repeat the native dense three-second run above.
At 0.75 and 3 s, sort the 504 equal-mass particle radii and
read the 95th percentile at index floor(0.95 × (N − 1)).
Count samples at or beyond 0.16 m at both endpoints.

| Case | 95th percentile at 0.75 → 3 s (m) | Count at or beyond 0.16 m at 0.75 → 3 s |
|---|---:|---:|
| Coarse | 0.140213 → 0.147554 | 4 → 12 |
| Fine | 0.148825 → 0.155540 | 20 → 20 |
| Coarse half-step | 0.135840 → 0.144721 | 0 → 12 |
| Fine half-step | 0.147016 → 0.157702 | 16 → 20 |

At 3 s, 2.38–3.97% of the mass lies at or beyond the radius
limit. The 95th percentile grows by 6.71–10.69 mm in every
case and stays within the limit. The excess reaches multiple
particles, including cases that already exceeded the limit
at 0.75 s. The maximum-radius check remains in effect. The
matrix exits 1 with the same final geometry as before. CUDA
was untested for this trace. The temporary edit was restored.


#### Check resident compression at dense sampling

On 84c3e93, apply the zero-fraction compression diagnostic
to the native 504-particle column at the same 3 s endpoint.
Use the original dense column bounds and retain the swept
wall impulse, transport, and APIC storage.

| Case | Dense baseline angle | Dense compression angle | Dense compression radius (m) |
|---|---:|---:|---:|
| Coarse | 29.703637° | 35.033895° | 0.172380 |
| Fine | 26.954622° | 29.949241° | 0.182183 |
| Coarse half-step | 21.853351° | 27.032469° | 0.173593 |
| Fine half-step | 20.623778° | 25.609058° | 0.184981 |

The angles increase by 2.99–5.33°, while the radii change
by at most 2.1 mm. All four cases still miss the angle and
radius bounds and pass the short drift and kinetic energy
bounds. The dense native command exits 1. The compression
control raises height without resolving the excessive spread.
CUDA was untested and both temporary edits were restored.


#### Locate the first radius-limit crossing

On 274e124, repeat the native dense three-second matrix.
For each particle, record its height at the first accepted
step with radius at least 0.16 m. Report the cohort still
at or beyond the limit at 3 s. A particle within 1e-8 m
of the floor counts as on the floor.

| Case | Outside at 3 s | On floor at first crossing | On floor at 3 s | Highest first crossing (m) |
|---|---:|---:|---:|---:|
| Coarse | 12 | 0 | 12 | 0.058806 |
| Fine | 20 | 0 | 20 | 0.082933 |
| Coarse half-step | 12 | 0 | 8 | 0.068649 |
| Fine half-step | 20 | 0 | 20 | 0.088380 |

Every particle outside at 3 s first crossed above the floor.
Most later settled onto the floor. This points to transport
above the floor as the stage at which the excess radius first
appears. The trace does not exclude effects from prior floor
contact or grid forces. The native matrix reproduces the
baseline geometry and exits 1. CUDA was untested and the
fixture edit was restored.


#### Check floor history before the radius crossing

On e1f4567, repeat the native dense three-second run. Record
the first accepted endpoint at height at most 1e-8 m and the
first accepted endpoint at radius at least 0.16 m for each
particle. Report the cohort still at or beyond the limit
at 3 s.

| Case | Outside at 3 s | Earlier floor endpoint | First-crossing time range (s) |
|---|---:|---:|---:|
| Coarse | 12 | 0 | 0.676875–1.177500 |
| Fine | 20 | 0 | 0.403125–0.723438 |
| Coarse half-step | 12 | 0 | 0.933750–1.394063 |
| Fine half-step | 20 | 0 | 0.516406–0.907813 |

No member of this cohort had an accepted floor endpoint
before crossing. Grid-mediated effects from floor contact on
other particles remain possible. In all four cases the last
first crossing occurs before 1.4 s. The matrix reproduces
the baseline geometry and exits 1. CUDA was untested and
the temporary fixture edit was restored.


#### Increase horizontal force regularization for the dense pile

On 30b8c8b, increase the stress acceleration denominator
floor from 1e-4 to 1e-3 times the minimum particle mass
on granular x and z faces. Keep the vertical denominator,
APIC deposit, and wall update unchanged. Repeat the native
dense column command with the four end times set to 3 s.

| Case | Angle | Radius (m) | Omitted stress impulse L1 (N·s) |
|---|---:|---:|---:|
| Coarse | 29.701663° | 0.170304 | 0.000261082 |
| Fine | 26.984940° | 0.182075 | 0.000404656 |
| Coarse half-step | 21.852960° | 0.172973 | 0.000460565 |
| Fine half-step | 20.623852° | 0.185666 | 0.000148400 |

Relative to the dense baseline, the maximum radius change
is 0.014 mm and the maximum angle change is 0.031°. All
four cases still miss the radius and angle bounds. The
tenfold horizontal mass floor does not materially affect
the spread in this control. The native command exits 1.
CUDA was untested and both temporary edits were restored.


#### Reduce horizontal APIC deposition in the dense column

On b5e0042, scale the x and z rows of the granular APIC
deposit by zero or one half. Retain the vertical row, fitted
slope storage, deformation gather, stress, and contact.
Run the native dense column at 3 s and the default native
dry-scene test at its original endpoint.

| Case | Dense baseline radius (m) | Zero retention angle / radius (m) | Half retention angle / radius (m) |
|---|---:|---:|---:|
| Coarse | 0.170290 | 37.965752° / 0.138150 | 34.472800° / 0.148559 |
| Fine | 0.182075 | 36.157305° / 0.143913 | 32.805200° / 0.155027 |
| Coarse half-step | 0.172975 | 43.445968° / 0.120120 | 35.365751° / 0.133155 |
| Fine half-step | 0.185666 | 41.610549° / 0.127542 | 33.231875° / 0.142933 |

Every dense radius passes with either control. Half retention
puts three angles in band, but the coarse half-step exceeds
35° and the default coarse pile reaches 52.852714° with a
0.092381 m radius. Zero retention makes the default failure
larger. These results identify horizontal affine transfer as
a contributor to spread and show the regression from uniform
attenuation. CUDA was untested and temporary edits were restored.


#### Attenuate only horizontal expansion in granular APIC

On 14667da, subtract one quarter of the positive horizontal
trace from each horizontal diagonal entry at deposit. The
positive trace is halved while horizontal compression and
off-diagonal entries are unchanged. Run the original native
dry-scene test and both three-second column samplings.

| Case | Sparse angle | Dense angle | Dense radius (m) | Dense last ≈1 s angle drift |
|---|---:|---:|---:|---:|
| Coarse | 38.163989° | 34.527709° | 0.148471 | 0.283589° |
| Fine | 37.744717° | 32.176911° | 0.156825 | 0.176797° |
| Coarse half-step | 36.913127° | 34.533284° | 0.134807 | 7.407639° |
| Fine half-step | 34.018457° | 31.469931° | 0.146609 | 7.153803° |

The dense command exits 0, but both half-step angles fall
more than 7° over the last second. The sparse matrix exits
with three angles above the band and the default native pile
fails at 52.495345° with a 0.092817 m radius. Positive
horizontal trace alone accounts for much of the response to
row attenuation. This control does not demonstrate repose.
CUDA was untested and both temporary edits were restored.


#### Scale horizontal APIC expansion attenuation by step duration

On becba6d, retain exp(−200 dt) of the positive horizontal
trace at deposit, with dt in seconds. Apply the removed
trace equally to the two horizontal diagonal entries.
Repeat the default native test and the sparse and dense
three-second matrices with the original acceptance bounds.

| Case | Sparse angle | Dense angle | Dense radius (m) |
|---|---:|---:|---:|
| Coarse | 36.505935° | 32.734790° | 0.156364 |
| Fine | 34.905420° | 30.761359° | 0.163859 |
| Coarse half-step | 28.110172° | 26.281559° | 0.156245 |
| Fine half-step | 25.969430° | 24.036955° | 0.170850 |

All three test invocations fail. Only the dense coarse
nominal case meets both geometry bounds. Both half-step
angles fall below 30° at either particle sampling. The
default coarse angle is 47.046222°. This rate weakens the
short-scene overshoot but does not improve agreement across
timesteps. CUDA was untested and both edits were restored.


#### Limit horizontal APIC attenuation to volumetric expansion

On 67b15d7, subtract one quarter of max(0, min(horizontal
trace, total trace)) from each horizontal diagonal at deposit.
This preserves horizontal expansion balanced by vertical
compression. Run the default native scene and the sparse and
dense three-second matrices with their original bounds.

| Case | Sparse angle | Dense angle | Dense radius (m) |
|---|---:|---:|---:|
| Coarse | 33.085378° | 31.634141° | 0.160664 |
| Fine | 30.849635° | 29.763998° | 0.166359 |
| Coarse half-step | 25.798491° | 24.268430° | 0.161680 |
| Fine half-step | 24.534414° | 22.530705° | 0.173688 |

The default coarse angle drops to 38.981000° compared with
52.495345° when all positive horizontal trace is attenuated.
It still fails the default check. Both sparse nominal cases
pass at 3 s and both half-step cases fail. All four dense
radii exceed 0.16 m. CUDA was untested and both temporary
edits were restored.


#### Attenuate positive horizontal trace after collapse

On c7c1f2e, halve the positive horizontal APIC trace only
on steps starting at or after the accepted 0.75 s sample.
Run the default native dry scene and both three-second
column samplings with their original bounds.

| Case | Sparse angle | Dense angle | Dense radius (m) |
|---|---:|---:|---:|
| Coarse | 30.609475° | 29.903200° | 0.168704 |
| Fine | 24.554257° | 27.649621° | 0.181639 |
| Coarse half-step | 23.531808° | 22.566166° | 0.168540 |
| Fine half-step | 21.579585° | 21.160088° | 0.183133 |

The default test passes and the 0.75 s samples match
baseline. The late-only control reduces the dense radii
by 0.44–4.44 mm, leaving all four above 0.16 m. Its maximum
angle gain across both samplings is 0.77°. Both extended
runs fail. Most of the response to this attenuation requires
changing the initial collapse. CUDA was untested and the
temporary edits were restored.


#### Attenuate positive horizontal trace during collapse only

On ec9f2c7, halve positive horizontal APIC trace up to
the accepted 0.75 s sample, then use the original transfer.
Run the default native scene and both three-second matrices.

| Case | Sparse angle | Dense angle | Dense radius (m) |
|---|---:|---:|---:|
| Coarse | 37.042214° | 32.521315° | 0.162503 |
| Fine | 35.025561° | 30.320245° | 0.166853 |
| Coarse half-step | 29.510845° | 26.970362° | 0.159899 |
| Fine half-step | 28.473819° | 24.630873° | 0.173583 |

The default test retains the 52.495345° full-run overshoot.
No dense case passes both geometry bounds. Dense radii grow
by 10.0–27.0 mm relative to full-run attenuation, and both
half-step angles fall below 30°. Initial attenuation changes
the collapse, while continued attenuation was also needed
for the earlier dense endpoint passes. All three invocations
fail. CUDA was untested and both temporary edits were restored.


#### Run the persistence matrices without editing the fixture

The sparse and dense column commands accept an optional endpoint
of 1.5, 3, or 6 seconds. An omitted endpoint selects 1.5 s.
Invalid endpoints and extra arguments return 2. The physical
bounds are identical at all endpoints.

```powershell
$bin = 'out/requalified'
foreach ($target in 'fluidity_m5_dry_scene_tests', 'fluidity_cuda_m5_dry_scene_tests') {
  & "$bin/$target.exe" --q-sand-column-long-settle 3
  & "$bin/$target.exe" --q-sand-dense-column-long-settle 3
  & "$bin/$target.exe" --q-sand-column-long-settle 6
}
```

The sparse 3 s run reproduces all four previous angles and
short drift values on native and SM120. The sparse native
6 s run returns 1 with angles 33.842809°, 23.424520°,
17.104058°, and 13.060319° in fixture order. Both default
dry-scene CTest entries pass. Q-SAND remains open.


On 4339f4c, the sparse SM120 command at 6 s also returns 1.
Its envelope angles are 33.842809°, 23.424520°, 17.104058°,
and 13.060318° in fixture order. Native and SM120 agree
within 0.000001° on the final envelope and last-second
drift. The fine half-step falls from 15.249811° to 13.060318°
over that last second while meeting the final 0.05 s drift
limit. The remaining failure reproduces on both paths.


#### Compare the compatible floor at six seconds

On cddc51d, run the existing aligned compatible-floor matrix
without changing the solver, scene bounds, or endpoint sampling.

```powershell
& ./out/requalified/fluidity_m5_dry_scene_tests.exe --q-sand-compatible-floor-long-settle all aligned 6
& ./out/requalified/fluidity_cuda_m5_dry_scene_tests.exe --q-sand-compatible-floor-long-settle all aligned 6
```

| Case | Native ordinary floor | Native compatible floor | Compatible floor absolute drift over last ≈1 s |
|---|---:|---:|---:|
| Coarse | 33.842809° | 31.267274° | 0.245937° |
| Fine | 23.424520° | 18.236314° | 1.364131° |
| Coarse half-step | 17.104058° | 17.135791° | 2.288315° |
| Fine half-step | 13.060319° | 13.385210° | 2.755325° |

The native compatible-floor matrix exits 1. Only the coarse
nominal angle meets the band. All four cases meet the radius,
short drift and kinetic-energy bounds. The fine half-step
keeps flattening despite the compatible transfer.


The parallel owning CUDA transfer candidate at a03f8ee
ran the same six-second command in 374 s of wall time,
while the original CUDA matrix was still running on the
same device. The optimized matrix exits 1. All four
geometry, short drift, and last-second drift lines match
the native output at printed precision. This duration is
a shared-device observation and is not a controlled
before-and-after speedup measurement.


#### Preserve total trace while attenuating horizontal APIC expansion

On 6a920d5, subtract one quarter of the positive horizontal
APIC trace from each horizontal diagonal and add their sum to
the vertical diagonal. Leave the off-diagonals and the rest
of the dry step unchanged. This preserves the full deposit
trace while redirecting horizontal expansion vertically.
Run the default native dry scene and the sparse 3 s command.

| Case | Envelope at 3 s | Radius (m) |
|---|---:|---:|
| Coarse | 38.826184° | 0.110178 |
| Fine | 38.634935° | 0.118343 |
| Coarse half-step | 44.440546° | 0.100108 |
| Fine half-step | 41.305530° | 0.109930 |

All four angles exceed 35°. The default coarse case also
fails at 58.566080° and 0.088590 m. Moving the removed
expansion into the vertical diagonal increases the
overshoot relative to horizontal attenuation alone.
CUDA and dense sampling were untested. The temporary
solver edit was restored.


#### Reproduce the dense baseline on SM120

On 31bfab4, run the CUDA dry-scene executable with
`--q-sand-dense-column-long-settle 3`. All four cases
match the earlier native dense heights, radii, angles,
short-window drift, and kinetic energy to the printed
precision. Both paths fail the angle and radius checks
in all four cases. The CUDA command exits 1. The sparse
and dense failures therefore both reproduce on SM120.


#### Apply granular return at the floor impact boundary

On e1b6b44, when swept contact occurs strictly inside the
step, return the pre-impact trial strain to the granular cone
before integrating the post-impact gradient. Carry the
returned plastic Jacobian into the final return. Keep all
transfer and contact impulses unchanged.

The default native dry scene passes. The sparse 3 s command
exits 1 with angles 30.519543°, 23.904277°, 22.853454°, and
20.811628°. The largest change from baseline is 0.000028°.
Returning at the actual contact split leaves the observed
flattening intact. CUDA and dense sampling were untested.
The solver edit was restored.


#### Refine the ordinary-floor timestep at fixed grid width

On a7bb264, repeat the native sparse 3 s command with the
two half-step durations halved again. Keep the nominal
cases and the same floor, material, sampling and bounds.

| Case | Quarter-step duration (s) | Half-step angle | Quarter-step angle | Quarter-step radius (m) | Quarter-step final ≈1 s drift |
|---|---:|---:|---:|---:|---:|
| Coarse | 0.00046875 | 22.853454° | 27.727291° | 0.140939 | 4.174450° |
| Fine | 0.000390625 | 20.811628° | 25.706297° | 0.156032 | 3.315365° |

Both angles increase by about 4.9° with this refinement
and remain below 30°. Both also lose more than 3° over
the last second while passing the final 0.05 s drift
check. The nominal cases reproduce baseline and the
matrix exits 1. This refinement does not establish a
settled angle. CUDA was untested and the fixture edit
was restored.


#### Check persistence at ordinary-floor eighth timesteps

On add9bf8, replace the half-step durations in the sparse
column fixture with 0.000234375 and 0.0001953125 s.
Keep the nominal comparison cases. Run the native column
commands at 3 and 6 s.

| Case | Angle at 3 s | Angle at 6 s | Radius at 6 s (m) | Last ≈1 s drift at 6 s |
|---|---:|---:|---:|---:|
| Coarse eighth-step | 34.222781° | 26.950165° | 0.141473 | 1.635993° |
| Fine eighth-step | 30.312137° | 24.968552° | 0.156011 | 1.528740° |

Both eighth-step cases enter the angle band at 3 s and
leave it by 6 s. Both still flatten during the final
second. The nominal controls reproduce baseline. Both
matrices exit 1. Further timestep refinement can shift
the transient angle without demonstrating repose. CUDA
was untested and the fixture edit was restored.


#### Predict the deposited granular stress at midstep

On 3f69e24, advance a temporary elastic trial by half a
step using the stored APIC slope. Apply the granular
return to evaluate the stress deposited on the ordinary
floor path. Keep the accepted state and final deformation
update on their original path. This is a stress-lag control
using the available particle slope.

The default native dry-scene test passes. The sparse 3 s
angles are 30.477668°, 23.974733°, 22.832948°, and
20.801352°. The largest angle change is 0.070484° and the
largest radius change is 0.343 mm. The extended command
exits 1. The diagnostic does not support this predictor
as a remedy for flattening. CUDA and dense sampling were
untested. The solver edit was restored.


#### Require persistence in the explicit long-horizon checks

The extended column commands at 3 and 6 s require the
envelope at the sample approximately one second before
the endpoint to be within 30–35°, and its change to the
endpoint to be at most 2°. This extends the existing
short drift tolerance to the reported longer window.
It measures two samples. Motion between them is unresolved.

On d891161 with this fixture change, native and SM120 give
the same four acceptance lines at 3 s. The coarse nominal
case passes the original endpoint checks but fails
persistence because its earlier angle is 29.062617°.
The other three fail angle and persistence. Both default
dry-scene executables pass.


#### Gather the deformation gradient at the trajectory midpoint

On 42260e0, sample the ordinary granular deformation gradient
halfway along the post-force PIC displacement for steps with
no swept floor hit. Use the same frozen grid velocities.
Retain the original transport, affine storage, and gradient
split at hits. Run the default native test and the sparse
three-second command.

The default test passes. At 3 s the four angles are 30.497792°,
23.898065°, 22.854290°, and 20.806276°. The largest change is
0.021751°. All four still fail persistence and the command
returns 1. Spatial centering in the frozen field leaves the
observed flattening intact. CUDA and dense sampling were
untested. The temporary source edit was restored.


#### Attenuate expansive volumetric strain before granular return

On 157abf7, remove the positive trace of the ordinary-floor
granular gradient from the incoming deformation increment,
using equal diagonal corrections. Preserve the deviatoric
gradient, post-impact gradient, transport and contact.
Compare full and half removal.

Full removal rejects every sparse case before the 0.75 s
sample and fails the default native dry scene. Half removal
passes the default test. At 3 s its angles are 33.382799°,
24.432601°, 25.197573°, and 19.566472°. Only the coarse
nominal case meets all checks. The other three miss angle,
radius and persistence. The half-removal command exits 1.
CUDA and dense sampling were untested. The solver was restored.


At 6 s with half removal, the coarse nominal case still
passes with a 31.896233° envelope, 0.480711° drift over
the final second, and 0.159309 m radius. The other cases
reach 22.472928°, 20.986038°, and 11.002120°, all with
radii above 0.16 m. The sparse native command exits 1.
The fine half-step loses another 2.312968° over the last
second and ends below the unmodified solver. The solver
edit was restored. CUDA and dense sampling were untested.


#### Measure plastic activity over the final second

On 157abf7, move only the native sparse column plastic
diagnostic window from the final 0.05 s to the steps starting
at or after 5 s. Run the unchanged solver to 6 s. The floor
layer consists of particles ending a step within 0.05 m of
the floor.

| Case | Projected particle-time | Plastic volume change particle-time | Mean absolute log Jpl path | Floor-layer contribution |
|---|---:|---:|---:|---:|
| Coarse | 3.17% | 1.59% | 0.00349686 | below 1e-14 |
| Fine | 4.76% | 4.76% | 0.01526072 | 0.01271852 |
| Coarse half-step | 10.62% | 10.62% | 0.01487805 | 0.01487805 |
| Fine half-step | 4.76% | 4.76% | 0.00632193 | 0.00632193 |

The two flattening half-step cases accumulate their measurable
late plastic volume path in the floor layer. The coarse
nominal case gains angle with almost no floor-layer path.
These counts locate activity and do not establish causation.
The command reproduces the baseline angles and exits 1.
CUDA was untested and the fixture edit was restored.


#### Retain particle velocity during the final second

On e0a0931, repeat the preceding native plastic diagnostic
and set particle_energy_blend to 0.6 on steps starting at
or after 5 s. Keep the original duration and wall settings.
The stored correction uses the existing history eligibility
and mass-weighted momentum correction.

Three cases reproduce the baseline angles and plastic
diagnostics at printed precision. The coarse half-step
angle falls from 17.104058° to 16.681018°, while its mean
absolute log Jpl path falls from 0.01487805 to 0.01417311.
Its final-second angle loss increases from 1.747565° to
2.170605°. The matrix exits 1. This late transfer change
reduces measured plastic volume activity in that case
without preserving pile height. CUDA was untested and the
temporary fixture edit was restored.


#### Advect with the retained residual outside floor contact

On bb4a0e7, repeat final-second velocity retention at energy
0.6. Add 0.6 times the stored residual correction to the PIC
advection speed when both the original and corrected sweeps
remain outside contact. Keep the stored velocity and the
deformation update on their original paths. This isolates
the advection part of the design's separation policy.

Three cases reproduce the native baseline at printed precision.
The coarse half-step reaches 16.810816°, compared with
16.681018° for stored-only retention and 17.104058° for the
unmodified solver. The coarse half-step loses 2.040808° over
the last second. The native matrix exits 1. CUDA was untested
and both temporary edits were restored.


#### Probe ordinary-floor force balance at six seconds

On 260b100, copy each native sparse column at 6 s. Clear
translation and APIC slopes on the copy, then advance one
ordinary step at the case's nominal duration, original
gravity, and floor friction. Divide the resulting stored
velocity by duration. All particles have equal mass. The
above-floor group starts the probe above 1e-8 m.

| Case | RMS acceleration (m/s²) | Mean vertical (m/s²) | Above-floor count | Above-floor mean vertical (m/s²) |
|---|---:|---:|---:|---:|
| Coarse | 0.297 | 0.053 | 2 | 1.663 |
| Fine | 1.279 | −0.185 | 3 | −3.882 |
| Coarse half-step | 1.112 | −0.193 | 2 | −6.077 |
| Fine half-step | 1.839 | −0.393 | 3 | −8.247 |

The half-step piles' few remaining elevated particles
accelerate downward at 62–84% of gravity from rest. This
probe includes the current stress, transfer, and contact.
It measures their combined support and does not identify
one term as the cause. All main trajectories reproduce the
baseline. The command exits 1, CUDA was untested, and the
temporary fixture edit was restored.


#### Inspect the stress carried above the floor

Repeat the preceding native probe and evaluate the accepted
Kirchhoff stress before clearing motion. Across the three
cases that fail angle at 6 s, all eight particles above
1e-8 m have both vertical and mean normal stress zero at
printed precision. Each accelerates downward in the probe.
The fine half-step apex has zero stress and accelerates
downward at 6.243399 m/s². The coarse nominal apex carries
−33.234156 Pa of vertical stress and accelerates upward at
1.598239 m/s². Its other elevated particle is stress-free
and also accelerates upward, so nearby particle stress can
support a particle with zero local stress.

The native command reproduces the baseline and exits 1.
CUDA was untested and the temporary fixture edit was restored.


#### Track the final apex stress through settlement

On fb92414, record vertical Kirchhoff stress at the first
accepted samples at or after 0.75, 1.5, 3, and 5 s. For
each native sparse case, trace the particle highest at 6 s
back through those samples. Negative values are compressive.

| Case | 0.75 s (Pa) | 1.5 s (Pa) | 3 s (Pa) | 5 s (Pa) | 6 s (Pa) |
|---|---:|---:|---:|---:|---:|
| Coarse | −32.63 | 0 | 0 | −21.08 | −33.23 |
| Fine | −69.16 | −36.28 | 0 | 0 | 0 |
| Coarse half-step | −35.43 | −19.41 | 0 | 0 | 0 |
| Fine half-step | −13.87 | −51.14 | −22.72 | 0 | 0 |

All three failing apex particles carry compression at 1.5 s.
Two lose it by the 3 s sample, while the fine half-step
retains it at 3 s and loses it by 5 s. The passing coarse
apex regains compression after the 3 s sample. These
snapshots do not resolve transitions between samples. The
native run reproduces the six-second geometry and exits 1.
CUDA was untested and the fixture edit was restored.


#### Resolve the final apex unloading time

On e1fb2b1, trace each particle on every accepted native
step through 6 s. For the particle highest at 6 s, report
its last step with vertical Kirchhoff stress below −1e-6 Pa
and its last step with a log Jpl increase above 1e-9.

| Case | Last compressed (s) | Last expansive return (s) |
|---|---:|---:|
| Coarse | 6.000000 | 3.757500 |
| Fine | 2.679688 | 6.000000 |
| Coarse half-step | 2.125313 | 6.000000 |
| Fine half-step | 3.653125 | 6.000000 |

The failing apex particles remain unloaded after their
last compressive step and still undergo expansive return
at 6 s. In the specified cohesionless return, a positive
plastic log-volume increment implies a positive trial
elastic log Jacobian. This identifies continued expansion
but does not isolate its source in transfer or contact.
The native command reproduces baseline and exits 1. CUDA
was untested and the temporary fixture edit was restored.


#### Compare compatible-floor work receipts

For the six-second compatible-floor matrix above, native and
parallel SM120 agree at printed precision on both reported work
quantities. The original serial SM120 run also matches the first
three completed cases in fixture order (coarse, coarse half-step,
fine). Its fine half-step result was pending at this comparison.

| Case | Cut stress work (J) | Wall particle kinetic change (J) |
|---|---:|---:|
| Coarse | −0.707868066 | −3.12828179 |
| Coarse half-step | −0.0830633094 | −2.20779397 |
| Fine | −0.505290049 | −2.67737705 |
| Fine half-step | −0.0839133049 | −1.88997347 |

The reported cut work is separate from Q-SOLID's uncut identity.
The parallel change reproduces this work receipt along with the
failing pile geometry. It does not resolve the cut work imbalance.


#### Attenuate expansive strain only during late settlement

On 7826560, repeat the half-removal expansive strain control
only on ordinary granular steps starting at or after 3 s.
Use the native six-second sparse command and its original
acceptance criteria.

| Case | Baseline angle | Late-only angle | Baseline mean log Jpl | Late-only mean log Jpl |
|---|---:|---:|---:|---:|
| Coarse | 33.842809° | 33.842839° | 0.309825 | 0.286393 |
| Fine | 23.424520° | 23.424520° | 0.250506 | 0.229187 |
| Coarse half-step | 17.104058° | 17.104058° | 0.177330 | 0.146114 |
| Fine half-step | 13.060319° | 13.098997° | 0.182548 | 0.170024 |

Mean log Jpl decreases in every case while the angles change
by at most 0.039° and radii agree at printed precision. The
native matrix exits 1. Reducing late expansive strain after
the pile has formed has little effect on geometry in this
control. CUDA was untested and the temporary edits were restored.


#### Attenuate expansive strain only during collapse

On 7826560, apply the same half removal only before the
accepted 0.75 s sample. At 6 s, the native sparse angles
are 31.949541°, 22.438349°, 20.188247°, and 10.066382°.
The coarse nominal case passes. The other three fail
angle, radius, and persistence. The matrix exits 1.

The collapse-only control reproduces the full-run angles
within 0.94° and the radii within 3 mm. The late-only control
changes the baseline geometry much less. The large change
in radius therefore originates mainly during formation for
this control. CUDA was untested and both edits were restored.


#### Shift the vertical grid origin for sparse three-second piles

On 10d1333, repeat the existing floor-grid-phase command
with 63 particles and a 3 s endpoint. Keep h/80 and the
physical floor, drop and material unchanged.

| Grid width (m) | Aligned | Quarter-cell shift | Half-cell shift |
|---|---:|---:|---:|
| 0.075 | 22.853454° | 23.808758° | 20.100073° |
| 0.0625 | 20.811628° | 21.964544° | 19.457961° |

The maximum span is 3.71°. The finer pile is shallower at
every tested alignment and all six angles miss the band.
The native command exits 1. The test varies alignment for
the entire trajectory, so it does not isolate first impact.
CUDA was untested and the fixture edit was restored.


#### Return to an aligned grid after collapse

On 0abeeb7, repeat the sparse phase matrix with a shifted
grid through the accepted 0.75 s sample. Recreate the
aligned grid at that boundary and carry over the particle
state without remapping it. Advance to 3 s.

| Grid width (m) | Aligned | Quarter shift before 0.75 s | Half shift before 0.75 s |
|---|---:|---:|---:|
| 0.075 | 22.853454° | 23.847315° | 22.404500° |
| 0.0625 | 20.811628° | 22.963537° | 21.012102° |

All cases miss the angle band. Holding a half-cell shift
through the full run gives lower angles than switching
back to aligned by 2.30° on coarse and 1.55° on fine.
The post-collapse grid therefore contributes to the
full-run phase effect. The native command exits 1.
CUDA was untested and the fixture edit was restored.


#### Shift the grid after the pile forms

On bc8e93b, start all sparse half-step runs with the aligned
grid. At the accepted 0.75 s sample, recreate the grid with
the specified vertical offset and carry the particle state
unchanged. Continue to 3 s.

| Grid width (m) | Aligned | Quarter shift after 0.75 s | Half shift after 0.75 s |
|---|---:|---:|---:|
| 0.075 | 22.853454° | 22.763191° | 20.601170° |
| 0.0625 | 20.811628° | 20.297780° | 19.466541° |

The 0.75 s angles agree within each width. The late half-cell
shift lowers the final angle by 2.25° on coarse and 1.35° on
fine. The fine result is within 0.009° of the full-run shift,
so this part of the phase effect does not require changing
the collapse. The native command exits 1. CUDA was untested
and the fixture edit was restored.


#### Alternate vertical grid alignment after collapse

On 7494211, repeat the native sparse half-step phase matrix
with an aligned grid through the 0.75 s sample. After that
sample alternate the aligned and shifted grids by step
parity. Carry the particle state unchanged between grids.

| Grid width (m) | Aligned | Alternate quarter shift | Alternate half shift | Fixed late half shift |
|---|---:|---:|---:|---:|
| 0.075 | 22.853454° | 22.848691° | 21.668147° | 20.601170° |
| 0.0625 | 20.811628° | 20.576246° | 20.059458° | 19.466541° |

The half-cell alternation finishes between the aligned and
fixed late shift in both cases. It recovers part of the
angle lost with the fixed shift, and all cases remain below
30°. The 0.75 s samples match the baseline. The native
command exits 1. CUDA was untested and the fixture edit
was restored.


#### Extend the normalized stress and gradient pair to six seconds

On 78381d0, use the normalized weight gradients from the
compatible stress scatter for deformation, keeping the
complete ghost APIC fit and the original swept contact.
Use the same gradient for the particle-side cut work.
Run the native command with fine-half aligned 6.

The default dry-scene test passes. The six-second angle
is 13.182260° versus 13.385210° with the complete ghost
gradient. The paired case loses 2.790216° over the last
second and exits 1. Its cut stress work residual is
2.92e-17 J versus −0.0839133049 J before pairing. Pairing
closes the reported spatial work balance while leaving
late flattening. CUDA and the other cases were untested.
The temporary solver edit was restored.


#### Average granular volumetric strain within grid cells

On d3a249a, group ordinary-floor granular particles by their
start cell on the aligned grid. Replace the gathered velocity
gradient trace with the reference-volume-weighted trace in that
cell. Use equal diagonal corrections to preserve the deviator.
Use this gradient in the existing swept split. Transfer,
stress scatter and the granular return remain unchanged.

The default native dry scene passes. The sparse 3 s angles are
29.620245°, 22.485443°, 20.685610°, and 20.152352°. All four
angles decrease relative to the baseline, by up to 2.17°.
All pass the radius and short-drift checks and fail angle and
persistence. The native matrix exits 1. CUDA and dense
sampling were untested. The temporary solver edit was restored.


#### Reduce symmetric shear in the incoming deformation

On cb653b1, halve the symmetric off-diagonal part of the
ordinary-floor granular gradient in the incoming deformation
increment. Preserve its skew part and all diagonal entries.
Keep stress scatter, the post-contact gradient, advection,
and affine storage on their original paths.

The default native dry-scene test passes. The sparse 3 s
angles are 30.450681°, 24.028899°, 23.551505°, and 21.751824°.
The half-step angles gain 0.70° and 0.94°, but all four cases
still fail persistence and the matrix exits 1. This control
leaves the volume rate unchanged and does not resolve repose.
CUDA and dense sampling were untested. The edit was restored.


#### Suppress late yielding on the ordinary floor

On 33ee019, repeat the six-second native sparse matrix with
translated cohesion set to 10 on steps starting at or after
3 s. Preserve prior state and other settings. This diagnostic
moves the yield surface away and changes the admissible
tensile response. It is outside the cohesionless gate.

| Case | Baseline angle at 6 s | Yield suppression angle at 6 s | Final ≈1 s drift |
|---|---:|---:|---:|
| Coarse | 33.842809° | 21.872167° | 4.332104° |
| Fine | 23.424520° | 9.672184° | 5.713894° |
| Coarse half-step | 17.104058° | 12.625588° | 3.522127° |
| Fine half-step | 13.060319° | 10.823780° | 3.354686° |

Plastic volume stops accumulating after the switch to the
reported precision. All four angles decrease, even though
radii stay close to baseline. This ordinary-floor result
differs from the earlier compatible-floor candidate's
response to late yield suppression. It does not isolate
the effects of shear return from tensile stress. The
native command exits 1. CUDA was untested and the fixture
was restored.


#### Suppress only the compressive shear return during late settlement

On 33ee019, disable deviator projection for compressive
granular trials beginning at 3 s. Keep the cohesionless
tensile apex and the original grid and wall path. The
diagnostic uses a distinct host-only return function
and admits stresses outside the specified yield cone.

| Case | Baseline angle at 6 s | No late shear return |
|---|---:|---:|
| Coarse | 33.842809° | 36.473024° |
| Fine | 23.424520° | 23.424520° |
| Coarse half-step | 17.104058° | 17.104058° |
| Fine half-step | 13.060319° | 13.113389° |

The three shallow cases change by at most 0.054°. The
coarse nominal case overshoots the upper bound. Unlike
raising cohesion, suppressing only the compressive shear
return does not further flatten all four cases. The native
command exits 1. CUDA was untested and edits were restored.


#### Deposit only the skew APIC slope on the ordinary floor

On 50ce28b, replace the granular APIC slope at deposit with
its skew part on the ordinary floor. Leave fitted slope
storage, deformation, stress and contact unchanged.

| Case | Angle at 3 s | Radius (m) |
|---|---:|---:|
| Coarse | 31.113488° | 0.092979 |
| Fine | 31.409299° | 0.099825 |
| Coarse half-step | 48.244810° | 0.087905 |
| Fine half-step | 46.243948° | 0.092843 |

Every radius falls below 0.10 m. Both half-step angles
overshoot and the default native scene fails at 61.115376°.
The sparse matrix exits 1. Retaining rotation alone removes
too much spread in this control. CUDA and dense sampling
were untested. The temporary source edit was restored.

#### Remove symmetric off-diagonal shear from granular APIC deposition

On e58302b, subtract half the sum of each transposed off-diagonal
pair from both entries of that pair in the ordinary-floor granular
APIC deposit slope. This preserves all three diagonal entries and
the skew part. Use the original slope for storage and leave
deformation, stress and contact on their original paths.

Build the native dry-scene target, run it without arguments, then
run it with `--q-sand-column-long-settle 3`. The default angle
is 29.691360° and the default command exits 1.

| Case | Angle at 3 s | Radius (m) |
|---|---:|---:|
| Coarse | 28.357544° | 0.149164 |
| Fine | 20.920179° | 0.159373 |
| Coarse half-step | 20.810808° | 0.153329 |
| Fine half-step | 17.355195° | 0.172360 |

All four cases fail angle and persistence. The fine half-step
also exceeds the radius limit. The native matrix exits 1.
Removing off-diagonal shear while retaining normal affine strain
does not reproduce the steepening from rotation-only deposition.
CUDA and dense sampling were untested. The source edit was restored.

#### Separate vertical and horizontal normal strain in granular APIC deposition

On 75de2ef, replace the diagonal of the ordinary-floor granular
APIC deposit slope by its mean diagonal entry. In the horizontal
control, average only the xx and zz entries and preserve yy.
Both controls preserve the trace and the full off-diagonal part.
Use the original slope for storage, with the original deformation,
stress and contact updates.

Run the native dry-scene executable without arguments and with
`--q-sand-column-long-settle 3`. The horizontal-only control
passes the default test at 32.770452°. Averaging all three
entries fails the default test at 60.859696° and 0.086553 m.

| Case | Horizontal mean angle | Horizontal mean radius (m) | Full mean angle | Full mean radius (m) |
|---|---:|---:|---:|---:|
| Coarse | 30.349030° | 0.137801 | 34.523784° | 0.103113 |
| Fine | 23.380153° | 0.148390 | 33.175054° | 0.118203 |
| Coarse half-step | 22.692118° | 0.144055 | 48.336718° | 0.095580 |
| Fine half-step | 20.128799° | 0.157242 | 45.328855° | 0.108407 |

Both sparse matrices exit 1 and every case fails persistence.
The steepening seen with rotation-only deposition also occurs
with the full diagonal mean, which retains trace and shear.
Averaging the horizontal entries alone has a much smaller effect.
CUDA and dense sampling were untested. Both source edits were restored.

#### Separate collapse and settlement in granular APIC diagonal averaging

On 2d068b4, repeat the ordinary-floor sparse six-second matrix.
Immediately before each step, average the three diagonal entries
of the input affine slopes, preserving trace and off-diagonal
entries. Apply this change either on steps starting before
0.75 s or on steps starting at or after 0.75 s. Use the
unmodified solver, initial conditions and acceptance checks.
Build the native dry-scene executable and run it with
`--q-sand-column-long-settle 6` for each time window.

| Case | Baseline at 6 s | Before 0.75 s only | From 0.75 s only |
|---|---:|---:|---:|
| Coarse | 33.842809° | 29.946711° | 33.316463° |
| Fine | 23.424520° | 22.903833° | 21.892961° |
| Coarse half-step | 17.104058° | 24.077650° | 16.984032° |
| Fine half-step | 13.060319° | 22.688298° | 14.306082° |

Early averaging produces 60.86°, 58.93°, 66.83° and 65.62°
at 0.75 s in the same order as the table. By six seconds
those piles have fallen below 30°. Late averaging starts
from the baseline 0.75 s state and misses the final angle
band in three cases. The coarse case ends in band but
drifts 1.514891° in the final approximately one-second
window. Both commands exit 1. The early control raises
the two half-step final angles by 6.97° and 9.63°, while
late averaging changes them by at most 1.25°. The fixture
edit was restored. CUDA and dense sampling were untested.

#### Separate floor and elevated stress in the six-second apex support probe

On 555595b, copy each ordinary-floor sparse state at six seconds
and clear translation and affine slopes. Identify the highest
particle in the original state. For each copy, keep all elastic
deformations, keep only those starting at y <= 1e-8 m, keep only
those above that height, or reset them all. Set each discarded
elastic deformation to the identity and preserve plastic volume,
positions, and mass. Advance each copy one step with the original
duration, gravity, floor friction and acoustic CFL. Divide the
apex's vertical stored velocity by duration. Run the native
dry-scene executable with `--q-sand-column-long-settle 6`.

| Case | All stress (m/s²) | Floor only (m/s²) | Elevated only (m/s²) | Zero stress (m/s²) |
|---|---:|---:|---:|---:|
| Coarse | 1.598239 | 0.890146 | -9.101908 | -9.810000 |
| Fine | -0.328745 | -0.328745 | -9.810000 | -9.810000 |
| Coarse half-step | -4.654775 | -4.654775 | -9.810000 | -9.810000 |
| Fine half-step | -6.243399 | -6.243399 | -9.810000 | -9.810000 |

Floor particles provide all measured stress support for the
three shallow apex particles. That support ranges from 3.57 to
9.48 m/s² against 9.81 m/s² gravity. The coarse nominal
apex also receives 0.71 m/s² from elevated stress. The probe
measures the response of the existing discrete path from rest.
The four original angles reproduce the baseline and the command
exits 1. CUDA and dense sampling were untested. The fixture
edit was restored.

#### Offset ordinary-floor granular contact by particle reference spacing

On 65c7cd0, pass a per-particle floor height of
`0.25 * cbrt(reference_volume)` to the ordinary-floor granular
sweep. Leave the grid origin, transfer, stress scatter,
compatible-floor path, snow path and angle estimator unchanged.
The fixture still measures height from the original physical
plane, so its angle includes the added clearance.

| Case | Sparse 6 s angle | Dense 3 s angle | Dense radius (m) |
|---|---:|---:|---:|
| Coarse | 19.022598° | 28.581353° | 0.170913 |
| Fine | 5.153735° | 26.334554° | 0.182571 |
| Coarse half-step | 17.183332° | 22.616351° | 0.173931 |
| Fine half-step | 13.543265° | 21.343035° | 0.186041 |

Build the native dry-scene target and run it without arguments,
with `--q-sand-column-long-settle 6`, and with
`--q-sand-dense-column-long-settle 3`. All three commands exit
1. The default run exits before scene output. Both matrices
fail every angle and persistence check, and the dense matrix
also fails every radius check. This offset does not produce
a stable pile. CUDA was untested and the source edit was restored.

#### Match the granular deformation increment Jacobian to the velocity trace

On 950cfc4, for each incoming and projected granular increment
with positive Euler Jacobian, rescale the Euler trial matrix
uniformly by the cube root of exp(dt tr L) / det(I + dt L).
This matches the constant-gradient volume factor while
preserving the Euler trial stretch ratios. The nonpositive
Jacobian path, the return, transfer, contact and snow are
unchanged. Build the native dry-scene target and run it without
arguments and with `--q-sand-column-long-settle 6`.

| Case | Baseline at 6 s | Matched Jacobian at 6 s |
|---|---:|---:|
| Coarse | 33.842809° | 33.953710° |
| Fine | 23.424520° | 23.665081° |
| Coarse half-step | 17.104058° | 17.088157° |
| Fine half-step | 13.060319° | 13.113653° |

The default test passes. The six-second matrix exits 1 and
the three shallow piles remain below 30°. The maximum angle
change is 0.241°. The Euler increment's volume error therefore
does not account for the shallow angles in these native sparse
cases. CUDA and dense sampling were untested. The temporary
source edit was restored.

#### Include the tangential effect of a varying floor arrival time

On 0fb2dd3, after the usual granular sweep split, apply an
additional increment I + a outer b only for interior impacts
with incoming normal velocity vy < 0. Here a is the change in
tangential advection velocity across impact divided by vy, with
zero normal component. The row b is e_y plus the incoming
duration times the normal row of the gathered velocity gradient.
The correction approximates the tangential arrival-time term
as a left-composed increment. It preserves its own determinant
at first order only when a dot b vanishes. The normal endpoint,
contact response, scatter and constitutive return are unchanged.

Build and run the native dry-scene executable without arguments
and with `--q-sand-column-long-settle 6`. The default passes
at 33.563096°. The matrix exits 1.

| Case | Baseline at 6 s | Arrival correction at 6 s |
|---|---:|---:|
| Coarse | 33.842809° | 33.578260° |
| Fine | 23.424520° | 24.637866° |
| Coarse half-step | 17.104058° | 19.854237° |
| Fine half-step | 13.060319° | 18.607054° |

The half-step angles improve by 2.75° and 5.55° but remain
below 30°. This probes the tangential term without using the
singular normal derivative of the clamped endpoint as a material
deformation map. CUDA and dense sampling were untested.
The temporary source edit was restored.

#### Add the tangential arrival term to the accepted deformation

On d522b8e, use the preceding arrival vectors a and b but
add their outer product times the accepted elastic deformation
to the sweep-split trial, instead of left-composing the
correction with the trial. Keep the normal row unchanged.
In a second run, also average the three diagonal APIC entries
before each ordinary-floor granular step starting before 0.75 s,
as in the earlier collapse-only control.

Build the native dry-scene target and run it with
`--q-sand-column-long-settle 6`. The additive term alone also
passes the no-argument run at 33.555075°.

| Case | Additive term | Combined controls | Combined final drift |
|---|---:|---:|---:|
| Coarse | 33.530778° | 28.441280° | 0.212546° |
| Fine | 24.565894° | 24.032534° | 0.838778° |
| Coarse half-step | 19.849614° | 24.706440° | 0.886700° |
| Fine half-step | 18.603412° | 24.135115° | 0.995536° |

The additive term differs from the previous composed control
by less than 0.073° in every case. Combining it with the
collapse control keeps all radii and final drift within their
limits, but every angle falls below 30°. Both matrices exit 1.
CUDA and dense sampling were untested. Both edits were restored.

#### Repeat the combined arrival and collapse controls at dense sampling

Use the same native executable as the preceding combined control,
with additive tangential arrival correction and diagonal APIC
averaging before 0.75 s. Run `--q-sand-dense-column-long-settle 3`
with 504 particles. Compare with the unmodified dense baseline.

| Case | Baseline angle | Combined angle | Baseline radius (m) | Combined radius (m) | Combined final drift |
|---|---:|---:|---:|---:|---:|
| Coarse | 29.703637° | 27.382271° | 0.170290 | 0.166422 | 0.048279° |
| Fine | 26.954622° | 27.203813° | 0.182075 | 0.175204 | 0.523100° |
| Coarse half-step | 21.853351° | 23.608954° | 0.172975 | 0.179292 | 4.673116° |
| Fine half-step | 20.623778° | 22.939049° | 0.185666 | 0.188458 | 2.636972° |

All four dense angles and radii miss their bounds and the
command exits 1. Both half-step radii increase, and both
half-step drifts exceed one degree over the final approximately
one-second window. The sparse radius and drift improvements
therefore do not hold at dense sampling. CUDA was untested.

#### Check dense surface slopes at fixed physical radii

On 7b0dace, run the unmodified native dense trajectory with
`--q-sand-dense-column-long-settle 3`. At the final state,
call the existing quadrant fitter with reference radii
0.12, 0.14 and 0.16 m as well as the measured maximum radius.
In each fit, six bins span 15–85% of the supplied radius.
The table gives the minimum and maximum across all four
resolved quadrants. The extra fits do not change acceptance.

| Case | 0.12 m | 0.14 m | 0.16 m | Measured maximum radius |
|---|---:|---:|---:|---:|
| Coarse | −1.21–3.81° | 35.94–36.01° | 19.72–23.89° | 31.16° |
| Fine | −11.06–−5.23° | 19.40° | 26.94–31.15° | 43.48–43.90° |
| Coarse half-step | 0.28–2.85° | −2.38–4.41° | 4.90–7.05° | 14.44° |
| Fine half-step | −14.92–−13.24° | 12.30° | −2.06–2.03° | 18.08° |

The two half-step cases remain below 30° for every fit, including
fits that omit the outer edge. The nominal cases vary strongly
with the interval and give no consistent slope across scales.
All four original gate results reproduce the native baseline
and the command exits 1. The supplemental fit was not run on
CUDA. The temporary fixture edit was restored.

#### Hold normal speed constant in the post-impact Coulomb derivative

On 2a705d8, zero the gathered normal row in a copy supplied
to the post-impact Coulomb gradient on the ordinary granular
floor. The incoming gradient and normal speed used for the
Coulomb cap are unchanged. This removes the derivative of
the cap with respect to normal speed while preserving the
tangential speed derivative and the stick/slip decision.
Build the native dry-scene target and run it without arguments
and with `--q-sand-column-long-settle 6`.

| Case | Baseline at 6 s | Frozen normal speed at 6 s |
|---|---:|---:|
| Coarse | 33.842809° | 33.610925° |
| Fine | 23.424520° | 23.190354° |
| Coarse half-step | 17.104058° | 17.104058° |
| Fine half-step | 13.060319° | 13.060319° |

The default test passes at 32.791300°. The matrix exits 1.
Both half-step angles match baseline at printed precision.
The nominal angles change by less than 0.235°. This term
alone does not explain the timestep sensitivity in the
sparse pile. CUDA and dense sampling were untested.
The temporary source edit was restored.

#### Replace low-mass face velocities in granular deformation only

On 288f960, for each ordinary-floor granular stencil sample
with deposited face mass below the existing force-mass floor,
replace that sample's velocity by the particle PIC velocity
component in the deformation gradient only. Apply the change
as a velocity difference times the existing weight gradient.
Leave PIC transport, affine storage, force and contact unchanged.

Build the native dry-scene executable and run it without
arguments and with `--q-sand-column-long-settle 6`.
The default passes at 32.826018°. The matrix exits 1
with angles 33.843215°, 23.421113°, 17.104206° and
13.060128° in coarse, fine, coarse half-step and fine
half-step order. The largest change from baseline is
0.003407°. The three shallow cases still miss the angle
band. CUDA and dense sampling were untested. The temporary
source edit was restored.

#### Include gravity on empty vertical faces in the ordinary gather

On 160298e, set the post-force velocity of zero-deposited-mass
vertical faces to dt times gravity on the ordinary granular
path. Keep stress omission, occupied faces, old grid velocity,
and all other paths unchanged. Build both dry-scene targets.
Run the native executable with `--q-sand-column-long-settle 6`
and the SM120 executable without arguments.

All four native angles reproduce the baseline to six printed
decimal places. The native command exits 1. The SM120 default
passes at the baseline 32.825896°. Empty-face gravity is ruled
out as a cause of the measured native sparse angle deficit at
this precision. The CUDA long-window matrix and dense sampling
were untested for this control. The source edit was restored.

#### Release the granular column with isotropic overburden stress

On 11014d1, initialize the six-second native sparse fixture
with isotropic compressive Kirchhoff stress of magnitude
rho g times depth below the column top. Use diagonal elastic
stretch exp(-pressure / (3 lambda + 2 mu)) and initialize
log J_pl to minus three times its log so the total Jacobian
starts at one. Keep the release height, positions, mass,
friction and update unchanged. This is an initial-stress
sensitivity control. The released column has free sides,
so isotropic compression does not put it in equilibrium.

Build the native dry-scene target and run it with
`--q-sand-column-long-settle 6`. The angles at 0.75 s are
14.543956°, 9.874377°, 18.795867° and 10.695996° in coarse,
fine, coarse half-step and fine half-step order. At six
seconds they are 0°, 0°, 3.042217° and 0.776028°. Every case
fails angle and radius. The command exits 1. The initial
isotropic compression increases spreading substantially
under this release, so it does not resolve Q-SAND. CUDA
and dense sampling were untested. The fixture edit was restored.

#### Retain incident rotation with resident floor compression

On e31cc4f, at ordinary-floor zero-fraction granular contacts,
keep the compressive part of incoming yy and retain the
incoming skew part of the xy and zy pairs. Keep the symmetric
off-diagonal entries of the projected gradient and leave
positive-time impacts, the particle sweep and stress unchanged.
Build the native dry-scene target. Run without arguments and
with `--q-sand-column-long-settle 6` and
`--q-sand-dense-column-long-settle 3`.

| Case | Sparse angle at 6 s | Sparse final drift | Dense angle at 3 s | Dense radius (m) |
|---|---:|---:|---:|---:|
| Coarse | 35.428256° | 0.727771° | 34.332577° | 0.174015 |
| Fine | 30.203405° | 0.142391° | 29.557011° | 0.183271 |
| Coarse half-step | 33.318075° | 0.812641° | 26.607787° | 0.174958 |
| Fine half-step | 29.258834° | 0.368356° | 24.929450° | 0.185992 |

The default test passes at 34.841448°. All sparse radii
and final drift magnitudes pass, but two sparse angles
miss the band. All dense radii exceed 0.16 m and three
dense angles miss the band. Both matrices exit 1.
CUDA was untested and the temporary source edit was restored.

#### Taper positive horizontal APIC expansion across the bottom cell

On 62d518f, reduce positive horizontal trace in the ordinary
granular APIC deposit by half at the floor, tapering the
reduction linearly to zero at one cell width above it.
Apply the correction equally to xx and zz. Keep the remaining
deposit, deformation and contact on their original paths.

| Case | Sparse angle at 6 s | Dense angle at 3 s | Dense radius (m) |
|---|---:|---:|---:|
| Coarse | 36.693231° | 31.453640° | 0.170684 |
| Fine | 26.959741° | 28.300727° | 0.180841 |
| Coarse half-step | 20.732937° | 23.806462° | 0.172571 |
| Fine half-step | 14.997625° | 22.304677° | 0.185392 |

Build the native dry-scene target and run it without arguments,
with `--q-sand-column-long-settle 6`, and with
`--q-sand-dense-column-long-settle 3`. All exit 1. The default
angle is 36.034513°. Dense radii change by at most 1.24 mm and
all still exceed 0.16 m. This floor-local control was tested
alone. CUDA was untested and the temporary source edit was restored.

#### Attenuate positive horizontal APIC expansion above the floor

On 90dca92, reduce positive horizontal trace of the ordinary
granular APIC deposit by half above one cell width. Taper
the reduction linearly to zero at the floor. Apply the
correction equally to xx and zz. This is complementary
to the preceding bottom-cell control. Keep deformation,
contact, and the other transfer components unchanged.

| Case | Sparse angle at 6 s | Dense angle at 3 s | Dense radius (m) | Dense final drift |
|---|---:|---:|---:|---:|
| Coarse | 42.258354° | 33.666166° | 0.147836 | 0.245770° |
| Fine | 37.882920° | 31.916787° | 0.155212 | 0.201096° |
| Coarse half-step | 24.549948° | 33.525692° | 0.134823 | 7.209156° |
| Fine half-step | 25.782893° | 31.075900° | 0.135353 | 6.892670° |

Build the native dry-scene target and run it without arguments,
with `--q-sand-column-long-settle 6`, and with
`--q-sand-dense-column-long-settle 3`. All exit 1. The default
angle is 50.678178°. Dense radii decrease by 22.5–50.3 mm and
all four dense angles and radii pass at 3 s, while both dense
half-step cases fail persistence. The sparse cases all miss
the angle band at 6 s. CUDA was untested and the source edit
was restored.

#### Combine above-floor affine attenuation and resident compression with rotation

On 3c582c1, apply both preceding controls from the sections
on above-floor APIC expansion and incident rotation with
resident compression. The deposit attenuation is unchanged
and the deformation correction applies only at zero-fraction
ordinary-floor granular hits. Build the native dry-scene
target and run without arguments, with
`--q-sand-column-long-settle 6`, and with
`--q-sand-dense-column-long-settle 3`.

| Case | Sparse angle | Sparse final drift | Dense angle | Dense radius (m) | Dense final drift |
|---|---:|---:|---:|---:|---:|
| Coarse | 42.163343° | 0.800365° | 37.025709° | 0.156213 | 0.409451° |
| Fine | 37.375190° | 0.536487° | 35.655646° | 0.163239 | 0.813339° |
| Coarse half-step | 33.766763° | 0.202970° | 35.132700° | 0.137850 | 6.222385° |
| Fine half-step | 35.735615° | 0.205914° | 32.966197° | 0.148634 | 5.739108° |

All three commands exit 1. The default angle is 51.541239°.
Every sparse final drift is below one degree, but three
angles overshoot. Dense half-step drift remains above five
degrees. The combined control therefore does not address
the dense persistence failure. CUDA was untested and the
temporary source edit was restored.

#### Extend ordinary-floor sparse settlement to twelve seconds

On 92b3d73, add a 12 s endpoint to the ordinary column fixture
and run the native executable with `--q-sand-column-long-settle 12`.
Keep the production solver, initial state, sampling, and acceptance
limits. Compare against the recorded 6 s baseline.

| Case | Angle at 6 s | Angle at 12 s | Final ≈1 s drift | Radius at 12 s (m) |
|---|---:|---:|---:|---:|
| Coarse | 33.842809° | 33.955949° | 0.231448° | 0.137319 |
| Fine | 23.424520° | 24.353065° | 0.081166° | 0.146767 |
| Coarse half-step | 17.104058° | 2.304259° | 3.401364° | 0.143950 |
| Fine half-step | 13.060319° | 0.000000° | 0.407319° | 0.155423 |

The coarse nominal case remains in range and changes by 0.113°
between endpoints. Both half-step piles lose nearly all their
height, while their radii change by less than 0.3 mm from the
six-second baseline. This confirms that the late half-step
angle loss is vertical collapse at essentially fixed radius.
The matrix exits 1. CUDA and dense sampling were untested.
The temporary fixture edit was restored.

#### Extend resident floor compression to twelve seconds

On 39fe7a6, retain min(0, incoming yy) in the projected
gradient for ordinary granular floor hits at zero fraction.
Use the same sparse 12 s endpoint as the preceding baseline.
Build the native dry-scene target and run
`--q-sand-column-long-settle 12`.

| Case | Compression angle at 6 s | Compression angle at 12 s | Final ≈1 s drift | Radius at 12 s (m) |
|---|---:|---:|---:|---:|
| Coarse | 36.442893° | 36.295710° | 0.001830° | 0.142033 |
| Fine | 30.159032° | 29.959037° | 0.001188° | 0.147397 |
| Coarse half-step | 34.609817° | 35.255504° | 0.266517° | 0.142443 |
| Fine half-step | 30.112880° | 28.195258° | 0.216714° | 0.152230 |

The unmodified half-step piles flatten to 2.304259° and 0°.
With resident compression both retain substantial height,
and all four final drift magnitudes are below one degree.
All four angles miss the 30–35° band and the matrix exits 1.
The 6 s fine angle passes therefore do not persist through
12 s. CUDA and dense sampling were untested at this horizon.
The temporary solver and fixture edits were restored.

#### Reproduce resident floor compression on SM120 and at dense native sampling

On 6439464, repeat the preceding resident compression edit and
12 s endpoint. Build both dry-scene targets. Run the CUDA target
with `--q-sand-column-long-settle 12` and the native target with
`--q-sand-dense-column-long-settle 6`. Both commands exit 1.

| Case | Sparse SM120 angle at 12 s | Dense native angle at 6 s | Dense native radius (m) | Dense native final drift |
|---|---:|---:|---:|---:|
| Coarse | 36.295710° | 33.035818° | 0.172380 | 0.301150° |
| Fine | 29.959547° | 27.953610° | 0.182183 | 2.063976° |
| Coarse half-step | 35.255504° | 30.757054° | 0.173835 | 1.157020° |
| Fine half-step | 28.195258° | 27.423564° | 0.184981 | 0.340904° |

The CUDA sparse cases agree with native within 0.000510° and
all four miss the angle band. Every dense native radius
exceeds 0.16 m. Two dense cases miss the angle band and two
exceed one degree of final drift. Dense CUDA was untested.
The temporary solver and fixture edits were restored.

#### Combine resident floor compression with volumetric APIC attenuation

On bb2d77d, retain min(0, incoming yy) in the projected gradient
at zero-fraction ordinary-floor granular hits. At deposit,
subtract one quarter of max(0, min(horizontal trace, total trace))
from each horizontal diagonal entry. Build the native dry-scene
target. Run without arguments, with `--q-sand-column-long-settle 6`,
and with `--q-sand-dense-column-long-settle 3`.

| Case | Sparse angle at 6 s | Dense angle at 3 s | Dense radius (m) |
|---|---:|---:|---:|
| Coarse | 35.891483° | 36.721958° | 0.165004 |
| Fine | 32.176903° | 33.042230° | 0.167704 |
| Coarse half-step | 35.369021° | 29.215302° | 0.162576 |
| Fine half-step | 32.859685° | 27.710259° | 0.173388 |

All three commands exit 1. The default angle is 41.713225°.
The sparse fine cases pass, while both coarse angles overshoot.
All dense radii exceed 0.16 m despite decreasing relative to
the baseline. Dense half-step final drift is below one degree
in both cases, but neither reaches 30°. CUDA was untested and
the experimental source edit was restored.

#### Separate volumetric and deviatoric resident floor compression

On 9eaf5c9, decompose the zero-fraction ordinary granular floor
compression rate q = min(0, incoming yy). In one build add
diag(-q/2, q, -q/2) to the projected gradient. In the other
build add diag(q/3, q/3, q/3). The first control preserves
the gradient trace and the second adds only volumetric
compression. Both retain the original contact sweep and
stress scatter. Run the native dry-scene target without
arguments and with `--q-sand-column-long-settle 6`.

| Case | Trace-preserving angle | Volumetric angle |
|---|---:|---:|
| Coarse | 34.023226° | 34.473953° |
| Fine | 23.585661° | 27.146366° |
| Coarse half-step | 17.710028° | 28.981326° |
| Fine half-step | 13.872172° | 25.044496° |

Both default runs pass and both matrices exit 1. The
volumetric correction raises both half-step angles by
about 12° from baseline. The trace-preserving correction
raises them by less than 1°. Neither qualifies Q-SAND.
CUDA and dense sampling were untested. The source edit was restored.

#### Apply volumetric floor compression only after positive-time impacts

On a8476fd, add one third of min(0, incoming yy) to each
diagonal entry of the projected gradient only for ordinary
granular floor hits with positive incoming duration.
This complements the preceding volumetric control at
zero-fraction hits. Leave the incoming gradient, sweep,
stress scatter, and acceptance criteria unchanged.

Build the native dry-scene target and run
`--q-sand-column-long-settle 6`. In coarse, fine, coarse
half-step and fine half-step order the angles are
33.846206°, 23.425065°, 17.107071°, and 13.060889°.
The largest change from the ordinary baseline is 0.003397°.
The matrix exits 1. The large response to volumetric
compression at zero-fraction hits is absent in the
positive-time impact control. CUDA and dense sampling
were untested. The temporary source edit was restored.

#### Reconstruct resident compression from incoming normal travel

On 1d44fad, at ordinary granular floor contacts with zero
incoming duration, set the projected yy gradient to
min(0, incoming PIC normal speed / cell width). This replaces
the gathered normal derivative in the resident compression
control with a one-cell difference from the stationary wall.
Keep the sweep, remaining gradient entries, and stress unchanged.

Build the native dry-scene target and run it without arguments
and with `--q-sand-column-long-settle 6`. The default angle is
43.751317°. The six-second angles are 18.779850°, 17.825465°,
25.304388°, and 26.519294° in coarse, fine, coarse half-step
and fine half-step order. Both commands exit 1. All matrix
cases also fail radius and persistence. The one-cell normal
travel estimate increases the half-step angles relative to
baseline, while increasing spread. CUDA and dense sampling
were untested. The source edit was restored.

#### Reconstruct resident normal strain with a reflected wall stencil

On 73d7542, for ordinary granular zero-fraction floor hits,
use the vertical stencil to recompute the normal strain rate.
Set wall velocity to zero and reflect each below-wall normal
face from the opposite above-wall face at the same horizontal
coordinates, with opposite sign. Use zero if no donor is
present. Retain only the compressive part of the resulting
yy rate. The other gradient entries, sweep and scatter use
the original path. This tests aligned grids.

Build the native dry-scene target and run it without arguments
and with `--q-sand-column-long-settle 6`. The default angle
is 43.042325°. Six-second angles are 34.298857°, 29.164190°,
35.521469°, and 31.811544° in coarse, fine, coarse half-step
and fine half-step order. Both commands exit 1. All four
matrix radii fail. CUDA and dense sampling were untested.
The temporary source edit was restored.

#### Isolate mean and deviatoric stress scattered from floor particles

On 8b141bc, for ordinary granular particles starting at
y <= 1e-8 m, scatter only the isotropic or deviatoric part
of the accepted Kirchhoff stress in separate builds.
Elevated particles scatter the full tensor. Keep the
accepted state, constitutive return, and contact unchanged.
Run the native target with `--q-sand-column-long-settle 6`.

| Case | Full stress angle | Isotropic only | Deviatoric only |
|---|---:|---:|---:|
| Coarse | 33.842809° | 7.026112° | 0.000000° |
| Fine | 23.424520° | 6.246532° | 0.000000° |
| Coarse half-step | 17.104058° | 12.260547° | 10.890136° |
| Fine half-step | 13.060319° | 12.221954° | 8.368249° |

Both matrices exit 1. Every radius remains within bounds.
Removing either component reduces the final pile height
in all four cases. The altered stresses feed back on the
trajectory, so these results do not partition force at a
fixed state. CUDA and dense sampling were untested.
The temporary solver edit was restored.

#### Separate pre-force and force-induced resident normal compression

On fc4b8d9, at zero-fraction ordinary granular floor hits,
differentiate the vertical face field before the force kick
and differentiate the post-minus-pre-force field separately.
Use zero for empty faces. In each build replace the projected
yy entry with the compressive part of one of those derivatives.
Use the original sweep, stress and other gradient entries.
Run the native target without arguments and with
`--q-sand-column-long-settle 6`.

| Case | Pre-force angle | Force increment angle |
|---|---:|---:|
| Coarse | 35.988065° | 33.842809° |
| Fine | 31.925407° | 23.424520° |
| Coarse half-step | 38.096627° | 17.104058° |
| Fine half-step | 31.747293° | 13.078997° |

Both matrices exit 1. The force-increment control changes
the baseline by at most 0.018678° and passes the default test.
The pre-force control puts both fine cases in band, but
overshoots both coarse angles and fails the default at
38.990592°. The principal response comes from the pre-force
field in these controls. CUDA and dense sampling were untested.
The temporary solver edit was restored.

#### Separate translation and affine contributions to resident compression

On cc205aa, make a second ordinary granular deposit with zero
affine slopes. At zero-fraction floor hits, reconstruct the
pre-force normal gradient using either this translation
momentum or the difference from the full deposit, divided by
the original face mass. Empty faces contribute zero. Use
only the compressive part as the projected yy entry. Keep
the actual grid velocity, stress, and sweep unchanged.

Build the native dry-scene target for each control and run
without arguments and with `--q-sand-column-long-settle 6`.

| Case | Translation only | Affine only |
|---|---:|---:|
| Coarse | 34.440162° | 35.390673° |
| Fine | 32.163320° | 23.874034° |
| Coarse half-step | 37.241163° | 19.435941° |
| Fine half-step | 31.584749° | 13.421292° |

Both matrices exit 1. Translation alone preserves the fine
cases' angle passes from the full pre-force control. It fails
the coarse nominal radius and the coarse half-step angle.
The default fails at 40.239740° with translation alone and
passes at 33.133681° with affine alone. The affine-only
matrix puts every angle outside the band. CUDA and dense
sampling were untested. The temporary solver edit was restored.

#### Isolate elevated particle translation in resident compression

On 8d0ce95, repeat the translation-only resident compression
control, but zero translation from particles at y <= 1e-8 m
in the diagnostic deposit. Preserve those particles' deposited
mass and use the original full field for transport and contact.
Run the native target with `--q-sand-column-long-settle 6`.

The angles are 34.440162°, 32.163320°, 37.241163°, and
31.584749° in coarse, fine, coarse half-step and fine half-step
order. Each matches the full translation control to six decimals.
The coarse nominal radius is 0.168620 m and the coarse half-step
overshoots the angle band. The command exits 1. Elevated
translation suffices to reproduce this control at printed
precision. CUDA and dense sampling were untested. The temporary
source edit was restored.

#### Check elevated-particle translation at dense sampling

Run the native executable from the preceding elevated-translation
control with `--q-sand-dense-column-long-settle 3`. The fixture
uses 504 particles. The diagnostic deposit and acceptance
criteria are unchanged.

| Case | Angle | Radius (m) | Final ≈1 s drift |
|---|---:|---:|---:|
| Coarse | 35.602237° | 0.183283 | 1.087018° |
| Fine | 32.813826° | 0.189250 | 0.362673° |
| Coarse half-step | 34.090560° | 0.181174 | 1.404309° |
| Fine half-step | 31.556021° | 0.185804 | 1.059072° |

The matrix exits 1. Three angles are in band, and all radii
exceed 0.16 m. All four radii increase relative to the dense
baseline, by 0.14–13.0 mm. Elevated-particle compression alone
does not control dense spreading. CUDA was untested.

#### Pair elevated-translation compression with above-floor affine attenuation

On 5ec4251, combine the elevated-translation resident compression
control with the above-floor APIC attenuation. Both use their
original formulas and only affect the ordinary granular floor
path. Build the native dry-scene target and run it without
arguments, with `--q-sand-column-long-settle 6`, and with
`--q-sand-dense-column-long-settle 3`.

| Case | Sparse 6 s angle | Dense 3 s angle | Dense radius (m) |
|---|---:|---:|---:|
| Coarse | 36.598770° | 41.117354° | 0.181242 |
| Fine | 32.590496° | 37.957227° | 0.184764 |
| Coarse half-step | 47.433442° | 41.865351° | 0.147956 |
| Fine half-step | 41.681197° | 40.328496° | 0.156332 |

All three commands exit 1. The default angle is 53.087644°.
Both dense half-step radii pass, but all dense angles overshoot
and all dense final drifts exceed one degree. Neither sparse
nominal radius passes. The two effects do not combine into
a qualifying result. CUDA was untested. The solver edit was restored.

#### Blend the fitted and differentiated gradients on the ordinary granular floor

On 137c774, average the gathered quadratic velocity derivative
and the fitted APIC slope before the ordinary granular deformation
update. Both estimates reproduce affine velocity fields on complete
stencils. Use the blend for both parts of the sweep split, keeping
stress scatter and stored affine slope unchanged. This is a
gradient-only sensitivity control.

Build the native dry-scene target and run it without arguments
and with `--q-sand-column-long-settle 6`. The default passes at
32.902857°. The six-second angles are 35.098515°, 23.992068°,
17.865474°, and 13.465275° in coarse, fine, coarse half-step
and fine half-step order. The matrix exits 1. The half-step
deficit persists and the coarse nominal angle exceeds 35°.
CUDA and dense sampling were untested. The source edit was restored.

#### Center ordinary granular stress and deformation in time together

On 518ac2b, predict ordinary granular stress by returning a half
step of deformation driven by the stored particle APIC slope.
For the accepted update, average the velocity derivatives before
and after the grid force kick. Use that derivative for both sweep
segments and keep translation, affine storage, and the sweep on
their original paths. This is an explicit predictor using two
different gradient estimates.

Build the native dry-scene target and run it without arguments
and with `--q-sand-column-long-settle 6`. The default passes at
33.567886°. The six-second angles are 34.190576°, 24.030630°,
22.358646°, and 15.505871° in coarse, fine, coarse half-step
and fine half-step order. The matrix exits 1. Both half-step
angles improve, but three angles and the fine half-step radius
fail. CUDA and dense sampling were untested. The edit was restored.

#### Center ordinary granular sweep velocity without changing stored velocity

On 296abaf, advance the ordinary granular contact sweep using
the mean of the pre-force and post-force PIC velocities.
Keep the stored velocity, stress scatter and gradient samples
at their original time levels. The sweep still determines
the duration of each deformation segment.

Build the native dry-scene executable and run it without
arguments and with `--q-sand-column-long-settle 6`. The default
passes at 31.878876°. The six-second angles are 27.767141°,
16.222215°, 17.624930° and 13.549266° in coarse, fine, coarse
half-step and fine half-step order. All four radii pass, but
all angles and persistence checks fail. The matrix exits 1.
CUDA and dense sampling were untested. The source edit was restored.

#### Center granular stress, deformation and transport together

On 419d9b7, combine the half-step APIC stress predictor with
midpoint PIC transport and the midpoint deformation derivative
on the ordinary granular floor. Keep the stored velocity and
affine fit at the end of the step. Use the original post-force
PIC velocity to evaluate the Coulomb gradient projection.
Build the native dry-scene executable and run it without
arguments and with `--q-sand-column-long-settle 6`.

The default passes at 32.644255°. The sparse matrix exits 1.
Its angles are 31.281196°, 19.653738°, 21.210125°, and
15.335547° in coarse, fine, coarse half-step and fine half-step
order. The coarse case is in band, the other three are below
30°, and the fine half-step exceeds the radius limit. CUDA and
dense sampling were untested. The temporary solver edit was restored.

#### Sample gradients at each swept granular segment midpoint

On 7be0c6d, on ordinary granular floor hits, gather the frozen
post-force grid derivative at the midpoint of the incoming
sweep segment and at the midpoint of the projected remainder.
Use the original velocity for the Coulomb derivative and keep
the original sweep, stress, and no-hit update.
Build the native dry-scene executable and run it without
arguments and with `--q-sand-column-long-settle 6`.

The default passes at 32.825894°. The six-second sparse angles
are 33.842812°, 23.424489°, 17.104056° and 13.060318° in coarse,
fine, coarse half-step and fine half-step order. The matrix
exits 1. The largest change from baseline is 0.000031°.
Spatial centering on the hit path does not materially change
the observed flattening. CUDA and dense sampling were untested.
The temporary solver edit was restored.

#### Match absolute timesteps across grid widths with resident compression

On d5584e9, repeat the zero-fraction resident normal compression
control at 6 s. Use 0.001875 s and 0.0009375 s on both grids
by replacing the two fine durations in the ordinary column matrix.
Build the native dry-scene target and run
`--q-sand-column-long-settle 6`.

| Timestep (s) | Coarse angle | Fine angle | Fine radius (m) | Fine final drift |
|---|---:|---:|---:|---:|
| 0.001875 | 36.442893° | 31.450098° | 0.145866 | 0.008558° |
| 0.0009375 | 34.609817° | 30.095979° | 0.151478 | 0.570024° |

Both fine cases and the coarse half-step pass at this endpoint.
The coarse nominal case still overshoots, and the matrix exits 1.
At equal timesteps the finer grid remains 4.51–4.99° shallower.
The matched run leaves the original qualification matrix unchanged.
CUDA and dense sampling were untested. Both temporary edits were restored.

#### Check resident compression at the intermediate grid width

On 12e33f5, apply the same zero-fraction resident normal compression
and run two additional sparse cases at h = 0.06875 m. Match the
coarse and fine controls at dt = 0.001875 s and 0.0009375 s.
Build the native dry-scene target and temporarily select those two
cases with `--q-sand-column-long-settle 6`.

| Timestep (s) | Angle | Radius (m) | Final drift |
|---|---:|---:|---:|
| 0.001875 | 34.333569° | 0.143437 | 0.013097° |
| 0.0009375 | 34.041633° | 0.146712 | 0.410078° |

Both additional cases pass at six seconds. At equal timesteps
their angles lie between the coarse and fine results. The
nominal angle changes monotonically across the three widths,
and so does the half-step angle. The result confirms a
grid-width dependence over this interval. The original coarse
nominal failure remains. CUDA and dense sampling were untested.
Both temporary edits were restored.

#### Extend intermediate-grid resident compression to twelve seconds

On ea88f37, repeat the two preceding intermediate-grid cases
with the same resident compression edit and change both
diagnostic TestScene endpoints to 12 s. Build the native dry
scene executable and use the same temporary selector
`--q-sand-column-long-settle 6`; the selected cases explicitly
override the endpoint to 12 s.

| Timestep (s) | Angle at 6 s | Angle at 12 s | Radius (m) | Final drift |
|---|---:|---:|---:|---:|
| 0.001875 | 34.333569° | 34.330527° | 0.143437 | 0.002197° |
| 0.0009375 | 34.041633° | 31.731821° | 0.146712 | 0.246440° |

The native command exits 0. Both angles remain in band and
both radii agree with their six-second values at printed
precision. The half-step angle drops 2.31° between endpoints,
so the twelve-second pass is specific to the measured window.
This is an additional sparse control at h = 0.06875 m. The
original coarse and fine qualification cases are still open.
CUDA and dense sampling were untested. Both edits were restored.

#### Repeat the twelve-second intermediate-grid cases at dense sampling

On 893c39f, repeat the preceding two intermediate-grid controls
with the same resident compression, grid width, and absolute
timesteps, but pass a sampling multiplier of two to both
TestScene calls. Select those calls with the temporary
`--q-sand-dense-column-long-settle 6` selector, with both
TestScene endpoints set to 12 s.

| Timestep (s) | Sparse angle | Dense angle | Dense radius (m) | Dense final drift |
|---|---:|---:|---:|---:|
| 0.001875 | 34.330527° | 1.678780° | 0.176568 | 5.451659° |
| 0.0009375 | 31.731821° | 28.562173° | 0.178961 | 0.229744° |

The native command exits 1. Neither sparse pass survives
particle refinement. Both dense radii exceed 0.16 m, and
the nominal case nearly flattens by 12 s. The control was
not tested on CUDA. The temporary solver and fixture edits
were restored.

#### Perturb dense column sampling symmetrically by layer

On 27dee49, perturb only the initial horizontal coordinates of
the 504-particle column. Move every x and z coordinate away
from its center plane by 0.1 particle spacing (2.5 mm) on
even-numbered layers, and toward it by the same amount on odd
layers, starting the layer count at zero. Preserve reflection
symmetry, mass, vertical sampling and the production solver.
Run the native executable with `--q-sand-dense-column-long-settle 3`.

| Case | Baseline angle | Perturbed angle | Perturbed radius (m) |
|---|---:|---:|---:|
| Coarse | 29.703637° | 29.842311° | 0.171881 |
| Fine | 26.954622° | 27.370794° | 0.183571 |
| Coarse half-step | 21.853351° | 21.994743° | 0.175245 |
| Fine half-step | 20.623778° | 21.095098° | 0.187424 |

All four cases fail angle, radius and persistence. The
maximum final angle shift is 0.472° and the radius increases
by 1.50–2.27 mm. This is a sensitivity check for one
deterministic perturbation. CUDA was untested and the
temporary fixture edit was restored.

#### Reverse particle traversal in the dense column

On f3ba955, reverse the initialized 504-particle vector just
before starting the simulation. Preserve all particle state
and the solver. Build the native dry-scene target and run
`--q-sand-dense-column-long-settle 3`.

In coarse, fine, coarse half-step and fine half-step order,
the angles are 29.703637°, 26.954622°, 21.853351° and 20.623778°.
The angles and radii match the forward-order baseline to six
printed decimals. All four cases fail angle, radius and
persistence. The command exits 1. This reversal does not
account for the dense failures. CUDA was untested and the
temporary fixture edit was restored.

#### Separate sticking and sliding resident floor compression

On 0e68e04, retain min(0, incoming yy) on zero-fraction ordinary
granular floor hits. In separate builds, apply this only when
both projected tangential advection components are zero, or
only when either component is nonzero. Keep contact and scatter
unchanged. Run the native target with `--q-sand-column-long-settle 6`.

| Case | Sticking only | Sliding only |
|---|---:|---:|
| Coarse | 36.442893° | 33.877097° |
| Fine | 30.159032° | 23.433129° |
| Coarse half-step | 34.583457° | 17.104058° |
| Fine half-step | 30.091364° | 13.060319° |

Sticking-only compression reproduces the all-resident angles
within 0.027° and passes three of four cases. Sliding-only
compression stays within 0.035° of the unmodified baseline.
Both matrices exit 1. The coarse nominal case still overshoots
with sticking compression. CUDA and dense sampling were untested.
The temporary solver edit was restored.

#### Retain the normal gradient row at compressive sticking contacts

On 16ed801, at ordinary granular zero-fraction hits with zero
tangential advection and negative incoming yy, retain the entire
incoming normal gradient row. Compare with the preceding
sticking-only control that retains only the yy component.

| Case | Normal row at 6 s | Radius (m) | Final drift |
|---|---:|---:|---:|
| Coarse | 36.269710° | 0.142682 | 0.448922° |
| Fine | 29.372680° | 0.150946 | 0.488840° |
| Coarse half-step | 33.383098° | 0.145358 | 0.871744° |
| Fine half-step | 28.450544° | 0.155952 | 0.390217° |

Build the native dry-scene target and run without arguments
and with `--q-sand-column-long-settle 6`. Both commands exit 1.
The default angle is 35.403258°. All six-second radii pass,
but both fine angles fall below the band and the coarse
nominal angle stays above it. CUDA and dense sampling were
untested. The temporary solver edit was restored.

#### Bound sticking compression by accepted elastic volume

On 8fc3502, at zero-fraction ordinary granular sticking hits,
retain the incoming compressive yy rate. When the accepted
elastic Jacobian is in (0, 1], limit the magnitude of the
added rate times the remaining duration to
max(0, log(1.05) + log(J_E)). This diagnostic bounds the
increment in the rate-times-duration approximation. It does
not impose a determinant constraint on the accepted state.

Build the native dry-scene target and run without arguments
and with `--q-sand-column-long-settle 6`. The default passes
at 32.969025°. The matrix exits 1 with angles 35.233292°,
24.654518°, 19.693978°, and 15.234264° in coarse, fine,
coarse half-step and fine half-step order. All four radii pass.
Both half-step angles remain below the band. CUDA and dense
sampling were untested. The temporary solver edit was restored.

#### Add a third particle density to the coarse nominal column

On 2c27d3d, run the ordinary native column to 3 s at
h = 0.075 m and dt = 0.001875 s with a sampling multiplier
of three. Temporarily select this case with
`--q-sand-dense-column-long-settle 3`. The 1701 particles
represent the same initial column and mass.

| Particles | Angle | Radius (m) |
|---:|---:|---:|
| 63 | 30.519543° | 0.137319 |
| 504 | 29.703637° | 0.170290 |
| 1701 | 30.988024° | 0.180906 |

At 1701 particles the final angle is in band, but the radius
exceeds 0.16 m and the final approximately one-second drift
is 2.410359°. The command exits 1. The fitted surface slopes
range from 34.338659° to 43.592505° across the four quadrants.
The radius increases again as the particle spacing decreases.
The experiment covers one width and timestep on native. The
temporary fixture edit was restored.

#### Compare radial distributions across three particle densities

On 76bc791, run the unmodified coarse nominal native trajectory
to 3 s at sampling multipliers 1, 2 and 3. Sort the final
particle radii and read percentiles at floor(p * (N - 1)).
Count particles at or beyond 0.16 m. All particles have equal
mass within each case, so count fractions are mass fractions.
These measurements leave the maximum-radius gate in place.

| Particles | Median (m) | 95th percentile (m) | Maximum (m) | At or beyond 0.16 m |
|---:|---:|---:|---:|---:|
| 63 | 0.090052 | 0.129222 | 0.137319 | 0 |
| 504 | 0.085809 | 0.147554 | 0.170290 | 12 |
| 1701 | 0.089580 | 0.151377 | 0.180906 | 44 |

Run the native dry-scene target with the temporary three-density
selector `--q-sand-dense-column-long-settle 3`. The command
exits 1. The median remains within 4.25 mm across the three
samplings. The upper tail broadens, with 2.38% and 2.59% of
the mass at or beyond 0.16 m in the two denser cases. The
temporary fixture edit was restored. CUDA was untested.

#### Track the first floor sample of particles beyond the radius limit

On 66f6ef0, run the unmodified 504-particle coarse nominal
native column to 3 s. For each particle store its radius at
the first accepted sample with y <= 1e-8 m. At the endpoint
report this value for each particle at or beyond 0.16 m.
Temporarily select this single case with
`--q-sand-dense-column-long-settle 3`.

Each of the 12 outer particles reaches its final radius by
the first sampled floor arrival, to nine decimal places.
Three groups of four reach 0.161270946 m, 0.165806263 m and
0.170290335 m. This includes the step on which they arrive,
so the check does not resolve movement within that step.
Subsequent floor motion does not account for the radius
excess in this case. The native command exits 1. Other
widths, timesteps and CUDA were untested. The fixture
edit was restored.

#### Attenuate horizontal affine expansion only on descending particles

On 3db0e38, halve positive horizontal APIC trace at deposit
only for ordinary granular particles with y > 0 and stored
vy < 0. Apply the correction equally to xx and zz. Keep
stress, advection, and contact unchanged.

| Case | Dense angle at 3 s | Dense radius (m) | Final drift |
|---|---:|---:|---:|
| Coarse | 32.794847° | 0.162763 | 0.668052° |
| Fine | 29.268760° | 0.177578 | 0.161254° |
| Coarse half-step | 30.288746° | 0.153540 | 6.016420° |
| Fine half-step | 26.744627° | 0.176222 | 4.671231° |

Build the native dry-scene target and run without arguments
and with `--q-sand-dense-column-long-settle 3`. Both commands
exit 1. The default angle is 47.949808°. Every dense radius
decreases from baseline, but only the coarse half-step meets
the radius limit and its drift exceeds six degrees. CUDA
was untested. The temporary solver edit was restored.

#### Extend descending-particle affine attenuation to six seconds

Use the same native executable as the preceding descending-only
control and run `--q-sand-dense-column-long-settle 6`.

| Case | Angle at 6 s | Radius (m) | Final drift |
|---|---:|---:|---:|
| Coarse | 35.030341° | 0.162763 | 0.035581° |
| Fine | 28.953276° | 0.177578 | 0.591381° |
| Coarse half-step | 26.393717° | 0.160434 | 0.167602° |
| Fine half-step | 24.637614° | 0.178922 | 0.006262° |

All final drift magnitudes fall below one degree, but both
half-step angles are below 30° and all radii exceed 0.16 m.
The nominal coarse angle exceeds 35° by 0.030341°. The native
matrix exits 1. CUDA was untested for this control.

#### Separate transfer and stress increments on an outer dense trajectory

On 7f72554, retain the old grid field for the ordinary gather
even with PIC storage. Trace dense coarse nominal particle 78
through the first step ending at the floor. For each step,
project the pre-force PIC velocity minus the stored particle
velocity, and the post-force PIC velocity minus the pre-force
PIC velocity, onto the outward horizontal diagonal
(-1, 0, -1) / sqrt(2). The second difference is the horizontal
stress kick, as gravity acts vertically. Sum each signed
difference over the 961 steps.

| Contribution | Sum (m/s) | Sum of magnitudes (m/s) |
|---|---:|---:|
| Pre-force transfer | -2.996078 | 3.296349 |
| Stress kick | 2.999957 | 2.999957 |

Stress supplies outward velocity and the pre-force transfer
largely cancels it over the trajectory. The sums measure
changes in velocity, while accumulated displacement also
depends on the velocities between steps. Run the native
dry-scene target with the dense three-second selector limited
to the coarse nominal case. It reproduces 29.703637° and
exits 1. The temporary instrumentation was restored.
Other particles, grid widths, timesteps and CUDA were untested.

#### Compare all twelve outer trajectories through first floor arrival

On d13ff71, repeat the preceding pre-force and post-force
velocity trace for the twelve outer particles in the coarse
nominal dense native case. Project velocity changes onto
each particle's outward horizontal quadrant diagonal.
For each particle sum the signed changes through its first
accepted step ending at y <= 1e-8 m. Also sum the actual
endpoint-minus-start radius change on steps starting at
least one cell width (0.075 m) above the floor.

| Arrival radius (m) | Particles | Total outward travel (m) | Travel on steps starting above one cell | Transfer sum (m/s) | Stress sum (m/s) |
|---:|---:|---:|---:|---:|---:|
| 0.161270946 | 4 | 0.072882599 | 74.108% | -2.997414 | 3.003507 |
| 0.165806263 | 4 | 0.077417915 | 78.270% | -2.968290 | 2.973103 |
| 0.170290335 | 4 | 0.081901987 | 81.700% | -2.996078 | 2.999957 |

The four members of each group agree at printed precision.
The horizontal stress kick and transfer changes nearly
cancel in each group, while most outward displacement
accumulates on steps starting above the bottom cell. The
dense three-second selector was temporarily limited to
this case. The angle remains 29.703637° and the native
command exits 1. The instrumentation was restored. Other
widths, timesteps and CUDA were untested.

#### Reduce the horizontal stress kick above one cell

On 4daaecd, at ordinary granular stress scatter, multiply the
x and z rows by 0.75 for particles at least one cell width
above the floor. Keep the vertical row, accepted material
state, transfer, and sweep unchanged. The diagnostic scatter
tensor need not be symmetric and does not replace the
constitutive model.

| Case | Dense angle at 3 s | Dense radius (m) |
|---|---:|---:|
| Coarse | 29.854496° | 0.168938 |
| Fine | 27.556813° | 0.181220 |
| Coarse half-step | 22.365907° | 0.171207 |
| Fine half-step | 21.172541° | 0.184822 |

Run the native dry-scene target without arguments and with
`--q-sand-dense-column-long-settle 3`. The default passes at
33.939206°. All dense angles and radii miss their bounds.
Radii decrease by 0.84–1.77 mm from baseline and the dense
command exits 1. CUDA was untested. The solver edit was restored.

#### Partition the outer particle stress kick by source height

On c89a558, accumulate an additional stress force array from
ordinary granular particles below one cell width. At particle
78 in the dense coarse nominal native column, gather this
force and the full force using the original stencil, actual
force mass denominator and timestep. Project both kicks onto
the outward diagonal (-1, 0, -1) / sqrt(2). The extra gather
does not update the trajectory.

| Receiving steps through first floor arrival | Full stress sum (m/s) | Sources below one cell (m/s) |
|---|---:|---:|
| All 961 steps | 2.999957 | 2.180594 |
| Steps starting at or above one cell | 1.286001 | 0.511620 |

Below-cell sources account for 72.7% of the signed stress
kick over the full trace and 39.8% while the receiver starts
at or above one cell. These are velocity increments rather
than displacement shares. Run the native dry-scene target
with the dense three-second selector restricted to the coarse
nominal case. The original 29.703637° angle is reproduced
and the command exits 1. Other receivers and CUDA were
untested. The temporary instrumentation was restored.

#### Reduce the horizontal stress kick below one cell

On 8705b1b, repeat the horizontal stress scatter control
with the affected source height below one grid cell. Multiply
the x and z rows of ordinary granular stress by 0.75 there.
Keep the vertical row, accepted state, transfer and sweep
unchanged. Run the native dry-scene target without arguments
and with `--q-sand-dense-column-long-settle 3`.

| Case | Dense angle at 3 s | Dense radius (m) |
|---|---:|---:|
| Coarse | 31.141223° | 0.169628 |
| Fine | 28.087343° | 0.181312 |
| Coarse half-step | 23.457813° | 0.170427 |
| Fine half-step | 22.095402° | 0.183481 |

All dense radii remain above 0.16 m despite reductions of
0.66–2.55 mm from baseline. The coarse nominal angle enters
the band, but its final drift is 2.221714°. The default angle
is 35.042638°. Both commands exit 1. CUDA was untested.
The temporary solver edit was restored.

#### Center only horizontal sweep velocity at dense sampling

On 9d7c281, average the pre-force and post-force PIC velocity
for x and z in the ordinary granular sweep. Keep the vertical
sweep velocity, stored velocity, stress and deformation
gradient unchanged. Run the native dry-scene target with
`--q-sand-dense-column-long-settle 3`.

| Case | Dense angle at 3 s | Dense radius (m) |
|---|---:|---:|
| Coarse | 29.751616° | 0.167446 |
| Fine | 26.991456° | 0.179908 |
| Coarse half-step | 21.925674° | 0.170528 |
| Fine half-step | 20.676814° | 0.183843 |

The native matrix exits 1. Dense radii shrink by 1.82–2.84 mm,
and all four still fail radius, angle and persistence. The
angle changes are under 0.073°. CUDA was untested. The
temporary solver edit was restored.

#### Compare horizontal PIC energy changes at two timesteps

On 9d51857, retain the pre-force grid field for the ordinary
gather even with PIC storage. Sum the changes in horizontal
particle translational kinetic energy from stored velocity to
pre-force PIC velocity and from pre-force to post-force PIC
velocity. Sum accepted steps through 0.75 s, before applying
the contact response in each energy sample. Temporarily select
the two coarse dense cases with the native three-second selector.

| Timestep (s) | Pre-force transfer change (J) | Grid stress change (J) |
|---:|---:|---:|
| 0.001875 | -0.305238 | 0.453743 |
| 0.0009375 | -0.301720 | 0.429639 |

Halving the timestep does not double the measured transfer
loss over this window. These totals include exchange with
affine motion and omit contact and stored affine energy.
The original angles reproduce at printed precision, and the
command exits 1. CUDA was untested. Both edits were restored.

#### Retain only the vertical stored velocity residual

On a8b3aa2, set the column fixture energy blend to 0.6.
Clear x and z of the granular stored residual correction
before its existing mass-weighted mean subtraction. Keep
the PIC sweep and force unchanged. Run the native target
with `--q-sand-column-long-settle 6` and
`--q-sand-dense-column-long-settle 3`.

| Case | Sparse angle at 6 s | Dense angle at 3 s | Dense radius (m) |
|---|---:|---:|---:|
| Coarse | 34.584213° | 39.329010° | 0.169545 |
| Fine | 32.507461° | 34.342390° | 0.176543 |
| Coarse half-step | 39.684134° | 34.519722° | 0.167899 |
| Fine half-step | 32.309501° | 31.442275° | 0.184124 |

Both matrices exit 1. Three sparse cases pass and the
coarse half-step overshoots. Three dense angles enter the
band, but all dense radii exceed 0.16 m. CUDA was untested.
The temporary solver and fixture edits were restored.

#### Retain only the horizontal stored velocity residual

On e025972, repeat the directional energy blend control with
energy blend 0.6 and clear y instead of x and z before the
mass-weighted mean subtraction. Keep PIC sweep and force
unchanged. Run the native target with
`--q-sand-column-long-settle 6` and
`--q-sand-dense-column-long-settle 3`.

| Case | Sparse angle at 6 s | Dense angle at 3 s | Dense radius (m) |
|---|---:|---:|---:|
| Coarse | 20.945768° | 22.976445° | 0.191765 |
| Fine | 11.449431° | 19.665596° | 0.210523 |
| Coarse half-step | 12.279811° | 17.303027° | 0.191819 |
| Fine half-step | 8.932510° | 15.654676° | 0.209234 |

Both native matrices exit 1 and all angles fall below 30°.
Every dense radius increases relative to baseline by
18.84–28.45 mm. The horizontal-only control therefore
does not reproduce the height improvement of vertical
retention and worsens dense spread. CUDA was untested.
Both temporary edits were restored.

#### Scale vertical-only retention by elapsed time

On 26ba380, repeat the vertical-only residual control using the
fixture's exponential retention with decay rate
-log(0.594) / 0.001875 per second. The nominal coarse step
matches the fixed-blend retention of 0.594. The fixture still
caps retention at 0.99. Build the native dry-scene target and
run `--q-sand-column-long-settle 6`.

| Case | Fixed blend angle | Time-scaled angle | Time-scaled radius (m) |
|---|---:|---:|---:|
| Coarse | 34.584213° | 34.583951° | 0.159528 |
| Fine | 32.507461° | 32.942141° | 0.142815 |
| Coarse half-step | 39.684134° | 38.314077° | 0.147770 |
| Fine half-step | 32.309501° | 32.458966° | 0.149234 |

Three sparse cases pass and the coarse half-step still exceeds
35°. Its final drift is 1.460427°. The native command exits 1.
CUDA and dense sampling were untested. Both temporary edits
were restored.

#### Check time-scaled vertical retention at dense sampling

Use the same native executable as the preceding time-scaled
vertical residual control. Run
`--q-sand-dense-column-long-settle 3`.

| Case | Angle at 3 s | Radius (m) | Final drift |
|---|---:|---:|---:|
| Coarse | 39.348933° | 0.169524 | 1.210048° |
| Fine | 32.560659° | 0.189992 | 2.852193° |
| Coarse half-step | 42.952930° | 0.170530 | 3.066236° |
| Fine half-step | 33.732831° | 0.184375 | 4.339711° |

Both fine angles enter the band. Every dense radius exceeds
0.16 m and every final drift exceeds one degree. The native
command exits 1. CUDA was untested for this diagnostic.

#### Measure the dense baseline throughout the final second

On 2f68b4a, build the native dry-scene target and run
`--q-sand-dense-column-long-settle 3`. The production
trajectory and Q-SAND acceptance rules are unchanged.

| Case | Minimum angle | Maximum angle | Final radius (m) |
|---|---:|---:|---:|
| Coarse | 27.454340° | 29.703637° | 0.170290 |
| Fine | 25.648390° | 26.954622° | 0.182075 |
| Coarse half-step | 21.853351° | 23.823039° | 0.172975 |
| Fine half-step | 20.623778° | 21.975051° | 0.185666 |

No accepted sample in the final one-second window enters
the target band. Every radius exceeds 0.16 m. The command
exits 1. This run measured the native path.

#### Verify uncut stress force and torque balance

On 3cdf186, extend the affine work fixture to sum grid force
and torque about the particle center for a symmetric Kirchhoff
stress on an uncut stencil. Compare each norm with 1e-12 in
its respective SI units. The native and SM120-enabled default
dry-scene executables both pass with force norm 7.38e-16 N,
torque norm 7.63e-16 N m, and scaled work residual 3.25e-16.
Both executables exercise the same host stress scatter. Their
work checks use the respective native and CUDA gather paths.
The tested scatter uses reference volume, as required for
Kirchhoff stress.

```powershell
cmake --build out/requalified --target fluidity_m5_dry_scene_tests fluidity_cuda_m5_dry_scene_tests
& ./out/requalified/fluidity_m5_dry_scene_tests.exe
& ./out/requalified/fluidity_cuda_m5_dry_scene_tests.exe
```

#### Recheck the six-second sparse matrix after scatter validation

On be0d159, build the native and SM120 dry-scene targets and
run both with `--q-sand-column-long-settle 6`. Both commands
exit 1. All 47 native output lines match the baseline from
the full-window diagnostic before the scatter change.

| Case | Native angle | SM120 angle | Native final window | SM120 final window |
|---|---:|---:|---:|---:|
| Coarse | 33.842809° | 33.842809° | 33.033516–33.842809° | 33.033516–33.842809° |
| Fine | 23.424520° | 23.424520° | 23.089546–23.424520° | 23.089546–23.424520° |
| Coarse half-step | 17.104058° | 17.104058° | 17.104058–18.851623° | 17.104058–18.851623° |
| Fine half-step | 13.060319° | 13.060318° | 13.060319–15.249811° | 13.060318–15.249811° |

Both backends agree on all gate outcomes. The coarse nominal
case passes, and the other three remain below 30° throughout
the final window. This comparison covers sparse sampling.

#### Recheck valid sand trajectories after stress overflow rejection

On 50beafa, run the native and SM120 dry-scene executables with
`--q-sand-column-long-settle 6`. Both commands exit 1. All 47
output lines for each backend match its preceding scatter
validation baseline exactly. The coarse nominal case passes,
and the other three remain below 30° throughout the final
second. This comparison covers the sparse matrix.

#### Increase stationary-floor friction in the dense ordinary column

On 069c8bd, set the floor friction coefficient to 1.4 instead of
0.7 for the dense native column. Keep the internal 30° friction
angle, release, solver and acceptance thresholds. Build the native
dry-scene target and run `--q-sand-dense-column-long-settle 3`.

| Case | Angle at 3 s | Radius (m) |
|---|---:|---:|
| Coarse | 30.234763° | 0.172830 |
| Fine | 27.427657° | 0.183087 |
| Coarse half-step | 22.114128° | 0.173651 |
| Fine half-step | 21.043314° | 0.185892 |

All four radii increase relative to the original floor coefficient
and exceed 0.16 m. The coarse nominal final angle enters the band,
but its persistence check fails. The matrix exits 1. CUDA was
untested and the temporary fixture edit was restored.

#### Freeze the accepted elastic and plastic state during late settlement

On ec4ac11, snapshot the sparse native column's particle
material state at the first step starting at or after 3 s.
After each subsequent accepted step, restore only its elastic
deformation and log plastic Jacobian from the snapshot.
Positions, velocities and affine slopes continue to evolve.
Run `--q-sand-column-long-settle 6`.

| Case | Baseline angle | Frozen-state angle |
|---|---:|---:|
| Coarse | 33.842809° | 32.648906° |
| Fine | 23.424520° | 23.424520° |
| Coarse half-step | 17.104058° | 17.104058° |
| Fine half-step | 13.060319° | 13.650498° |

The matrix exits 1. Preventing further changes in the
scattered stress after 3 s leaves three cases below 30°.
This freezes all material history, including dilation, and
does not isolate one constitutive term. CUDA and dense
sampling were untested. The fixture edit was restored.

#### Freeze granular material history at the 0.75 s collapse sample

On 117deaf, repeat the frozen-state control with the snapshot
taken at the first step starting at or after 0.75 s. Restore
elastic deformation and plastic log volume after each subsequent
accepted step and let position, translation and affine slope
continue evolving. Run the native target with
`--q-sand-column-long-settle 6`.

| Case | Final angle | Final window | Radius (m) |
|---|---:|---:|---:|
| Coarse | 33.852974° | 33.852974–34.131054° | 0.139211 |
| Fine | 28.615953° | 28.387421–28.615953° | 0.146942 |
| Coarse half-step | 31.444870° | 31.342935–31.448861° | 0.156926 |
| Fine half-step | 24.823254° | 24.823254–25.087146° | 0.157221 |

The native matrix exits 1. The coarse half-step stays in
band throughout the final window, while both fine cases
stay below it. Earlier freezing preserves more height than
freezing at 3 s, but the control removes ongoing material
response and is not a solver candidate. CUDA and dense
sampling were untested. The fixture edit was restored.

#### Freeze history only for particles on the floor at the collapse sample

On 047c8aa, repeat the 0.75 s material-history snapshot control
only for particles with y <= 1e-8 m at the snapshot. Membership
is fixed from that sample. Restore their elastic deformation and
plastic log volume after each subsequent accepted step.
Run the native target with `--q-sand-column-long-settle 6`.

All four final angles, radii and final-window extrema reproduce
the ordinary baseline to printed precision. The coarse nominal
passes and three cases remain below 30°. The command exits 1.
The height preservation from freezing the entire material state
does not occur when only the initial floor cohort is frozen.
CUDA and dense sampling were untested. The fixture edit was restored.

#### Freeze history only for particles elevated at the collapse sample

On 1b46e01, repeat the 0.75 s snapshot with its membership
reversed. Restore material history only for particles whose
snapshot y exceeds 1e-8 m, after each subsequent accepted step.
Run the native target with `--q-sand-column-long-settle 6`.

The final angles are 33.852974°, 28.615953°, 31.444870° and
24.823254° in coarse, fine, coarse half-step and fine half-step
order. All final radii, angles and window extrema match the
all-particle freeze to printed precision. The coarse cases pass,
both fine cases fail, and the command exits 1. The elevated
cohort alone reproduces the height preservation in this control.
This freeze removes ongoing constitutive response. CUDA and
dense sampling were untested. The fixture edit was restored.

#### Freeze only elevated-particle plastic volume after collapse

On e0766e3, repeat the elevated-cohort snapshot at 0.75 s.
After each accepted step restore only log J_pl from the
snapshot and leave elastic deformation on its normal path.
Run the native target with `--q-sand-column-long-settle 6`.
This diagnostic overrides the plastic part of the accepted
volume history.

All four angles, radii and final-window extrema match the
ordinary baseline at printed precision. The mean log J_pl
matches the elevated material freeze, but the shape does
not. The matrix exits 1. Retaining plastic volume alone
does not reproduce the height preservation. CUDA and dense
sampling were untested. The temporary fixture was restored.

#### Restore elevated-particle elastic volume with admissibility checking

On 108a8e5, select particles elevated at the 0.75 s sample.
After each accepted step rescale their elastic deformation
uniformly to the snapshot determinant and compensate log J_pl
to preserve total volume for that rescaling. If the resulting
state is inadmissible, restore the full snapshot elastic
deformation and the step's accepted log J_pl. That fallback
does not preserve total volume and can also change elastic shape.
Run the native target with `--q-sand-column-long-settle 6`.

| Case | Final angle | Radius (m) |
|---|---:|---:|
| Coarse | 33.753345° | 0.139858 |
| Fine | 28.944545° | 0.146986 |
| Coarse half-step | 26.198014° | 0.162808 |
| Fine half-step | 18.876430° | 0.158026 |

The matrix exits 1. Both half-step angles improve relative to
baseline but stay below 30°. The coarse half-step also exceeds
the radius limit. Fallback frequency was not measured, so the
results describe the combined diagnostic, not an isolated
elastic-volume response. CUDA and dense sampling were untested.
The temporary fixture edit was restored.

#### Keep the accepted state when elastic volume restoration fails

On 1371ed0, repeat the elevated elastic determinant control,
including plastic compensation, but skip each inadmissible
rescaling instead of restoring the snapshot elastic tensor.
Run the native target with `--q-sand-column-long-settle 6`.

| Case | Attempts | Rejected | Final angle | Radius (m) |
|---|---:|---:|---:|---:|
| Coarse | 103637 | 934 | 34.359739° | 0.139858 |
| Fine | 124320 | 65 | 28.944373° | 0.146986 |
| Coarse half-step | 252045 | 24 | 26.198017° | 0.162808 |
| Fine half-step | 309120 | 2141 | 18.871190° | 0.158087 |

Fewer than 1% of attempts are rejected in each case. The
half-step angle improvements persist relative to the ordinary
baseline, but three angles remain below 30° and the coarse
half-step fails radius. The matrix exits 1. CUDA and dense
sampling were untested. The temporary fixture was restored.

#### Restore elevated elastic shape at the accepted elastic volume

On 2582abb, for particles elevated at the 0.75 s snapshot,
rescale the snapshot elastic tensor to the determinant of the
new accepted tensor after each step. Leave plastic log volume
unchanged and retain the accepted tensor if the replacement
fails the granular state check. Build the native dry-scene
target and run `--q-sand-column-long-settle 6`.

| Case | Attempts | Rejected | Final angle |
|---|---:|---:|---:|
| Coarse | 103637 | 50386 | 33.842356° |
| Fine | 124320 | 50236 | 23.423281° |
| Coarse half-step | 252045 | 251497 | 17.104054° |
| Fine half-step | 309120 | 240727 | 12.756178° |

The matrix exits 1. Rejection ranges from 40.4% to 99.8%,
so this experiment cannot determine whether preserving shape
would retain height if all replacements were admissible.
CUDA and dense sampling were untested. The fixture was restored.

#### Center the granular volume rate on ordinary steps without a floor hit

On ebbbb7e, retain the Euler trial stretch ratios. For ordinary
granular steps without floor contact and with a positive trial
determinant, rescale the trial to the accepted determinant
times exp(dt times the midpoint trace). Compute that trace
from the mean of pre-force and post-force face velocities.
Keep the swept update and constitutive return unchanged.
Run the native target with `--q-sand-column-long-settle 6`.

The final angles are 34.105289°, 23.928965°, 21.827275° and
14.887363° in coarse, fine, coarse half-step and fine half-step
order. The coarse nominal passes, while the other three stay
below 30°. The fine half-step also exceeds the radius limit.
The command exits 1. CUDA and dense sampling were untested.
The temporary source edit was restored.

#### Recheck all registered M5 tests on upstream main

On a2ed1a7, build the eleven executable targets used by the
thirteen registered M5 and QSand entries in the SM120 build,
then run:

```powershell
ctest --test-dir out/requalified -R '^(M5|QSand)' --no-tests=error --output-on-failure
```

| Registered suite | Native | SM120 |
|---|---|---|
| Dry scenes | Pass | Pass |
| Sand qualification | Fail | Fail |
| Planar coupling | Pass | Pass |
| Plastic checkpoint | Pass | Pass |
| Snow compaction | Pass | Pass |
| Contact scatter | Pass | Covered by native host fixture |
| Constitutive invariants | Pass | Pass |

CTest exits 8 with eleven passes. Both registered QSand commands
use the sparse column selector's default 1.5 s endpoint. Their
four angles agree at printed precision and are 29.113507°,
26.761332°, 29.006384°, and 27.033753° in coarse, fine, coarse
half-step and fine half-step order. All four pass radius, short
drift, and kinetic energy. The dense and longer runs are separate
commands. This CTest run does not repeat the authored Blender bake.

#### Limit sticking normal compression to the incoming volume rate

On f577352, at ordinary granular zero-fraction sticking
contacts set projected yy to min(0, max(incoming yy, tr L)).
This retains compression only if both the incoming normal
rate and full trace are negative, and caps its magnitude at
the smaller of the two. The sweep and stress are unchanged.
Run the native dry-scene executable without arguments and
with `--q-sand-column-long-settle 6`.

The default passes at 33.879300°. In coarse, fine, coarse
half-step and fine half-step order the six-second angles are
36.257700°, 28.631408°, 27.037958° and 23.794735°. All radii
pass, but all angles miss the band and the matrix exits 1.
CUDA and dense sampling were untested. The source edit was restored.


#### Retain half the incoming normal compression at sticking contacts

On 9231ce0, add 0.5 times min(0, incoming yy) to projected
yy at ordinary granular zero-fraction sticking contacts.
Run the native dry-scene executable without arguments, with
`--q-sand-column-long-settle`, and with
`--q-sand-column-long-settle 6`.

The default passes at 34.320743°. In coarse, fine, coarse
half-step and fine half-step order, the registered 1.5 s
angles are 34.159356°, 31.511967°, 30.521271°, and 29.676413°.
The fine half-step misses the angle band and the command exits 1.
At 6 s, the angles are 36.081744°, 28.937929°, 28.293980°,
and 24.359077°. All four fail angle and persistence. Radius,
short drift, and kinetic energy pass in both matrices.
CUDA and dense sampling were untested. The source edit was restored.


#### Freeze the Coulomb attenuation factor in the sliding gradient

On ff407bc, retain the sliding tangential velocity fraction
in the two tangential gradient rows and omit its spatial
derivative. Normal and sticking rows retain their previous
behavior. Run the native dry-scene executable without arguments,
with `--q-sand-column-long-settle`, and with
`--q-sand-column-long-settle 6`.

The default passes at 32.785069°. In coarse, fine, coarse
half-step and fine half-step order, the registered 1.5 s
angles are 29.158576°, 26.647253°, 29.006384°, and 27.033753°.
At 6 s, the angles are 34.034745°, 23.477401°, 17.104058°,
and 13.060319°. Only the coarse nominal case meets the angle
and persistence criteria at 6 s. Both matrices exit 1.
CUDA and dense sampling were untested. The source edit was restored.


#### Shift the ordinary floor grid in both horizontal axes

On 7ca2e6a, pass grid origin (h/2, 0, h/2) to all four
ordinary sparse column cases while retaining the release center,
particle positions and acceptance settings. Run
`fluidity_m5_dry_scene_tests.exe --q-sand-column-long-settle 1.5`
and repeat with endpoint 6.

| Case | Angle at 1.5 s | Angle at 6 s | Final second angle range |
|---|---:|---:|---:|
| Coarse | 27.028300° | 31.181198° | 30.555530–31.181198° |
| Fine | 24.650853° | 25.007161° | 24.669229–25.007161° |
| Coarse half-step | 28.363688° | 16.186329° | 16.186329–17.743491° |
| Fine half-step | 25.059682° | 15.000830° | 15.000830–16.462000° |

Both matrices exit 1. At 1.5 s all four angles are 0.64–2.11°
below the aligned results in the registered recheck. The smaller
timesteps still lose angle during the extended run. Radius,
kinetic energy and short drift pass throughout. CUDA and dense
sampling were untested. The fixture was restored.


#### Smooth the force mass floor without changing its scale

On 290721a, replace max(face mass, minimum force mass)
with hypot(face mass, minimum force mass) in the force
acceleration denominator. Keep the minimum force mass at
0.0001 times the minimum particle mass. Leave the transfer
normalization and wall updates unchanged. Run the native
`--q-sand-column-long-settle` selector at endpoints 1.5 and 6.

In coarse, fine, coarse half-step and fine half-step order,
the angles are 29.113580°, 26.762277°, 29.007095° and
27.034087° at 1.5 s, and 33.842902°, 23.424453°, 17.105240°
and 13.061550° at 6 s. The 1.5 s results move by less than
0.001° from the aligned registered recheck. Only the coarse
nominal case meets the 6 s angle and persistence criteria.
Both commands exit 1. CUDA and dense sampling were untested.
The temporary source edit was restored.


#### Combine partial resident compression and volumetric APIC attenuation

On b5d7c4d, add half of min(0, incoming yy) to the projected
normal gradient on ordinary granular floor hits at zero
fraction. At deposit, subtract one quarter of max(0,
min(horizontal trace, total trace)) from each horizontal
diagonal of the ordinary granular affine slope. Build the
native dry-scene target. Run without arguments, with
`--q-sand-column-long-settle`, with that selector at 6 s,
and with `--q-sand-dense-column-long-settle 3`.

| Case | Sparse angle at 1.5 s | Sparse angle at 6 s | Dense angle at 3 s |
|---|---:|---:|---:|
| Coarse | 37.231063° | 37.798608° | 35.106480° |
| Fine | 35.252256° | 31.389268° | 32.202886° |
| Coarse half-step | 35.049612° | 29.720430° | 26.573009° |
| Fine half-step | 32.518617° | 28.095787° | 25.341230° |

All four commands exit 1. The default angle is 40.416202°.
The fine nominal sparse run alone passes the 6 s angle
and persistence criteria. All dense radii fail. CUDA was
untested. The temporary solver edit was restored.


#### Check physical time similarity of the ordinary sparse sand cases

On 0d20fde, halve the physical step duration while keeping the
nominal step sequence and observation indices. Quadruple both
Lamé moduli and gravity, double the momentum quantum, and keep
mass, geometry and friction fixed. Compare kinetic energy in
the original time units and integrate the gravity impulse over
the rescaled physical duration.

Run the native `--q-sand-column-long-settle` selector at 1.5
and 6 nominal seconds. Both commands exit 1 at the angle gate.
All eight endpoint angles and all four six-second persistence
ranges match the aligned native baseline to the printed six
decimal places. This confirms the tested creep is invariant to
the absolute time and velocity scales under this simultaneous
rescaling. It persists when the ratios of the step duration to
the elastic and gravitational time scales stay fixed. CUDA and
dense sampling were untested. The fixture was restored.


#### Compare from-rest force balance on the ordinary floor

On ab25b5a, take copies of each native sparse column at the
1.5 and 6 s endpoints. Clear translation and affine velocity
in each copy. Advance once with gravity and the ordinary floor
at the trajectory timestep and again from a fresh copy at
half that timestep. Divide stored velocity by probe duration.
Classify particles by their height before the probe. Run
`fluidity_m5_dry_scene_tests.exe --q-sand-column-long-settle`
at both endpoints.

| Case | Elevated count at 1.5 s | Elevated mean vertical acceleration (m/s²) | Elevated count at 6 s | Elevated mean vertical acceleration (m/s²) |
|---|---:|---:|---:|---:|
| Coarse | 24 | −7.642477 | 2 | +1.663006 |
| Fine | 17 | −7.267797 | 3 | −3.882381 |
| Coarse half-step | 33 | −8.191271 | 2 | −6.076605 |
| Fine half-step | 33 | −8.495345 | 3 | −8.247484 |

Each pair of probes agrees to six decimal places in mean
vertical acceleration, RMS elevated horizontal acceleration,
and wall impulse divided by duration. The floor group has zero
mean vertical acceleration at printed precision. Elevated
horizontal RMS falls from 1.306549–1.507304 m/s² at 1.5 s to
zero at printed precision at 6 s. The largest late differences
are in vertical support of the remaining elevated particles.
The copied-state probes preserve all baseline trajectory angles.
Both matrices exit 1. CUDA and dense sampling were untested.
The temporary fixture was restored.


#### Retain both signs of normal strain at resident sticking contacts

On 237f160, use the incoming yy velocity gradient as the
projected yy gradient at zero-fraction ordinary granular
floor hits with zero tangential advection. This retains
normal unloading as well as loading. Leave the contact sweep
and stress scatter unchanged. Run the native dry-scene target
without arguments and with `--q-sand-column-long-settle` at
endpoints 1.5 and 6.

The default passes at 32.201301°. In coarse, fine, coarse
half-step and fine half-step order the 1.5 s angles are
24.726926°, 23.894080°, 30.470306°, and 29.616725°.
The 6 s angles are 0°, 0°, 12.662061°, and 12.488222°.
Both nominal piles lose all height, and both half-step
piles remain below 30° throughout the final second. All
radii pass. Both matrices exit 1. CUDA and dense sampling
were untested. The temporary solver edit was restored.


#### Refine the ordinary-floor grid at a matched timestep

On 51ddfb1, use the ordinary sparse column selector with
grid widths 0.075, 0.0625, 0.050, and 0.0375 m. Set all
four timesteps to 0.00078125 s. Retain the same 63 particles,
0.050 m initial particle spacing, geometry, and acceptance
criteria. Run the native `--q-sand-column-long-settle`
selector at endpoints 1.5 and 6.

| Grid width (m) | Angle at 1.5 s | Angle at 6 s | Radius at 6 s (m) |
|---|---:|---:|---:|
| 0.075 | 30.385098° | 18.186985° | 0.144753 |
| 0.0625 | 27.033753° | 13.060319° | 0.155423 |
| 0.050 | 20.562883° | 2.821687° | 0.159485 |
| 0.0375 | 13.868597° | 0.000000° | 0.152640 |

All four radii pass and both matrices exit 1. At fixed
timestep and sparse quadrature, reducing grid width lowers
the pile angle at both endpoints. The two finest grids
have unchanged radii to printed precision between 1.5 and 6 s,
while their heights continue to fall. This comparison does
not measure joint grid and particle refinement. CUDA and
dense sampling were untested. The temporary fixture was restored.


#### Repeat matched-timestep grid refinement with dense sampling

On 4b8caa9, repeat the four matched-timestep widths above
with 504 particles at 0.025 m spacing. Keep the original
physical column, timestep 0.00078125 s and gate limits.
Run the native `--q-sand-dense-column-long-settle` selector
at endpoints 1.5 and 6.

| Grid width (m) | Angle at 1.5 s | Angle at 6 s | Height at 6 s (m) | Radius at 6 s (m) |
|---|---:|---:|---:|---:|
| 0.075 | 27.342640° | 20.575754° | 0.065276 | 0.173887 |
| 0.0625 | 23.529643° | 20.800137° | 0.070528 | 0.185666 |
| 0.050 | 21.152362° | 19.913111° | 0.070854 | 0.195591 |
| 0.0375 | 19.162558° | 18.692792° | 0.068461 | 0.202343 |

All four cases fail angle and radius at both endpoints
and persistence at 6 s. At 0.0375 m the dense pile retains
0.068461 m of height at 6 s, compared with zero for the sparse
pile. The dense six-second heights span 5.6 mm, while radii
increase by 28.5 mm from the coarsest to finest grid. Increasing
quadrature density changes the primary failure from loss of
height to an overly wide pile in this refinement comparison.
CUDA was untested. The temporary fixture was restored.


#### Limit higher floor friction to the initial collapse

On 65342c1, use floor friction 1.4 before the 0.75 s
sampling boundary and 0.7 after it. Keep the 30° material
friction angle and the original column and gate criteria.
Run the native dry-scene target with
`--q-sand-column-long-settle 3` and
`--q-sand-dense-column-long-settle 3`.

| Case | Sparse angle | Sparse radius (m) | Dense angle | Dense radius (m) |
|---|---:|---:|---:|---:|
| Coarse | 30.712768° | 0.138248 | 30.234763° | 0.172830 |
| Fine | 23.918569° | 0.147504 | 27.427657° | 0.183087 |
| Coarse half-step | 22.856989° | 0.143314 | 22.114128° | 0.173651 |
| Fine half-step | 20.822540° | 0.155405 | 21.043314° | 0.185892 |

All eight persistence checks fail and both commands exit 1.
The dense angles and radii reproduce the constant-1.4 control
to the displayed precision. Returning to the original floor
friction after collapse does not reduce the measured spread.
CUDA was untested. The temporary fixture was restored.


#### Use fitted affine slopes only when a granular particle misses the floor

On e837d6e, use the gathered affine fit as the deformation
gradient only on ordinary granular steps without a floor hit
or platen. Preserve the differentiated velocity gradient for
the impact split and preserve scatter, transport, and stored
affine slopes. Run the native dry-scene executable without
arguments and with `--q-sand-column-long-settle` at 1.5 and 6 s.

The default passes at 32.977920°. In coarse, fine, coarse
half-step and fine half-step order, the 1.5 s angles are
30.359980°, 27.626512°, 29.032880°, and 27.267543°.
The 6 s angles are 35.712193°, 24.470802°, 18.687381°,
and 13.877397°. Both matrices exit 1. The coarse nominal
case overshoots at 6 s and the other three stay below 30°.
All radii pass. Dense sampling and CUDA were untested.
The temporary solver edit was restored.


#### Recheck registered M5 and the production mixed-scene gate

On 6dcd970, build the registered M5 native and SM120 targets
and rerun the CTest selection above. Eleven of thirteen pass.
Only QSand.native and QSand.cuda fail. Both reproduce all four
1.5 s angles from the a2ed1a7 recheck to the printed precision.

Build the CLI and its fixture, then run:

```powershell
cmake --build out/requalified --target fluidity_simulate fluidity_simulate_fixture
ctest --test-dir out/requalified -R '^SimulateCli.execution_validation$' --no-tests=error --output-on-failure
```

The validation test passes, including the mixed input preflight.
M6 wide-coupled qualification at 996a34e covers floating-box
pressure. The rotating-box tip 8dda003 changes fixed-orientation
tangential contact. Neither supplies production snow stress
evolution or a reconstructed moving snow–water interface.
The bounded planar and checkpoint tests pass, and the authored
snow–water execution still requires the production integration
described above.


#### Cap resident sticking compression by estimated wall traction

On bd97f98, at zero-fraction ordinary granular sticking hits,
estimate the wall reaction as particle mass times incoming
downward PIC speed divided by step duration and reference
area (reference volume to the two-thirds power). Retain
only compressive yy, limiting the rate so the linearized
normal stress increase using the longitudinal Lamé modulus
fits within the remaining margin to that reaction estimate.
Leave the contact sweep and stress scatter unchanged.

Run the native dry-scene executable without arguments and with
`--q-sand-column-long-settle` at 1.5 and 6 s. The default
fails at 35.674372°. In coarse, fine, coarse half-step and
fine half-step order, the 1.5 s angles are 37.360886°,
33.933252°, 32.198574°, and 32.666782°. The 6 s angles are
36.442893°, 30.159032°, 34.583457°, and 30.091364°. The six-second
angles reproduce the uncapped sticking-only control to printed
precision. The coarse nominal case still overshoots, while
the other three pass. All three commands exit 1. CUDA and dense
sampling were untested. The temporary solver edit was restored.


#### Project ordinary-floor normal face velocities before gather

On b77028e, zero negative normal velocities on occupied y faces
at or below the wall before gathering on the ordinary granular
path. In a second build remove half the inward velocity. Account
for the corresponding grid wall impulse and retain the particle
sweep. Run the native `--q-sand-column-long-settle` selector at
1.5 and 6 s.

Full projection rejects all four trajectories with
FAILED_PRECONDITION before 0.75 s. Half projection rejects
both half-step trajectories before 0.75 s. The nominal coarse
and fine trajectories complete but reach 7.837415° and
14.364537° at 1.5 s, then 2.625895° and 15.140421° at 6 s.
All four commands exit 1. The direct face clamp changes the
incoming normal speed available to the swept Coulomb response
and does not provide qualifying support. CUDA and dense
sampling were untested. The temporary solver edit was restored.


#### Constrain floor donors for the deformation gradient only

On 304b522, gather a second gradient on the ordinary granular
path using a copy of the post-force face velocities. Set inward
normal velocities to zero at and below the floor. In a second
build, reverse inward velocities strictly below the floor and
keep the zero at the floor. Use the original gather for particle
transport, swept contact and stored affine slopes. Run the native
`--q-sand-column-long-settle` selector at 1.5 and 6 s.

| Case | Zero at 1.5 s | Zero at 6 s | Reverse at 1.5 s | Reverse at 6 s |
|---|---:|---:|---:|---:|
| Coarse | 36.328689° | 31.977575° | 34.341872° | 28.822044° |
| Fine | 29.399970° | 26.863965° | 27.811605° | 19.365204° |
| Coarse half-step | 35.474188° | 31.864843° | 38.380056° | 29.251989° |
| Fine half-step | 32.609033° | 27.288894° | 31.852600° | 26.796721° |

All trajectories complete and all four commands exit 1.
With zero continuation both coarse cases stay in the angle band
through the final second, and both fine cases remain below 30°.
Reversing below-floor velocity lowers all four six-second angles.
CUDA and dense sampling were untested. The temporary source edit
was restored.


#### Combine the wall-constrained gradient with resident compression

On 33cce2e, zero inward normal face velocities at and below
the floor for the deformation gather only. At ordinary granular
zero-fraction sticking contacts, retain the negative yy entry
from that constrained gradient after the Coulomb projection.
The original gather still drives swept contact and stored
affine slopes. Run the native dry-scene executable without
arguments and with `--q-sand-column-long-settle` at 1.5 and 6 s.

| Case | Angle at 1.5 s | Angle at 6 s | Radius at 6 s (m) |
|---|---:|---:|---:|
| Coarse | 37.245644° | 31.464912° | 0.201599 |
| Fine | 29.304562° | 28.771109° | 0.185939 |
| Coarse half-step | 39.588874° | 33.191044° | 0.181759 |
| Fine half-step | 32.209802° | 30.644791° | 0.180533 |

Both coarse cases meet the six-second angle and persistence
criteria. The fine half-step enters the band at the endpoint,
but its window minimum is 29.721677°. All radii exceed the
limit at both endpoints, and all three commands exit 1.
The default exits before printing the pile summary. CUDA and
dense sampling were untested. The temporary solver edit was restored.


#### Retain only compression common to the original and wall gradients

On 4c16ce4, compute the zero-continuation wall gradient in a
second gather. Leave the original gather in use everywhere
except the projected yy entry for zero-fraction ordinary
granular sticking contacts. At those contacts set that entry
to min(0, max(original yy, constrained yy)). Rigid translation
with zero original gradient therefore contributes no compression.
Run the native dry-scene executable without arguments and with
`--q-sand-column-long-settle` at 1.5 and 6 s.

The ordinary wall impulse budget check passes. The default
pile still fails at 35.674372°. The four 1.5 s angles and
four 6 s angles match the traction-capped sticking control to
six decimal places. All radii pass, and the coarse nominal
angle remains above the band at both endpoints. All three
commands exit 1. CUDA and dense sampling were untested.
The temporary solver edit was restored.


#### Bound resident compression by the gravitational cell timescale

On cf5c427, at ordinary granular zero-fraction sticking contacts
retain a compressive rate of q / (1 + q / r), where q is the
positive magnitude of incoming normal compression and
r = sqrt(abs(gravity) / cell width). With zero gravity retain
the original projection. Leave transport and scatter unchanged.
Run the native dry-scene target without arguments and with
`--q-sand-column-long-settle` at 1.5 and 6 s.

The default angle is 35.042551°. The 1.5 s angles are
35.883539°, 32.901328°, 31.448252°, and 31.558075° in coarse,
fine, coarse half-step and fine half-step order. The 6 s angles
are 36.789443°, 29.852859°, 32.092946°, and 28.843850°.
All radii pass. Only the coarse half-step meets both the 6 s
angle and persistence criteria. All three commands exit 1.
CUDA and dense sampling were untested. The edit was restored.


#### Add the uniaxial-stress lateral strain at sticking floor contacts

On 30dcb4f, retain negative incoming yy at zero-fraction
ordinary granular sticking contacts and set both horizontal
diagonal rates to minus Poisson ratio times that normal
rate. Compute Poisson ratio from the material Lamé moduli.
This gives zero incremental lateral stress in the linearized
elastic relation before the granular return. Keep the contact
sweep and stress scatter. Run the native dry-scene executable
without arguments and with `--q-sand-column-long-settle`
at 1.5 and 6 s.

The default passes at 34.654762°. In coarse, fine, coarse
half-step and fine half-step order, the 1.5 s angles are
35.002942°, 32.292187°, 30.891628°, and 30.414653°.
The 6 s angles are 36.510201°, 29.161990°, 30.008283°,
and 26.622177°. All radii pass. No case stays in the angle
band across the final second. Both extended runs exit 1.
CUDA and dense sampling were untested. The temporary solver
edit was restored.


#### Limit sticking-contact lateral strain to rates above the gravity scale

On cea5a1a, retain incoming normal compression q at ordinary
zero-fraction granular sticking contacts. Apply lateral expansion
equal to Poisson ratio times max(0, q - sqrt(abs(gravity) / cell
width)) in each horizontal direction. Leave the sweep and stress
scatter unchanged. Run the native dry-scene executable without
arguments and with `--q-sand-column-long-settle` at 1.5 and 6 s.

All nine endpoint angles reproduce the traction-capped sticking
control to six decimal places. The default reaches 35.674372°.
The six-second coarse nominal case overshoots, while the other
three stay in range for every sampled step of the final second.
All three commands exit 1. CUDA and dense sampling were untested.
The temporary solver edit was restored.


#### Scale sticking compression by the quadratic acoustic margin

On 310f861, multiply retained compressive yy by max(0, 1 - f²)
at ordinary zero-fraction granular sticking contacts, where f is
the accepted acoustic Courant number divided by the configured
limit. Keep the particle sweep and stress scatter. Run the native
dry-scene executable without arguments and with
`--q-sand-column-long-settle` at 1.5 and 6 s.

The default reaches 35.415474°. In coarse, fine, coarse
half-step and fine half-step order the 1.5 s angles are
36.792724°, 33.461272°, 32.109284°, and 32.522288°.
At 6 s they are 36.442327°, 29.994633°, 34.320297°, and
29.952200°. All radii pass. Only coarse half-step stays in
range for the full final second. Both fine endpoints lose
their passing angles relative to the uncapped sticking control,
and the coarse nominal overshoot persists. All commands exit 1.
CUDA and dense sampling were untested. The solver edit was restored.


#### Split the resident sticking deformation increment

On c12a9ad, retain negative incoming yy at ordinary granular
zero-fraction sticking contacts. Compose two half-duration
Euler deformation increments using that frozen gradient before
the existing constitutive return. Keep stress scatter and the
contact sweep. Run the native dry-scene executable without
arguments and with `--q-sand-column-long-settle` at 1.5 and 6 s.

The default reaches 35.669255°. In coarse, fine, coarse half-step
and fine half-step order, the 1.5 s angles are 37.350104°,
33.925345°, 32.195558°, and 32.662864°. The 6 s angles are
36.442749°, 30.155036°, 34.574881°, and 30.087677°.
All 6 s angles move by less than 0.009° from the one-increment
sticking control. The coarse nominal case still overshoots, and
the other three remain in range throughout the final second.
All three commands exit 1. CUDA and dense sampling were untested.
The temporary solver edit was restored.


#### Require lateral convergence for sticking floor compression

On 0758555, retain negative incoming yy at ordinary granular
zero-fraction sticking contacts only when incoming xx and zz
are both nonpositive. Run the native dry-scene executable
without arguments and with `--q-sand-column-long-settle` at
1.5 and 6 s.

The default passes at 32.892782°. In coarse, fine, coarse
half-step and fine half-step order, the 1.5 s angles are
29.423627°, 26.763018°, 29.025285°, and 27.033753°.
At 6 s they are 34.473454°, 23.442551°, 17.324353°,
and 13.060319°. Only coarse nominal stays in range across
the final second. Both matrices exit 1. Requiring lateral
convergence removes the improvement in the three shallow
cases observed with unconditional sticking compression.
CUDA and dense sampling were untested. The edit was restored.


#### Attenuate sticking compression using accepted normal stress

On a487a59, divide the retained compressive yy rate at ordinary
zero-fraction sticking contacts by 1 + r², with r the ratio of
compressive Kirchhoff normal stress to reference density times
gravity magnitude times cell width. Leave the transport and
stress scatter unchanged. Run the native dry-scene executable
without arguments and with `--q-sand-column-long-settle` at
1.5 and 6 s.

The default passes at 34.127061°. In coarse, fine, coarse
half-step and fine half-step order, the 1.5 s angles are
33.863161°, 30.952831°, 30.619671°, and 29.691947°.
The 6 s angles are 36.111911°, 28.694363°, 28.824217°,
and 24.359193°. Both matrices exit 1. All radii pass at 6 s,
and all angles miss the band throughout the final second.
The stress attenuation leaves the coarse overshoot and weakens
support in the other cases. CUDA and dense sampling were untested.
The temporary solver edit was restored.


#### Restrict resident sticking compression to the latter part of collapse

On bd71276, retain the negative incoming yy rate at ordinary
zero-fraction granular sticking contacts for steps beginning
in [0.25, 0.75) s. Before and after that interval use the
original projection. Run the native dry-scene executable
without arguments and with `--q-sand-column-long-settle` at
1.5 and 6 s.

The default passes at 33.275906°. In coarse, fine, coarse
half-step and fine half-step order, the 1.5 s angles are
30.608690°, 28.323033°, 29.873525°, and 28.173767°.
At 6 s they are 35.563823°, 25.856557°, 23.404519°, and
18.382700°. All radii pass. Both matrices exit 1, and none
stays in range throughout the final second. The later part
of collapse alone retains some height in the shallow cases
but does not reproduce the full sticking control. CUDA and
dense sampling were untested. Both temporary files were restored.


#### Restrict resident sticking compression to the first 0.25 s

On 5ee1e84, repeat the preceding window control with retained
normal compression enabled only on steps beginning before
0.25 s. Run the native dry-scene executable without arguments
and with `--q-sand-column-long-settle` at 1.5 and 6 s.

The default reaches 35.416659°. In coarse, fine, coarse
half-step and fine half-step order the 1.5 s angles are
36.880087°, 33.493648°, 31.386409°, and 31.613307°.
At 6 s they are 36.432091°, 30.036746°, 32.099854°, and
29.111103°. All radii pass. Fine nominal and coarse half-step
stay in range throughout the final second. The first window
retains more height in each shallow case than the 0.25–0.75 s
window, and produces the coarse overshoot by itself.
All commands exit 1. CUDA and dense sampling were untested.
Both temporary files were restored.


#### Center the retained sticking strain across the grid force kick

On 052849c, gather the pre-force normal gradient as well as
the original post-force gradient. Retain the compressive part
of their mean as projected yy at ordinary zero-fraction
granular sticking contacts. Keep transport, stress scatter, and
the other deformation terms unchanged. Run the native dry-scene
executable without arguments and with `--q-sand-column-long-settle`
at 1.5 and 6 s.

The default reaches 36.433648°. In coarse, fine, coarse
half-step and fine half-step order, the 1.5 s angles are
38.864658°, 35.460421°, 33.744069°, and 34.169723°.
At 6 s they are 36.673558°, 31.180677°, 37.773265°, and
30.627617°. Both fine cases stay in the band through the
final sampled second. Both coarse cases overshoot. All radii
pass and all commands exit 1. CUDA and dense sampling were
untested. The temporary solver edit was restored.


#### Measure elevated mass in the six-second sand piles

On 885e375, sum the mass strictly above half a cell and
strictly above half the measured apex height, then divide by
total mass. The diagnostic reads the final accepted state and
leaves the angle estimator and acceptance intact. Run the
native `--q-sand-column-long-settle 6` and
`--q-sand-dense-column-long-settle 6` selectors.

| Case | Sparse above half cell | Sparse above half apex | Dense above half cell | Dense above half apex |
|---|---:|---:|---:|---:|
| Coarse | 3.1746% | 3.1746% | 7.1429% | 5.5556% |
| Fine | 3.1746% | 3.1746% | 8.7302% | 5.5556% |
| Coarse half-step | 1.5873% | 3.1746% | 3.9683% | 3.9683% |
| Fine half-step | 1.5873% | 3.1746% | 7.1429% | 5.5556% |

Exactly two particles in each sparse case exceed half the
apex height. The dense cases retain 20–28 such particles,
but all four exceed the radius limit. Both matrices exit 1.
CUDA was untested for this diagnostic. The fixture was restored.


#### Retain rotation about the wall normal at sticking contacts

On 0bdb1e9, retain the incoming skew xz component alongside
compressive yy at ordinary granular zero-fraction sticking
contacts. Keep the other projected entries, sweep and stress
scatter. Run the native dry-scene executable without arguments
and with `--q-sand-column-long-settle` at 1.5 and 6 s.

All nine endpoint angles match the sticking compression
reference to six decimal places. The default reaches 35.674372°.
At six seconds coarse nominal overshoots at 36.442893°, and the
other three stay in range throughout the final second. All three
commands exit 1. The isolated spin term has no measured angle
effect at this precision. CUDA and dense sampling were untested.
The temporary solver edit was restored.


#### Shift horizontal grid alignment with resident sticking compression

On 51d433f, add half the cell width to both horizontal grid
origin coordinates while keeping the physical release fixed.
At ordinary zero-fraction granular sticking contacts retain
negative incoming yy. Run the native sparse
`--q-sand-column-long-settle` selector at 1.5 and 6 s.

In coarse, fine, coarse half-step and fine half-step order,
the 1.5 s angles are 34.388440°, 29.896100°, 29.845389°,
and 27.697394°. The 6 s angles are 35.221434°, 28.980933°,
31.634207°, and 26.862392°. All radii pass. Only coarse
half-step stays within the angle band across the final second.
Both commands exit 1. The shift lowers the coarse overshoot
but also moves both fine cases below the band. CUDA and dense
sampling were untested. Both temporary edits were restored.


#### Return to the granular cone between sticking half increments

On 5510126, repeat the two half-duration deformation increments
at ordinary zero-fraction granular sticking contacts. After the
first increment return to the cone and carry its elastic map
and plastic Jacobian into the second. Keep the sweep and scatter.
Run the native default and `--q-sand-column-long-settle` at
1.5 and 6 s.

At six seconds the four angles are 36.442749°, 30.155022°,
34.574903°, and 30.087675° in coarse, fine, coarse half-step
and fine half-step order. All nine endpoint angles change by
at most 0.000022° relative to splitting the increment without
an intermediate return. The coarse nominal overshoot persists
and all three commands exit 1. Dense and CUDA were untested.
The temporary solver edit was restored.
