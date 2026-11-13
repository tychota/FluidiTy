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
