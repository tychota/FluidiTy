# 2D coarse–fine transfer qualification

The M0 adaptive-basis spike passes the bounded CPU fixture gates for a 2:1 MAC patch with one aligned half-plane.
The candidate uses cubic splines on each component's face-normal axis and quadratic splines on its transverse axis.
Coarse and fine splines use their original Greville anchors. Fine splines enter where their closed support fits inside
the fine half-plane. Coarse splines retain only the dyadic child terms outside that eligibility rule.
See the [qualification plan](../superpowers/plans/2026-11-08-coarse-fine-transfer-basis.md) and the
[adaptive promotion gate](../superpowers/specs/2026-11-07-fluidity-design.md#73-grid-coarsening).

## Reproduce the evidence

From the repository root, use the locked reference environment with uv 0.13.7 or later and Python 3.13.
The offline form assumes the interpreter and locked packages are already in the uv cache.

```powershell
uv run --offline --project reference --locked --group native python tools/adaptivity/qualify_coarse_fine_basis.py --repeats 7 --output out/adaptivity-task-2/qualification.json
uv run --offline --project reference --locked --group native pytest reference/tests tools/tests tools/native/tests -q
```

The JSON in the ignored `out/` tree records the result of each fixture and every basis query, including
actual precision, geometry, active support counts in coarse/fine order, and defects. The command returns
exit code 1 after writing the report if any fixture gate fails. A nonfinite metric is recorded as `null` and
its path appears in `nonfinite_metrics` in the failing report.

## Transfer moments and knot nesting

Each precision uses 48 deterministic queries per evaluator. The set combines both face-normal axes, both interface
axes, coarse cell widths 0.5 m and 2 m, and six local positions from coarse interior to fine interior. It includes an
interface knot, off-knot points, translated negative world coordinates, and a 90-degree rotation with a translated
origin. The uniform cubic-normal comparator uses the same query geometry.

For particle position `x`, anchor `x_i`, coarse width `h`, weight `w_i` and physical gradient `∇w_i`, set
`r_i = (x_i - x)/h` and `g_i = h∇w_i`. All entries below are maximum absolute component defects.
The four invariants are `Σw_i = 1`, `Σw_i r_i = 0`, `Σg_i = 0` and `Σr_i g_iᵀ = I`.

| Basis | Arithmetic | Partition | First moment | Gradient sum | Affine gradient |
| --- | --- | ---: | ---: | ---: | ---: |
| Uniform cubic-normal | FP32 | 1.19209e-7 | 2.79397e-8 | 5.96046e-8 | 5.96046e-8 |
| Coarse–fine | FP32 | 5.96046e-8 | 2.79397e-8 | 1.19209e-7 | 5.96046e-8 |
| Uniform cubic-normal | FP64 | 1.11022e-16 | 5.59448e-17 | 1.11022e-16 | 2.22045e-16 |
| Coarse–fine | FP64 | 1.11022e-16 | 5.55112e-17 | 2.22045e-16 | 1.11022e-16 |
| Uniform quadratic oracle | FP64 | 0 | 0 | 0 | 0 |

The FP32 limits are 1e-6 for partition and first moment and 2e-6 for each gradient defect. The FP64 cross-check
uses 1e-12 for all four. Both precisions passed on the uniform cubic-normal and transition bases with evaluation
and moment reduction in the stated dtype. The existing quadratic oracle computes in FP64. Its uniform zeros
were measured on these dyadic queries.

With the interface and face normal both on x, the coarse interior, mixed support and fine interior have
coarse/fine counts of (9, 0), (9, 3) and (0, 9). With the interface on x and face normal on y, those counts
are (8, 0), (8, 4) and (0, 8). These examples use width 0.5 m and agree across both precisions.

The standard MAC quadratic-normal lattice fails the dyadic knot nesting requirement. The coarse normal knot
at `x/h = 0.5` lies between fine knots at `0.25` and `0.75`. The FP64 oracle probe measured coarse
one-sided curvatures of `-2/h²` and `+1/h²`, while the fine basis function centered at that coarse knot
had curvature `-8/h²` on both sides. All fine functions are polynomial across the interval between those
fine knots, so their linear combinations cannot reproduce the coarse curvature jump. Both MAC orientations
confirmed the obstruction with physical face anchors preserved.

## Sealed pressure patch and masked support

The independent pressure fixture has one coarse cell of area 1 m² and two fine cells of area 0.25 m²
each. Each interface subface has length 0.5 m and normal center separation 0.75 m. The time step and inverse
density have unit values. Two columns of outward incidence `R` assign each flux to the coarse cell with sign
+1 and to its own fine cell with sign −1. Each subface has one unknown. Predicted fluxes of 0.375 and
−0.125 m²/s produce outward cell fluxes (0.25, −0.375, 0.125) m²/s and an aggregate coarse flux of
0.25 m²/s in both orientations.

For conductances `k`, assemble `A = R diag(k) Rᵀ` and measure the constant-pressure residual
`||A p||∞ / max(||A||∞ ||p||∞, 1e-12 m²/s)`. The zero-pressure probe exercises the denominator floor.
With coarse pressure pinned to zero, the two fine equations give `p_fine = q*_subface/k`. Update each
subface flux once. The predicted and corrected divergence use the same outward flux sum divided by each
cell area. The predicted divergence is (0.25, −1.5, 0.5) s⁻¹ in both precisions and orientations.

| Arithmetic, both orientations | Absolute constant-pressure residual (m²/s) | Relative residual | Maximum corrected divergence (s⁻¹) |
| --- | ---: | ---: | ---: |
| FP32 | 1.39078e-8 | 4.01185e-9 | 0 |
| FP64 | 6.29126e-17 | 1.81479e-17 | 0 |

Both precisions passed the 1e-6 relative residual and 1e-6 s⁻¹ local divergence limits. This is a direct
algebraic solve of the sealed three-cell fixture. Production pressure qualification requires the integrated
operator and volume-scaled divergence gates on solver cases.

The masked fixture starts with nine coarse and three fine active anchors. Removing its largest weight leaves
11 anchors with affine rank two and a raw partition defect of 0.3886032104 at both precisions. Keeping
the three anchors on the interface line leaves affine rank one and raw defects of 0.4563801885 in FP32
and 0.4563802083 in FP64. Rank is computed from differences between retained anchor positions. The
reported dispositions require PIC for incomplete support and for rank-deficient support, respectively.
Both defects are measured before normalization. Applying and validating physical fallback remains
production work, since normalization alone cannot restore missing affine support.

## CPU timings and remaining gates

Measured on Windows 11 build 26200 (AMD64 Family 26 Model 68), Python 3.13.18 and NumPy 2.2.6.
Each evaluator warmed once and then ran seven batches of the same 48 queries per precision. Evaluator
order alternated between repeats. The median includes the Python/NumPy evaluator, stencil creation and
the query loop.

| Arithmetic | Uniform cubic-normal median per batch (ms) | Coarse–fine median per batch (ms) | Coarse–fine / uniform |
| --- | ---: | ---: | ---: |
| FP32 | 2.737 | 5.298 | 1.94× |
| FP64 | 2.566 | 5.088 | 1.98× |

These measurements satisfy the M0 2D adaptive-basis evidence item for this single interface. Production
promotion requires 3D interface, edge and corner evidence, multilevel support, physical masking and fallback,
conservative remap, solver integration, and measured CUDA throughput on the target workload.
See the [design milestones](../superpowers/specs/2026-11-07-fluidity-design.md#152-milestone-capabilities-not-an-implementation-plan).

## M3 3D coarse–fine qualification

The standalone fixture tests cubic face-normal and quadratic transverse splines on an aligned half-space,
quadrant and octant, with 2:1 refinement. The half-space's 3D marginal reproduces the existing 2D oracle
when summed along the extruded axis. The 2D quadratic-normal knot mismatch still rules out that normal
kernel for this dyadic construction. The same-degree uniform 3D stencil and the native uniform transfer and
pressure suites provide separate controls.

```powershell
uv run --offline --project reference --locked --group native python tools/adaptivity/qualify_coarse_fine_3d.py --output out/adaptivity-m3/qualification.json
uv run --offline --project reference --locked --group native pytest reference/tests/test_coarse_fine_basis.py tools/tests/test_qualify_coarse_fine_basis.py tools/tests/test_qualify_coarse_fine_3d.py -q
```

Use uv 0.13.7 or later with a populated cache. The JSON records maxima and their query geometry, support
counts, both pressure orientations, masked defects and limits. A nonfinite transfer metric is recorded as null and fails the gate. The process exits 1 if a gate fails.
Each topology and dtype has 120 queries per evaluator across all three face components, two cell widths
(0.5 and 2 m), signed rotation, negative coordinates, and knot and off-knot points. The interface, edge
and corner had 48, 48 and 44 mixed-support queries, respectively, in each precision. The affine control
uses a normalized scalar intercept of 0.25 and slopes (0.375, −0.25, 0.5) for each MAC component.

| Arithmetic and stencil | Partition | First moment | Gradient sum | Affine gradient moment | Affine value | Affine gradient |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 adaptive | 1.19e-7 | 5.59e-8 | 2.38e-7 | 2.38e-7 | 2.98e-8 | 1.19e-7 |
| FP32 uniform | 1.19e-7 | 4.38e-8 | 1.20e-7 | 2.38e-7 | 2.98e-8 | 8.94e-8 |
| FP64 adaptive | 2.22e-16 | 7.29e-17 | 4.44e-16 | 4.44e-16 | 5.55e-17 | 1.11e-16 |
| FP64 uniform | 2.22e-16 | 7.98e-17 | 2.23e-16 | 4.44e-16 | 8.33e-17 | 1.67e-16 |
| FP32 limit | 1e-6 | 1e-6 | 2e-6 | 2e-6 | 1e-6 | 2e-6 |
| FP64 limit | 1e-12 | 1e-12 | 1e-12 | 1e-12 | 1e-12 | 1e-12 |

Position offsets and gradients use coarse cell units. The sealed finite-volume graph has one 1 m³ coarse
cell per incident plane and eight 0.125 m³ fine cells. Four 0.25 m² subfaces per plane use 0.75 m normal
center spacing. Both orientations gave outward coarse flux +0.25 m³/s and aggregate fine flux −0.25 m³/s
per plane, with zero balance defect. Flipping one fine incidence sign exposed a 0.75 m³/s defect.
The constant-pressure relative residual uses `max(||A||∞ ||p||∞, 1e-12 m³/s)` and a 1e-6 limit.

| Worst over both orientations | Interface | Edge | Corner |
| --- | ---: | ---: | ---: |
| FP32 constant-pressure relative residual | 2.50e-8 | 2.12e-8 | 3.67e-8 |
| FP32 corrected divergence (s⁻¹) | 1.19e-7 | 4.77e-7 | 4.77e-7 |

The local divergence limit is 1e-6 s⁻¹ for this direct solve. Across all topologies, FP64 corrected
divergence peaked at 2.22e-15 s⁻¹. Removing the largest weight kept rank 3 and produced a raw partition
defect of about 0.2893. Restricting the stencil to a plane reduced rank to 2 and produced defects of
0.2301–0.2374. Both masks require PIC under the fixture policy. Physical fallback and refusal with no
valid carrier are native integration checks.

Measured on Windows with Python 3.13.18 and NumPy 2.2.6. On the pinned source
`b009510852e7b306e8abc8ecab9828c7f97a2d52`, a Release MSVC 19.44.35229 CPU build passed all 457 tests, including the 16
under `MacTransfer3D.*:MacPressure3D.*`. The executable SHA-256 was
`da65c4ef08be127e67a3be82080e0e786d5b6416fba5193f558cccb20f7cf806`. Reproduce with the
[Windows native build procedure](../native/README.md#cpu-only-build-on-windows) and run:

```powershell
& ./out/build/windows-release-cpu/fluidity_native_tests.exe '--gtest_filter=MacTransfer3D.*:MacPressure3D.*'
```

These bounded cases support moving the cubic-normal candidate into native parity. Carry forward the worst
edge and corner pressure cases, both orientation signs, the two masks and the incidence corruption
counterexample. Production qualification still needs physical fallback, multilevel topology, remap,
F3/F4, and a matched-error wave benchmark. See the
[M3 plan](../superpowers/plans/2026-11-11-m3-adaptivity-and-band.md) for the M2 integration dependencies.
