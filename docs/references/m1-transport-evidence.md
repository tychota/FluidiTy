# M1 uniform MAC transport evidence

Retain the production quadratic transport baseline. Carry mixed cubic-normal/quadratic-transverse transport into a future controlled comparison and keep RT0 as the bounded whole-cell control. In that comparison, supply the same finite-volume projection and keep the staggered quadratic momentum transfer, fitted affine rows, contact and old/new history rules fixed. Compare the positional residual `β=0` with `α` fixed before separately changing the stored-velocity history. [Frozen method plan](../superpowers/plans/2026-11-10-m1-method-evidence.md) · [Fei et al., Eqs. 11–13](https://raymondyfei.github.io/asflip/main_reduced.pdf)

## Exact arithmetic and float64 reference

For complete uniform support, the centered cubic derivative is the difference of two half-shifted centered quadratics. Reindexing each normal derivative in the tensor sum gives the cell-centered quadratic convolution of the outward face difference divided by `h`. Thus the mixed trace inherits the local bound from its contributing cell divergences. RT0 gives the containing cell's divergence and its constant normal trace agrees with the shared face datum. Its transverse components can jump at faces. Individual face integrals of a mixed field require a separate condition. The mixed finite reindexing, one-dimensional derivative identity and physical scale remain outside the checked Lean results in this slice. [Schroeder et al., Eqs. 6 and 9–16](https://www.cs.ucr.edu/~craigs/papers/2021-div-free/paper.pdf) · [Roy-Chowdhury et al., §2](https://www.cs.ucr.edu/~craigs/papers/2022-interp-variants/paper.pdf) · [Formal contracts](formal-contracts.md)

The float64 patches at four spacings met `2e-13 m/s` in value and `2e-11 s⁻¹` in gradient for both splines on signed affine data and for RT0 on normal-only affine data. The mixed trace matched the independent cell convolution, and the RT0 trace matched the containing-cell target, within `2e-11 s⁻¹` on the curl and varying nonzero-divergence cases. The all-quadratic curl control had maximum absolute traces `2.61316, 5.22632, 10.45264, 20.90527 s⁻¹` from coarse to fine. That control holds discrete face amplitudes fixed, so its gradient scale grows as `1/h`. RT0's transverse affine shear control missed `2 s⁻¹` at every spacing and its velocity error was `h × 1 s⁻¹` at the sampled face.

### Manufactured spatial field

For each `h`, take 512 equal-volume strata in the fixed physical box `[-1/2, 1/2)³ m`. Sample the middle cell of each `1/8 m` stratum, choosing the positive-side middle cell at a tie. For signed stratum indices `(i,j,k)` from `-4` through `3`, take phase indices `(i+j, j+k, k+i) mod 4` into `(0, 1/4, 1/2, 3/4)`. The donors are analytic point samples at the signed MAC anchors. The targets are the one-metre sinusoidal velocity and its analytic component-row gradient. RMS weights the strata and the reported scalar entries equally.

| Reconstruction | Value RMS / max at `h=1/64 m` (m/s) | Gradient RMS / max at `h=1/64 m` (s⁻¹) | Value RMS rates on successive halvings | Gradient RMS rates on successive halvings |
| --- | ---: | ---: | --- | --- |
| Quadratic | 0.000977648 / 0.00211325 | 0.00504921 / 0.0157586 | 1.865, 1.983, 1.991 | 1.978, 1.977, 1.998 |
| RT0 | 0.0121621 / 0.0434988 | 1.48987 / 5.61731 | 1.124, 0.972, 0.986 | 0.094, 0.015, 0.008 |
| Mixed | 0.00114042 / 0.00246484 | 0.00590321 / 0.0180218 | 1.849, 1.981, 1.991 | 1.949, 1.977, 2.000 |

Both splines met the value bound `k² h²/3 + 2e-13 m/s` at each `h` for amplitude `1 m/s` and `k=2π m⁻¹`. Mixed incurs a larger smooth-field bias here and retains the commuting trace on the separate discrete fixtures. RT0's transverse derivative error persists under refinement. With wavelength `2h`, the normal sine is zero at all anchors up to roundoff for all three reconstructions. Across the sweep, velocity RMS and maximum errors were `0.408248` and `1 m/s`. Gradient RMS grew from `8.37758` to `67.0206 s⁻¹` as the target wavenumber increased, giving rates near `−1`.

### Affine time sweep

At fixed `h=1/64 m`, integrate `u=(x,−y,0) × 1 s⁻¹` for one second from three signed positions. The analytic flow scales `(x,y,z)` by `(exp(1),exp(−1),1)`. For `N=1 s/Δt`, the independent discrete position scales by `((1+Δt·1 s⁻¹)^N,(1−Δt·1 s⁻¹)^N,1)`. The Euler volume ratio is the product of `det(I+Δt L)`, with independent target `(1−(Δt·1 s⁻¹)²)^N`. The exponential material J accumulates `exp(Δt tr L)` and targets one, as does the exact geometric flow ratio.

| `Δt` (s) | Position RMS / max error to exact flow (m) | Euler geometric volume defect |
| ---: | ---: | ---: |
| 1/8 | 0.0345300 / 0.0762487 | 0.118374 |
| 1/16 | 0.0181702 / 0.0401767 | 0.0607019 |
| 1/32 | 0.00933161 / 0.0206459 | 0.0307816 |
| 1/64 | 0.00473022 / 0.0104684 | 0.0155054 |

All three reconstructions agree at the displayed precision. Peak position error occurs in `x` from `(-0.5, 0.25, -0.125) m`. The largest discrepancies from the independent discrete position and volume targets were `4.45e-16 m` and `1.56e-15`. The maximum material J error was `1.67e-15`. The finest dyadic rates were `0.980` for position RMS and `0.989` for Euler volume defect.

Reproduce the full records with `uv run --offline --project reference --locked pytest reference/tests/test_mac_transport_3d.py -q -s -k "curved_spatial or affine_euler"`. The 36 JSON rows include component RMS, maximum errors and their locations, and dyadic rates. The ignored evidence directory `out/m1/method-evidence/task-2` contains raw observations and test logs. For the negative controls, shifting the normal donor anchors by `h/2` failed both spline value budgets and substituting exponential J for the Euler determinant failed all three temporal checks.

## Separate coupled observables

| Observable | Fixture or budget | Failure isolated by the comparison |
| --- | --- | --- |
| Mass | Compare particle inventory to each staggered component's deposited mass, before and after quantization. F2 gives the unmasked exact moment, with per-channel nearest-rounding budget `Nq/2` under F8 and separate formation/decoding error. | A clipped or missing donor changes the inventory represented on a face lattice. |
| Linear momentum | Compare particle totals and summed component face momenta before projection, then account for force, pressure and contact impulses. Test centered clipped masks separately. | A first-moment error or a changed history correction can alter the transfer and its impulse. |
| Angular momentum | Compare staggered antisymmetric moments using each row's covariance, with matched weights and relocation in the exact transfer fixture. | Mask centering changes the angular centroid term. A different relocation, contact or quantization needs its own balance. [Ding et al., Eqs. 4–10 and §4.3](https://www.cs.ucr.edu/~craigs/papers/2019-mac-apic/paper.pdf) |
| True gradient and face flux | Use the affine, curl and source-divergence patches for `L` and the finite-volume residual for flux closure. Compare a wetted piston face integral to the same frozen datum in RHS and correction. | A fitted affine row can differ from `L`. A small trace can coexist with incorrect individual face or wall flux. |
| Material J | Use the affine strain time sweep and separate dam diagnostics for accumulated exponential J. | Mismatched carriers or projected-history donors can inject trace and J drift. |
| Geometric volume | Use the analytic Euler map and the separate receiving-volume and scene-volume gates. Track rendered volume as a geometry output. | Euler map distortion, wall receipt and reconstruction geometry can each diverge from material J. |
| Capillary response | Cross sampled/exact sphere curvature and ordinary/checked rest-pressure seed on the same geometry. Measure pressure jump and parasitic speed separately from frequency and damping. | Static pressure balance can coexist with a poor dynamic oscillation. |
| Work | Budget pressure and moving-wall work against the frozen flux and compare grid, advecting and stored velocities separately. | History or transport changes can redistribute kinetic energy even with matching flux and momentum accounting. |
| Replay | Compare committed snapshots and accept counts across runs, and committed-state equality after injected rejection. | Trial mutation, fallback or donor changes can alter replay despite passing the operator tolerances. |

The archived bounded CPU dam draft (`51c55a3`, `e5a91d0`) reduced peak material drift from `7.55912%` to `0.00008407%` in the guarded RT0 run. It retained quadratic stored velocity and affine rows, paired RT0 advection and J, and bypassed `β`. Projected-history masking (`5e61986`) remains a separate donor control, with free-fall acceleration, fallback and replay as checks. The archived piston (`f35cdb7`) showed wall-versus-solver flux differences near `0.00051%` alongside receiving deficits near `7.62%` and `7.63%`. Exact curvature (`bc174e5`) and checked rest pressure (`65ea18a`) remain separate controls. Draft projected-correction continuation reached `4.905%` frequency error and `0.136` damping on an ellipsoid, while the compact-field early response was near `3.80 rad/s` against `2.024 rad/s`. These are inherited draft diagnostics with their original scope and provenance.

Cuts, moving walls and coarse/fine transitions need separate aperture flux, continuation and support experiments, including nonzero source/compliance where applicable. A higher-degree momentum fit needs a separate matched gather/scatter experiment at fixed kernel and spacing. Coupled acceptance keeps the frozen free-fall, hydrostatic, volume, piston, capillary, pressure and replay gates. The support counts are 6, 81 and 108 samples for RT0, quadratic and mixed. Measure cost on a common real-device workload and compare equal-error or equal-cost cases only where measured ranges overlap. [M1 scene plan](../superpowers/plans/2026-11-09-real-3d-liquid-acceptance-and-throughput.md)
