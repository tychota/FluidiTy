# M1 uniform MAC transport evidence

Retain the production quadratic transport baseline. Carry mixed cubic-normal/quadratic-transverse transport into a future controlled comparison and keep RT0 as the bounded whole-cell control. In that comparison, supply the same finite-volume projection and keep the staggered quadratic momentum transfer, fitted affine rows, contact and old/new history rules fixed. Compare the positional residual `β=0` with `α` fixed before separately changing the stored-velocity history. [Frozen method plan](../superpowers/plans/2026-11-10-m1-method-evidence.md) · [Fei et al., Eqs. 11–13](https://raymondyfei.github.io/asflip/main_reduced.pdf)

## Exact arithmetic and float64 reference

For complete uniform support, the centered cubic derivative is the difference of two half-shifted centered quadratics. Reindexing each normal derivative in the tensor sum gives the cell-centered quadratic convolution of the outward face difference divided by `h`. Thus the mixed trace inherits the local bound from its contributing cell divergences. RT0 gives the containing cell's divergence and its constant normal trace agrees with the shared face datum. Its transverse components can jump at faces. Individual face integrals of a mixed field require a separate condition. Lean checks the finite signed reindexing with zero exterior weights and the centered cubic's formal polynomial derivative in normalized coordinates. The numerical bridge combines them with the transverse tensor factors and physical scaling by the inverse cell width. [Signed reindexing proof](../../proofs/README.md#finite-signed-mac-trace-reindexing) · [Cubic derivative proof](../../proofs/README.md#centered-cubic-normal-derivative) [Schroeder et al., Eqs. 6 and 9–16](https://www.cs.ucr.edu/~craigs/papers/2021-div-free/paper.pdf) · [Roy-Chowdhury et al., §2](https://www.cs.ucr.edu/~craigs/papers/2022-interp-variants/paper.pdf) · [Formal contracts](formal-contracts.md)

The float64 patches at four spacings met `2e-13 m/s` in value and `2e-11 s⁻¹` in gradient for both splines on signed affine data and for RT0 on normal-only affine data. The mixed trace matched the independent cell convolution, and the RT0 trace matched the containing-cell target, within `2e-11 s⁻¹` on the curl and varying nonzero-divergence cases. The all-quadratic curl control had maximum absolute traces `2.61316, 5.22632, 10.45264, 20.90527 s⁻¹` from coarse to fine. That control holds discrete face amplitudes fixed, so its gradient scale grows as `1/h`. RT0's transverse affine shear control missed `2 s⁻¹` at every spacing and its velocity error was `h × 1 s⁻¹` at the sampled face.

### Regular shared-face flux

At each spacing and normal axis, take one whole face with signed MAC index `(-2,1,-1)` and orient it toward the positive normal. Use local physical coordinates `(n,s,t)` in metres about its center. The prescribed normal velocity is `−3/4 + n − s + t/2 + n² + 2s² − t²/2` in m/s, with coefficients in the corresponding SI units. The first tangential velocity is `−(1+2n)s` and the second is zero. The face samples come from the analytic field at MAC anchors. The neighbors in each direction have unequal normal speeds and the outward MAC differences cancel exactly on all 27 supporting cells. The mixed trace is checked at the quadrature points in the maximum absolute norm with a tolerance of `2e-11 s⁻¹`.

The tensor Gauss rule splits each tangent at the transverse quadratic spline knots. They are the two face edges, so each tangent has one interval of length `h`. Two interior Gauss points per interval integrate the piecewise quadratic normal trace exactly in exact arithmetic. The RT0 rule samples the positive-side trace at the face and the negative-side trace one float64 step into that cell. It compares each integral with the signed stored target `−3h²/4`. The analytic continuum face integral adds the tangential second moments `h²/12` to the point sample, giving a flux excess of `h⁴/8` in SI units. Mixed reconstruction also adds `h²/3` from its normal cubic second moment and `(2−1/2)h²/4` from its transverse quadratic second moments to the normal speed. Thus the signed mixed excess is `17h⁴/24` relative to the continuum integral and `5h⁴/6` relative to the stored target. Each flux comparison uses absolute error in m³/s with tolerance `1e-15`.

| `h` (m) | Mixed minus stored flux (m³/s) |
| ---: | ---: |
| 1/8 | 2.0345052e-4 |
| 1/16 | 1.2715658e-5 |
| 1/32 | 7.9472860e-7 |
| 1/64 | 4.9670537e-8 |

RT0 matches the stored flux on both sides. Mixed matches the analytic excess at all three orientations with a maximum absolute trace of `6.33e-15 s⁻¹`. This gives a signed counterexample to individual face-flux preservation under trace commutation. The face flux, spatial value and gradient errors below, and coupled work are separate observables.

Run `uv run --offline --project reference --locked pytest reference/tests/test_mac_transport_3d.py -q -s -k single_face_flux` for the signed flux and trace records.

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

### Polynomial fit degree at fixed support

The weighted affine and total-degree quadratic fits use the same complete signed MAC donors and
cubic tensor weights at each spacing. With binary64 analytic data, matching-degree reproduction
meets 1e-12 m/s for value and 1e-10 s⁻¹ for gradient on all tested phases. The maximum observed
errors are 8.88e-16 m/s and 8.45e-15 s⁻¹. A quadratic spline tie reduces one quadratic design to rank
9 of 10; the deficient fit is reported without publishing an ambiguous quadratic value.

On the smooth field at h = 1/64 m, value RMS is 2.36e-4 m/s for the affine fit and 1.10e-6 m/s for
the quadratic fit. The finest measured value convergence rates are 2.00 and 3.01. The signed
quarter-cell cubic control explains the third-order bias without extrapolating from the sweep.
Both fitted gradients have RMS 4.61e-4 s⁻¹ and rate 2.00. Their agreement follows from the complete
cubic moment symmetry in this fixture. The separately differentiated cubic interpolation gives a
distinct gradient, with RMS 4.90e-4 s⁻¹. These are uniform complete-support results; the comparison
does not establish a gradient, cost, or stability gain for the coupled solver.

Reproduce with `uv run --offline --project reference --locked pytest reference/tests/test_mac_fit_degrees.py -q -s`.

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

## M1 closure review

The moving-piston transport regression survived CPU/CUDA parity because the scheduler gave both backends the same unfiltered donor history. The pressure solve stayed within its defect budget while the particle-volume drift reached 2.11%. The selected-cell donor filter restored the archived trajectory and reduced peak particle-volume drift to 0.051% on both piston runs. Keep pressure residual, per-particle volume and receiving volume as separate tests. The mixed-history fixture now exercises both actual CUDA groups, compares grouped and scalar CUDA checkpoints, and checks the CPU result.

CUDA qualification separates the physical solver from the execution path. Record the exact binary and scene, SM120 device receipt, ordered attempt outcome, and cache replay. Use clean native builds after shared-record layout changes. Run ordinary CUDA tests with the deliberately invalid device call, and the qualified named-pipe memcheck roster with that call excluded. Treat a failed device-exclusivity or artifact check as missing evidence. Use matched trajectories and published-frame times to judge optimization.

The pinned e815a2e hydrostatic CPU and RTX runs both completed 241 frames. At 2 s the maximum speed was 3.956e-7 m/s, below 1e-3 m/s. Maximum point-volume drift was 1.554e-7 relative, below 0.01. The frozen static-sphere rerun exposed a separate capillary failure on both backends. Fine interior pressure error was 5.288% against 5%, and parasitic speed was 0.01614 m/s against 0.004472 m/s. CPU and RTX agreed to about 2e-11 m/s on that speed. Both simulations finished, so report the failed physical comparison separately from successful execution.

For the piston, let R be the above-plane gain plus the above-plane wetted sweep. Its signed difference from the full wetted sweep is the reconstructed total-volume change plus the change in below-plane void and the dry sweep below the plane. The pinned fixed-radius primary deficit was about 6.22e-4 m³ and tracked the loss of reconstructed total volume. The below-plane terms were negligible. The diagnostic reports these terms separately and the frozen verdict keeps its original thresholds.

A manufactured determinant-one motion clarifies the role of refinement. At h = 1/8, 1/16 and 1/32 m, the receiving error with wall-fitted vertices fell from 0.0512% to 0.0149% to 0.00382%. With the same motion and center extrapolation, the finest reconstruction missed about half the wall area even though its maximum gap shrank to 11 µm. Exact-plane contact is sensitive to the sign of that gap. Refine a uniform control before proposing adaptive resolution. Qualify the wet-wall reconstruction and signed swept volume independently, including a genuinely detached interface and a partially wet or receding wall. Coarse/fine transfers will need their own conservative flux checks.

Before M1-B, re-run the pinned scene and device matrix on the final implementation. The receiving-volume and public oscillating-drop gates still require supported fixes. M2 and M3 should inherit independent material, interface, and wall budgets so reconstruction or remapping errors remain visible.
