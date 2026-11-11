# M3 Adaptivity and Narrow Band Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for the qualification slice. Steps use checkbox syntax.

**Goal** — Qualify 3D 2:1 MAC transitions independently and use the measured results to sequence remap, resampling, and the Eulerian-core handoff.

**Architecture** — Extend the existing dyadic spline oracle to an intersection of one, two, or three aligned half-spaces. Keep a same-degree uniform control and an independently assembled integrated-flux graph. Carry successful candidates into the native solver only after separate parity and transaction checks.

**Tech stack** — Python 3.13 and NumPy for the oracle, C++20/CUDA for native work, Lean for exact contracts.

**Spec** — [FluidiTy design, sections 7 and 15](../specs/2026-11-07-fluidity-design.md) and [formal contracts](../../references/formal-contracts.md).

## Scope and baseline

Pin `b009510852e7b306e8abc8ecab9828c7f97a2d52`, the common baseline of the active M2 integrated-stages and completion branches. The 2D oracle provides the cardinal kernels, dyadic masks, and pressure conventions. A direct native port would exercise actual layout earlier, with solver changes obscuring basis failures. The Python fixture isolates those failures and supplies an oracle for a later native port. A uniform-sparse band can be measured independently if adaptive promotion fails.

## Global constraints

- FP32 dimensionless partition and linear-moment defects ≤ 1e-6. Keep the 2D precedent of 2e-6 for normalized gradient defects and 1e-12 for FP64 transfer checks. Use 1e-6 for normalized affine value and 2e-6 for normalized affine gradient in FP32, with 1e-12 in FP64.
- Constant-pressure integrated relative residual ≤ 1e-6 with a declared 1e-12 m³/s floor. Use the identical incidence and coefficients for pressure assembly and correction. The bounded fixture uses 1e-6 s⁻¹ for volume-scaled divergence. Native qualification uses the matched solver's actual gate.
- Full wave ≤ 3 h and simultaneous GPU allocations ≤ min(0.70 × total, 0.80 × free at job start). Temporary uniform-sparse band ≤ 6 h under the same memory ceiling leaves adaptivity open.
- Compare against the pinned uniform 3D solver at equal error. Include cage/native Multires only if the reference benchmark calls for it.

## Review focus

- Later nonfinite queries must fail even after an earlier finite maximum.
- Incomplete full-rank support and rank-deficient support need distinct dispositions.
- Sign reversal on one shared face must be detected with mixed signed fluxes.
- A missing occupancy product cannot manufacture an interior or fill a cavity.
- Topology and ownership changes remain trial-owned through every failure injection.

## Task 1 — Standalone qualification

`tools/adaptivity/qualify_coarse_fine_3d.py` owns the query fixtures, dtype-specific report, signed finite-volume graph and CLI. `tools/tests/test_qualify_coarse_fine_3d.py` owns regression checks. Reuse the one-dimensional oracle in `reference/src/fluidity_reference/experiments/coarse_fine_basis.py`.

- [x] Add a test for `qualify() -> dict` that checks all precision/topology rows, mixed support, full-rank incomplete support, coplanar support, sign-sensitive flux and an incidence corruption control. Run the focused pytest file and observe a missing implementation failure.
- [x] Evaluate all three staggered components over aligned planes, edges and corners, signed rotations, negative coordinates, widths 0.5 and 2 m, and knots and off-knot positions. Keep original coarse Greville anchors. Add positive child terms by the first axis where containment fails. Compare the uniform cubic-normal control and retain the quadratic-normal nesting obstruction from 2D.
- [x] In a sealed graph, connect a 2×2×2 fine block to one coarse cell per incident plane via four uniquely owned subfaces per plane. Use both flux signs, compute fine/coarse sums independently, and check constant-pressure residual, corrected divergence, and a deliberately reversed incidence entry.
- [x] Report observed support rank and raw moment defects before normalization for a removed dominant row and a coplanar mask. Distinguish PIC-required from rank-deficient PIC-required. Qualification does not exercise physical fallback.
- [x] Re-run the fixture and 2D controls, record environment, gates, measurements and executable commands in `docs/adaptivity/README.md`. Verify native uniform controls with a provenance-qualified executable where available.

The 3D half-space matches the 2D marginal. All six dtype/topology rows passed, and the pinned uniform CPU suite passed 457 tests. The closest FP32 gate was corrected divergence at edges and corners (4.77e-7 s⁻¹ of the 1e-6 s⁻¹ fixture budget). Prioritize the measured worst pressure cases and both orientation signs in native parity. The missing-row case retains rank 3 but loses partition, so a rank-only guard is insufficient. The coplanar case loses rank. Both require separate physical fallback tests. See the [measured results](../../adaptivity/README.md#m3-3d-coarsefine-qualification).

## Task 2 — Native parity, masking and conservative remap

- [ ] Use the measured worst queries as C++/CUDA fixtures. Add rotated signs, negative coordinates, closed-knot ownership, deeper level junctions, aperture/pressure checks and perturbations near rank loss. Require stable rank handling, PIC reset on valid low-mass carrier, and explicit refusal when even PIC has no valid support.
- [ ] Restrict mass, reference/current volume, momentum, constituent inventories and enthalpy by sums. Prolong conservatively with bounded positivity, sum signed fine fluxes once and recover intensives by the selected EOS. Test refine/coarsen round trips, nontrivial signed flux and sources, pressure warm-start, and halo/hysteresis vetoes for cavities, sheets and gaps below 4h.
- [ ] Inject failure after topology, remap, and warm-start stages. Require unchanged committed topology, inventories, histories, IDs, source/RNG counters and accepted time. Compare interrupted/restarted and uninterrupted accepted state hashes.

## Task 3 — F3/F4 and split/merge

- [ ] Compile exact Lean F3/F4 contracts and keep numerical bridge results separate. Exercise symmetric 1→8 covariance 3h²I/16, angular moments, and the cubic deficit m h² Σ C_aa²/32 with pure diagonal and off-diagonal C.
- [ ] Check F4 per-row covariance PSD and positive-definite support, counterstreaming, mixture volume weights J V0 and EOS admissibility. Refuse uncertain eigenvalues, wrong phase/body/side, masked/transition supports and invalid children. Test retained-particle energy (initial 1% block budget) and separately declare the angular budget before running.
- [ ] Replay with canonical candidate ordering and stable IDs. Verify refusal preserves the parent/group and rollback preserves all committed state.

## Task 4 — Particle band and Eulerian interior

- [ ] Run independent deep-tank, translating-interface, descending-interface and true closed-cavity fixtures. Start with 6h band and 2h overlap and certify support and motion margins. Blend before the old-grid snapshot. Assign each volume once, reseed fully liquid cells only, and compare component inventory with signed flux/source and geometric drift.
- [ ] Deliberately strip interior particles. Assert no false inner surface and preserve the closed cavity and component lineage. Missing occupancy must fail clearly. Inject failures during transfer and reseeding and verify deterministic replay and rollback.
- [ ] Integrate once M2 publishes authenticated accepted occupancy/sign, component targets, obstacle closure, and the accepted MotionInterval with timestamps and valid carrier coverage. Mesh's geometric field, persistence, and replay are required for real false-inner-shell, cavity lineage and moving-interface correspondence checks. Manufactured occupancy and motion suffice for the independent fixtures. Whitewater source/volume lineage and Ocean Boundary settings are required for the integrated wave.

## Task 5 — Equal-error wave qualification

- [ ] Pin the executable, fixtures, scene/seed, device, architecture and cache policy. Fix error metrics and matching thresholds before short uniform/adaptive runs. Match physical inputs and compare at the same error using volume and cavity drift, surface and trajectory error, physical divergence and source balance.
- [ ] Record unprofiled launch-to-exit, publication intervals, retries, active levels, stencil classes, remap work and peak arena bytes including old/new topology, live stage products and in-flight copies. Follow the [CUDA measurement procedure](../../native/cuda-optimization.md).
- [ ] Run the full wave after short-run quality and provenance pass. Keep adaptive promotion open if only the uniform-sparse fallback meets its 6 h gate.
