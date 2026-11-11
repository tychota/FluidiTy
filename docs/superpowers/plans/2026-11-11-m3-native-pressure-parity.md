# M3 Native Pressure Parity Implementation Plan

> Use superpowers:executing-plans to implement this slice and verification-before-completion before reporting it.

**Goal** — Apply native integrated pressure assembly and correction to the measured 3D coarse/fine edge and corner graphs.

**Baseline** — Merge commit `601d775e5f989bbad38b7c13d518fc0be9df89e1` from PR #53. The Python receipt supplies the four pressure solutions and signed balances.

## Scope

Keep uniform MAC ownership and the public solver entry points intact. An internal explicit graph provides one stable face ordinal and two distinct cell endpoints per open subface. The checked builder computes conductance from area, mobility, spacing and duration and uses the existing integrated RHS, gauges and evaluation. This slice qualifies the native FP64 original operator and correction on a bounded graph. The next pressure slice connects checked adaptive topology to the bounded CPU and CUDA projection paths and compares the actual iterative solves with these same reference cases.

## Verification

- [x] Add four edge/corner cases covering both orientations. Check predicted per-plane signed flux, compatible gauge, a constant-pressure null mode, the independently solved candidate pressure, every volume-scaled defect and the signed residual identity. Check repeated face ownership, reversed endpoints with unchanged flux, an invalid endpoint, an invalid coefficient and the cell extent.
- [x] Run `fluidity_native_tests --gtest_filter=MacPressureExplicitGraph3D.*` with a stub that returns Unimplemented. Expect the valid graphs and status assertions to fail.
- [x] Implement the checked explicit graph builder and share finalization with the uniform prepared system.
- [x] Run focused tests and the full native CPU suite. Verify the Python fixture on the same pinned worktree and clang-format 19.1.5 on the changed native sources.
- [ ] Open a focused PR and wait for its platform and proof checks. Track M2 separately through PR #52 and its successors.

## Results

The four edge/corner graphs matched the direct NumPy pressure solutions. Both native maximum volume-scaled residual and physical defect passed the 1e-12 s⁻¹ fixture bound. The constant-pressure operator product was exactly zero. The per-plane coarse flux was +0.25 m³/s and the aggregate fine flux was its negative. Reversing one subface without changing its velocity produced a defect above the negative-control threshold.

The initial fixture failed with the unimplemented builder. Independent review then identified a lossy subnormal intermediate that the first implementation accepted. A focused regression reproduced the issue and passed after the explicit graph builder began rejecting unrepresentable mobility and conductance arithmetic. The regression leaves the uniform path's existing coefficient policy intact.

The MSVC Release build passed 29 focused pressure tests and all 504 tests in the native executable. The standalone Python qualification and its three focused tests passed on the same source baseline. The tracked native files passed clang-format 19.1.5.
