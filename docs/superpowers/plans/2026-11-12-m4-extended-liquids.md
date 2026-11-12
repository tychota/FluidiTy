# M4 Extended Liquids Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Qualify M4 extended liquids against §6, §8.3, §9.4, §13 and §15.

**Architecture:** Use the same tensor strain and positive face-inertia metric for both viscous time integrators, and carry all new persistent physics through the accepted-state transaction. Use the bounded CPU/Python operators and proof fixtures to validate the algebra before integrating CPU and CUDA stages.

**Tech Stack:** C++20, CUDA, NumPy/SciPy, Lean/mathlib, Blender.

**Spec:** docs/superpowers/specs/2026-11-07-fluidity-design.md and docs/references/formal-contracts.md.

## Baseline and approaches

Baseline is the locally available origin/main at 563a33e5fdd9488dfac9a2daa2a934226017e9b2. Local main is an ancestor 87 commits behind. The milestone builds on the existing fractional projection. A coupled Stokes solve becomes required if the fixed traction gate fails. Start with bounded algebra so certification and finite-compliance oracles can run while adaptive integration and surface qualification progress.

The M2 bounded wave reconstruction fix e0ca6aa is already in the baseline. Full multi-phase reconstruction and the wave qualification receipt are separate dependencies. M3's integration branch 1af7b0f includes face ownership and CUDA allocation accounting beyond the baseline. Recheck ancestry and production evidence before consuming either. Uniform operator/proof cases do not require the adaptive integration, while adaptive qualification requires production coarse/fine strain, matching face inertia, remap and rollback. Shared interface publication requires phase-aware reconstruction and Blender asset/cache integration. M2's single-phase reconstruction alone does not establish that geometry gate.

## Global Constraints

Use the accepted-state transaction and preserve the zero-compliance path. Require real CUDA and separate proof, independent oracle, differential and end-to-end evidence. Use stable substance and phase IDs. Fix tolerances before collecting qualification measurements.

## Review Focus

- Cut dual masses and highly cancelling stiffness contributions can invalidate the explicit branch selection.
- A sealed compliant component retains physical mean pressure. The particle-free core retains advected history.
- Split and merge retries preserve cavity inventory and checkpoint identity.
- Unequal-density constituent transfer conserves mass and publishes volume fractions keyed by stable substance ID.
- A phase-pair interface appears exactly once, with consistent orientation through projection and cache publication.

## Slices

1. **Viscosity and F7.** Begin with an independent exact-rational bound for finite strain rows. Prove the bound dominates the spectral radius, and separately prove the homogeneous explicit contraction and both forced energy identities. Cover a small dual mass, discontinuous weights, an exactly known bound rounded downward by float64, wall work and implicit residual. Use the existing strain and step fixtures under reference/tests/test_viscosity.py and native/tests/test_viscosity.cc. Compare CPU/CUDA against the independent bridge before claiming a production certificate. Add 3D geometry and stage 9 integration using the pressure face inertia and boundary lift. Measure traction for ratios 1, 10 and 1000 on planar extension and Couette at two resolutions and halved timesteps. Predeclare the stress scale, vanishing-traction floor and maximum error in each fixture, with 5% RMS. Promote to coupled Stokes on failure.
2. **EOS and compliance.** Use §8.3 Hencky and Tait n=7 with stable log/exponential formulas, finite-c predictor, reference-volume-weighted sensor and exact-infinite path. Exercise F6 using compliant piston and pressure defect, then pulse, standing wave, hydrostatic compression and water hammer. Persist accepted pressure history through core, rollback and restart.
3. **Trapped air.** Track closed components with candidate resolution R_eq ≥ 2h, absolute pressure with γ=1.4 and acoustic bound dt ≤ 0.5/ω_cav. Validate submerged hydrostatic/Laplace equilibrium, volume response, venting, split/merge and sub-resolution loss. Check a single pressure/compliance contribution and retry isolation.
4. **Multiple liquids.** Publish volume fractions by stable constituent ID and phase-specific density. Test conservative transfers, unequal-density remap, independent oil and water support, oriented two-phase mobility/jumps and replay.
5. **Tension and shared interface.** Use a single oriented phase-pair geometry and one capillary jump. Validate sphere jump and decreasing parasitic currents, drop l=2 within 5%, and the clear water/dyed water/oil headless bake and extension-disabled material reopen. Exclude triple junctions until a tested method supports them. Variable σ/Marangoni is deferred by §6.3.
6. **Final candidate.** Run the locked CPU and CUDA validation presets, proof build and axiom audit, real-device SM and sanitizer checks, replay and applicable full-job gates. Record source, fixture and executable identity, commands, exits and receipts. Review the whole branch and fix acceptance-impacting findings before any milestone completion claim.

## First red/green check

- [ ] In reference/tests/test_viscosity.py exercise a one-face weight-1 strain with inertia 3. Its exact certificate is 1/3 while binary64 rounds the estimate down. Require the returned certificate to enclose 1/3 when converted back to an exact rational.
- [ ] Run the focused test and verify the missing certificate fails.
- [ ] Implement the independent certificate in reference/src/fluidity_reference/viscosity.py. For the PSD mass-scaled operator, use its trace as an exact rational upper bound, summing the squared strain contributions divided by their positive face inertia. This intentionally trades tightness for independence from assembly and square roots.
- [ ] Run the viscosity suite and reference suite. Commit the bounded oracle and this plan after verification.


## Evidence ledger

- Baseline 563a33e: existing Python 3.13 environment, with PYTHONPATH set to reference/src. Command: python -m pytest reference/tests -q -p no:cacheprovider. Exit 0, 288 passed.
- First slice RED: python -m pytest reference/tests/test_viscosity.py -k exact_trace_certificate -q -p no:cacheprovider. Exit 1, both regressions failed because the certificate function was absent.
- First slice GREEN: python -m pytest reference/tests/test_viscosity.py -q -p no:cacheprovider. Exit 0, 22 passed.
- First slice full reference: python -m pytest reference/tests -q -p no:cacheprovider. Exit 0, 290 passed.
- The checked-in reference uses existing local packages for these runs. The installed uv 0.13.6 was rejected by the repository's >=0.13.7 requirement before the locked setup.

- F7 theorem slice RED: python -m pytest tools/tests/test_verify_proofs.py -k f7_contracts -q -p no:cacheprovider. Exit 1, four missing inventory targets.
- F7 theorem slice GREEN: python -m pytest tools/tests/test_verify_proofs.py -q -p no:cacheprovider. Exit 0, 34 passed.
- Lean 4.34.1, pinned Mathlib: lake --wfail build completed 2,451 jobs in /var/tmp/fluidity-m4-proof.DmTffk/proofs. lake env lean Fluidity/Viscosity.lean confirmed all four closures use only propext, Classical.choice and Quot.sound. Retained output: out/m4-f7-lean.log.

## Predeclared traction fixtures

Use a unit square in the x-y plane, unit depth, density 1 kg/m³ in both phases, μ_a = 1 Pa s, μ_b / μ_a in {1, 10, 1000}, L = 1 m and U = 1 m/s. Use h = L/16 and L/32, and compare dt = 0.005 s and 0.0025 s at both resolutions. Use the same exact boundary traces at each resolution.

- **Planar extension:** center the square at the origin, interface x = 0 with normal +x. Set strain rate a = U/L and velocity (a x, −a y). Set pressure in phase i to 2 μ_i a + S, where S = 2 μ_a U/L. The exact normal traction is −S and tangential traction is zero. The pressure jump exactly cancels the viscosity jump in normal stress. Drive from the analytic boundary velocity and use mean pressure consistent with the reference.
- **Couette:** interface y = 0 with normal +y. Set tangential wall speeds 0 at y = −L/2 and U at y = L/2, with periodic x. Use piecewise-linear tangential velocity with slopes τ/μ_i, where τ = U / (L/(2 μ_a) + L/(2 μ_b)). The exact tangential traction is τ and the normal traction is −S at pressure S. Freeze the planar geometry and use matching analytic initial data.
- Sample both one-sided tractions at each interface face and weight samples by physical interface area. For each side and component, require RMS error ≤ 0.05 S and maximum error ≤ 0.10 S. For a vanishing reference component, use the absolute floor 0.01 S and require both RMS and maximum to be within that floor. Apply these criteria to all ratios, resolutions and timesteps. A failed fractional gate requires a coupled Stokes path before production qualification.

## Production dependencies to resolve

- Stage 9 needs a sparse 3D tensor-strain assembly with prescribed-wall lifting and the pressure face inertia. The current bounded native solver is capped at 64 velocity columns and has no substep caller. The trace certificate is intentionally a bounded reference and can select implicit earlier than a tighter row-sum enclosure as the component grows.
- M3 integration commit 1af7b0f is outside the baseline and carries geometric face ownership, exterior subfaces and allocation measurement. Adaptive M4 traction needs an accepted production topology with those capabilities and coarse/fine strain quadratures. Uniform reference qualification can continue independently.
- The finite-compliance pressure field currently uses scene bulk modulus and the stored pressure of a matching cell in liquid_substep_3d.cc. EOS qualification needs an advected full-grid predictor and transaction-safe cell history, with remapping and a same-phase particle sensor. The one-cell reference piston deliberately tests the constitutive and pressure algebra independently.
- The current boundary operator distinguishes liquid and air, and cache/reconstruction has no phase-pair interface ownership. Cavity lineage needs closed-air component inventory and vent detection on physical geometry. Multi-liquid qualification needs shared oriented geometry, constituent inventories through remap/cache and the supported headless Blender workflow.

## Additional receipts

- Independent F7 verification passed using the pinned Lean exporter and nano-lean checker. The complete 46-target axiom, export and checker receipt is out/m4-proof-verification/evidence.json.
- Native F7 RED: fluidity_native_tests.exe --gtest_filter=Viscosity.RoundedBoundCannotAdmitAnUnstableExplicitStep. With inertia 7 and dt one ULP above 14, the old branch returned OK and increased energy. The old bound was 0.14285714285714282.
- Native F7 GREEN: MSVC 19.44 Debug, fluidity_m4_focused.exe --gtest_filter=Viscosity.*. Exit 0, 30 passed. CMake built the production library and test object, then link.exe linked the Ninja response file to a separate test binary while the original executable was in use. Log: out/m4-f7-native.log.
- clang-format 19.1.5 --dry-run --Werror passed on all tracked native files.
- EOS RED: python -m pytest reference/tests/test_eos.py -q -p no:cacheprovider. Exit 2, EOS oracle was missing.
- EOS GREEN: the locked Python reference suite passed 302 tests. EOS covers Hencky and Tait n=7, finite acoustic coefficients, exact infinite sound speed, the Tait domain boundary, and small predictor log volume ratios. The one-cell compliant piston retains the physical pressure and demonstrates convergence at 16, 32 and 64 steps for each law. Log: out/m4-eos-reference.log.
