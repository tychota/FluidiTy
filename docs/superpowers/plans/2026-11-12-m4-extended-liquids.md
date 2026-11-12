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
- Cavity RED: python -m pytest reference/tests/test_cavity.py -q -p no:cacheprovider. Exit 2, cavity oracle was missing.
- Cavity GREEN: six focused cases and the full locked suite (308 cases) passed. Coverage includes submerged-sphere initial absolute pressure, signed capillary lift, compression/expansion, acoustic timestep, and the candidate radius cutoff. Component discovery, topology lineage, and production boundary integration are open. Log: out/m4-cavity-reference.log.
- Multiphase RED: python -m pytest reference/tests/test_multiphase.py -q -p no:cacheprovider. Exit 2, pair and mixture oracle missing.
- Multiphase GREEN: five focused tests passed. Coverage includes clear/dyed-water volume fractions, unequal-density mass conversion, oil in a separate phase, series resistance and signed one-time capillarity/normal viscous jump under both link orientations. The two-cell pressure solve is a reference check. It exercises neither mesh geometry nor cache publication.
- Traction RED: the new extension/Couette tests failed on the missing reference functions. The uniform extension exposed cancellation at a physically zero pressure RHS, so the manufactured fixture projects the velocity increment about the exactly divergence-free initial field.
- Traction GREEN: 36 focused checks and 349 locked reference tests passed. Reproduce the 36 measurements with python -m fluidity_reference.traction. Retained JSONL: out/m4-traction-reference.jsonl.
- Traction ruling: promote the coupled solve. Fractional normal RMS, as a fraction of S, was 5.8365–6.6694 at ratio 10 and 988.4299–993.3092 at ratio 1000. The predeclared limit is 0.05. The coupled reference's largest normal RMS was below 5e-11, and the largest Couette tangential error was below 3e-12. These are 2D manufactured reference measurements.

## Coupled production solve after the traction gate

Use H = M + dt EᵀWE and the exact pressure-link incidence D. For each positive-area unknown, use the same signed link and inertia as pressure. Assemble the wall strain lift once. Internal jump contributes +a_f J_f to momentum, and a Dirichlet exterior pressure contributes the correctly signed missing-endpoint term. The equations are H u − dt Dᵀp = M u* + dt (b_wall + f_jump) and D u + C p = V s − q_wall + C p*. In the single-phase incompressible, homogeneous-boundary control, the pressure term does zero kinetic-energy work.

Use the positive velocity block and a pressure Schur solve on anchored/gauge-restricted components. The current mixed-precision pressure multigrid is a preconditioner candidate for the Schur operator, whose exact action is dt D H⁻¹ Dᵀ + C. Require a measured inner residual budget, true coupled residuals, volume-scaled physical defects, and fixed traction metrics. Reuse the existing pressure-link orientation and successful-attempt publication, and preserve the inviscid disabled stage as an identity. The current repository has no sparse 3D strain assembly or H inverse and no coupled device stage. Start with an independently checked full-cell tensor assembly and an SPD velocity-block solve, then add cut/free-surface and moving-wall quadratures, coupled projection and the production traction receipts.
- Native validator recovery: on snapshot 926b935, CUDA Release passed 596 CPU cases and all six real CUDA CLI cases, then rejected the Ocean case's missing run receipt. The producer now reports the actual accepted frame 10 / step 20 with its CUDA identity, and the validator rejects malformed or fallback records. The Ocean command and raw output are retained in out/m4-cuda-ocean.log. All 579 validator unit tests passed.
- Windows Debug CPU at 926b935 passed 622 native tests and the schema and CLI stages. The final receipt was withheld because pytest reported 1274 passes and one skip. The case test_liquid_capture_provenance.py::test_missing_or_symlinked_file_refuses_qualification requires file symlink creation, which this account received WinError 1314 for. The exact validator remains strict. Retry under a Windows account with file symlink creation available, then rerun tools/native/validate.py against the same candidate.
- Sparse 3D strain RED: the new native tests could not compile without AssembleFullCellMacStrain3D. GREEN: 33 Windows Debug native viscosity cases passed, including signed block ownership, all tensor components on affine flow, harmonic Couette shear at ratio 1000, duplicate cells and a missing face owner. clang-format 19.1.5 passed on all tracked native sources. Log: out/m4-viscosity-3d.log.
- The new sparse assembly supplies full-cell normal strains and interior edge shear. The next native step is the positive velocity-block solve, followed by wall elimination/lifting, cut/interface strain quadratures, and a Schur projection with the pressure face inertia. It is intentionally not registered as a scene capability yet.
- Velocity block RED: the sparse solve test could not compile without SolveMacViscousVelocityBlock3D. GREEN: 34 Windows Debug native viscosity cases passed. The new solve uses the same square-root-weighted rows for the velocity operator and Jacobi diagonal, excludes zero-inertia prescribed slots, and checks the true momentum residual before publishing. The fixture checks an exact two-unknown result, prescribed-slot elimination and iteration-budget rejection. Log: out/m4-viscous-velocity-block.log.
- Fresh review of 563a33e..be62da8 found no confirmed code defect in the reviewed bounded slices. It identified incomplete nonzero shear coverage, addressed by checking independent xy, xz and yz affine strain values along with the rigid-rotation control. Production F7 still needs the floating assembly/action error bridge. Couette's independent reduced solve measures tangential traction, with zero normal defect from the analytic symmetry and prescribed pressure. Production Couette must measure both.
- Release CUDA validation at be62da8 passed 599 CPU, 16 schema, 13 CLI, six CUDA CLI, 11 CUDA mock and 48 real-device tests. All three named-pipe Compute Sanitizer checks reported zero errors across 46 cases. The aggregate receipt was withheld at the same Windows symlink test (1327 Python passes, one skip). The exact skipped test separately passed on WSL with real symlinks. Raw Windows logs and JUnit are in the separate m4-native-qualification checkout under out/native-validation/windows-release-cuda/run-9davih7u.
- The M2 status reports frame 2 seed volume 1.328 m³ against 1.768 m³ represented liquid. The 6.25 cm projection cap cannot reach 0.5% agreement. Shared-interface end-to-end work needs a phase-aware surface path and a qualified source scaffold. M3 qualification reports open integration for adaptive family remap, exterior pressure, face ownership and the benchmark. While the baseline was pinned, local origin/main advanced to b540f2d with the band handoff. Recheck the adaptive integration ancestry and receipts before adopting a newer base.
- The next independent integration step is a coupled pressure/velocity solve using the full-cell rows and actual pressure-link metric. Use a method valid for inexact inner solves and check both true residual blocks. Wall and cut geometry, adaptive transition strain, EOS predictor/history, cavity persistence, constituent remapping, shared geometry and Blender acceptance still need implementation and end-to-end evidence.
- The velocity-block review found no concrete defect. Added a second independently assembled fixture with a four-term shear row, a normal row and unequal positive masses. All 35 native viscosity cases and clang-format passed. Full production use still needs measured Krylov range, scaling and residual replacement coverage.
- EOS drift RED: the sensor regression failed on the missing volume_ratio_sensor. GREEN: 38 focused EOS, pressure and resampling cases and 352 locked reference cases passed. Exact binary64 reference-volume quadrature is independent of face interpolation. The Tait fixture retains true compression, leaves the core at zero and clamps drift. The sealed exact-incompressible fixture balances only sensor cells.
- The complete WSL Python suite at 6b52702 passed all 1328 tests, including the symlink provenance case. Reproduce from a git archive of the commit with PYTHONPATH=reference/src and PYTEST_DISABLE_PLUGIN_AUTOLOAD=1, using python -m pytest reference/tests tools/tests tools/native/tests -q -o addopts= -p no:cacheprovider. Log: out/m4-linux-python-6b52702.log.

## Coupled CPU pressure receipt

The CPU coupling consumes the frozen pressure links, preserves their signed jump and exterior pressure conventions, and uses the pressure face inertia. The reduced velocity block receives the prescribed-wall strain lift. Restarted GCR uses measured responses from the inexact velocity inverse and accepts only a recomputed integrated residual, volume-scaled physical defect on every cell, and checked momentum solve. The pressure preconditioner uses the viscous velocity-block diagonal.

The manufactured compliant-pressure case, moving normal walls with an oriented internal jump, and twelve full-cell thin-slab extension cases pass. Extension covers 16² and 32² cells, both declared timesteps and ratios 1, 10 and 1000. It measures the normal RMS per side and uses same-phase shear samples on both sides. The worst normalized normal RMS, normal maximum and tangential maximum were 4.36e-5, 5.97e-5 and 8.22e-4 respectively. The inviscid preconditioner initially exhausted its budget at ratio 1000. With the viscous diagonal, enlarging the restart window from 12 to 32 cut the Debug extension campaign from about 41 s to 9.75 s. The final 130-test pressure/viscosity filter took 19.4 s. Raw receipt: out/m4-coupled-native.log. Reproduce with the locked Windows Debug CPU build and fluidity_native_tests.exe --gtest_filter=*Pressure*:*Viscosity*. clang-format 19.1.5 passed. Review requested per-side RMS and one-sided shear, both applied and verified.

Next, represent prescribed tangential wall strain in the sparse rows, then exercise both Couette tractions on CPU. Scene integration needs a validated material viscosity field, compatible encoding for existing scenes, stage routing, and a CUDA operator path. Cut and adaptive quadratures, end-to-end acceptance and the remaining M4 feature gates are open.

## Tangential wall strain and Couette strip receipt

Sparse strain rows now carry a checked additive boundary strain. Half-cell shear uses the signed wall-to-face derivative and half the interior shear volume, with an optional prescribed tangential derivative of the wall-normal velocity. The coupled solve lifts that affine strain into active momentum. The positive velocity block continues to apply the linear part.

The first finite-strip run failed because the inlet and outlet carried the exact normal velocity traces but omitted the corresponding shear in transverse momentum. Adding the analytic shear at those boundaries recovered the manufactured solution. At resolutions 16 and 32, timesteps 0.005 and 0.0025 s, and viscosity ratios 1, 10 and 1000, the worst normalized normal and shear maxima were 1.20e-6 and 8.66e-7. The tests measure both tractions, including the plate rows and the side rows. This strip uses analytic inlet and outlet traces with matching shear, so the declared periodic-link Couette fixture remains open. Raw results are in out/m4-couette-native.log and out/m4-couette-pressure-viscosity.log. The final pressure/viscosity filter passed 131 cases in 19.5 s. clang-format 19.1.5 passed.

To exercise the periodic Couette configuration, route explicit pressure-link graphs through the same coupled solver and build wraparound strain rows without duplicate velocity owners. That graph interface will also be needed for adaptive links. Scene routing and real-device qualification remain open.

## Periodic CPU Couette receipt

The explicit pressure-link graph now enters the same coupled CPU solve as the uniform pressure links. The fixture uses a unit-square, unit-depth extrusion with periodic x ownership, prescribed plate velocities, and independent wrapped tensor rows. It measures normal and same-phase tangential traction on each side of the interface. All twelve cases passed at 16² and 32², dt 0.005 and 0.0025 s, and ratios 1, 10 and 1000. Worst normalized normal RMS and maximum were zero. Worst tangential RMS was 2.59e-12 and maximum was 5.24e-11. The first square fixture used an incorrect two-column slot stride, which the graph ownership validation rejected. That negative run is retained in out/m4-periodic-stride-negative.log.

The focused periodic fixture ran in 93 ms in Windows Debug. The final pressure and viscosity filter passed all 132 cases in 20.8 s, and clang-format 19.1.5 passed. Receipts: out/m4-periodic-couette-focused.log and out/m4-periodic-couette-native.log. Reproduce with the locked Windows Debug CPU build and fluidity_native_tests.exe --gtest_filter=*Pressure*:*Viscosity*. The next slice is accepted scene routing for the CPU coupled stage, then its CUDA counterpart, wall/cut integration, and whole-scene qualification.

## Resolved CPU scene routing

The optional Pa s field now round-trips through FLRS and participates in the scene fingerprint. Missing and zero values retain the historical canonical bytes. A positive value enters the coupled projection in the normal CPU scheduler and publishes through the existing trial-state boundary. The accepted trace reports the coupled momentum residual. CPU scene support currently covers full cells and open or sealed full faces with natural tangential slip at static walls. Cut quadratures, moving no-slip walls, a translating piston, and backends without coupled viscosity support require further implementation. An unsupported cut attempt preserved the exact committed state bytes, and an unavailable backend rejected the request before stepping.

The aligned control initially revealed summation roundoff in full-cell clipping. The adapter now uses the geometry implementation's 64-epsilon bound. The real CPU Simulate process compared two separately resolved packets with the same shear seed. In the verified published frame 1, the first particle's absolute z velocity was 9.975234865692224e-5 m/s for the control and 9.825845698540285e-5 m/s with 100 Pa s. Across five accepted viscous substeps, the maximum recorded momentum residual was 3.56e-15 kg m/s and the maximum constraint defect was 5.52e-7 s⁻¹. Raw packets, protocol logs, point reads and caches are in out/m4-viscous-scene-cfe36227. Reproduce via ctest --preset windows-debug-cpu -R '^SimulateCli.cpu_execution$' --output-on-failure with the locked environment.

The final affected native filter passed 137 tests, including historical independent FlatBuffer bytes, viscosity damping and rollback. All 13 CPU CLI process tests and all 16 schema tests passed. clang-format 19.1.5 passed. Receipts: out/m4-scene-viscosity-affected.log and out/m4-scene-viscosity-process.log. Review found no confirmed routing defect and flagged whole-job allocation and the no-slip scene path as open. The next solver slice is a device velocity-block inverse routed through the same coupled pressure orchestration, with real-device parity and accepted-scene receipts.

## CUDA viscous velocity-block receipt

The checked host preparation supplies the same square-root strain weights and Jacobi diagonal as the CPU path. CUDA device 0 executes the complete inner PCG recurrence, including the true momentum residual check, with FP64 arithmetic and deterministic slot and row order. The first device implementation uses one thread. Its results establish operator parity on the bounded fixture. Coupled pressure integration, device occupancy and large-scene latency still require work.

CUDA 13.4 built the Release SM120 image and ran both new viscosity tests on an RTX 5060 Ti. The parity fixture independently applies the assembled three-by-three momentum matrix and checks the eliminated slot. The failure fixture covers an exhausted iteration budget, eliminated momentum rejection and zero RHS. The first build exposed the NVCC host-pass requirement to guard the architecture marker, recorded in out/m4-cuda-viscosity-arch-negative.log. The corrected build, CTest and memcheck logs are out/m4-cuda-viscosity-build.log, out/m4-cuda-viscosity-tests.log and out/m4-cuda-viscosity-memcheck.log. Memcheck reported zero errors. All 44 tests in the CUDA operator executable passed with the architecture marker in out/m4-cuda-operator-tests.xml. clang-format 19.1.5 passed across tracked and new native sources.

Reproduce in the locked native environment with the qualified Windows Release CUDA preset. Build fluidity_cuda_operator_tests, run ctest --preset windows-release-cuda -R '^CudaMacViscosity3D\.' --output-on-failure, then run compute-sanitizer --tool memcheck --error-exitcode 99 out/build/windows-release-cuda/fluidity_cuda_operator_tests.exe --gtest_filter=CudaMacViscosity3D.*.

## Coupled CUDA pressure operator receipt

The uniform coupled pressure operator now accepts a checked velocity inverse. The CUDA entry point runs the pressure iteration on the host and sends each velocity inverse to device 0. Its analytic one-cell fixture checks 4 Pa pressure, 1 and 3 m/s open-face velocities, the compliant flux balance, and the accepted momentum residual. All face velocities and the pressure iteration count agree with the CPU path. The shared CPU explicit-graph entry point retains its CPU velocity inverse.

All three Release CUDA viscosity cases passed on the RTX 5060 Ti, including the coupled fixture. Compute Sanitizer reported zero errors. The seven focused Debug CPU coupled and velocity-block tests passed, including the CPU periodic and high-contrast traction cases. Reproduce with the locked environment using ctest --preset windows-release-cuda -R '^CudaMacViscosity3D\.' --output-on-failure and ctest --preset windows-debug-cpu -R 'CoupledViscosity|VelocityBlock' --output-on-failure. Test and sanitizer output is in out/m4-coupled-cuda-tests.log, out/m4-coupled-cuda-memcheck.log and out/m4-coupled-cpu-tests.log. clang-format 19.1.5 passed across tracked native sources.

The full real-device CUDA operator executable passed all 45 cases. Its log and XML are out/m4-coupled-cuda-operators.log and out/m4-coupled-cuda-operators.xml. Review verified dispatch and transactional error propagation. The CUDA entry point uses the standalone allocator for each inner solve. The scheduler still needs integration with persistent CUDA workspace and end-to-end qualification.

## Shared scheduler adapter for arena integration

The full-cell scheduler geometry checks and strain assembly now accept a projection callable. The coupled pressure solve also accepts a borrowed callable so the CUDA scheduler can bind the active arena without an owning wrapper. The standalone CUDA operator remains independently callable. The scheduler CUDA callback still requires arena dispatch because the spec forbids device allocation inside a physical substep.

The seven focused CPU tests and three real-device CUDA viscosity tests passed. Reproduce with ctest --preset windows-debug-cpu -R 'SchedulerConsumesResolvedViscosity|UnsupportedViscousGeometry|CoupledViscosity' --output-on-failure and ctest --preset windows-release-cuda -R '^CudaMacViscosity3D\.' --output-on-failure. Logs are out/m4-arena-adapter-windows-debug-cpu-tests.log and out/m4-arena-adapter-windows-release-cuda-tests.log. The incremental CUDA build initially linked an object with the old callback ABI. Recompiling viscosity_velocity_block_3d_host.cc recovered the build. The initial linker receipt is out/m4-arena-adapter-stale-object.log. clang-format 19.1.5 passed.

## Arena-backed CUDA full-cell scene receipt

The CUDA scheduler now runs full-cell viscous projections on its arena stream. It reuses mutually exclusive pressure ranges for the strain data, velocity scratch and termination report. The bounds reserve up to six strain rows and eighteen strain terms per pressure cell and five FP64 vectors per incident face. The accepted substep matched CPU state and held device allocation and free counts constant. Host preparation and outer GCR still use host vectors, the inner kernel still uses one CUDA thread, and launch-level runtime fault injection remains open.

All 46 CUDA operator tests passed on the RTX 5060 Ti. Compute Sanitizer reported zero errors across all four focused viscosity cases. Both CPU and CUDA Simulate execution tests passed using equivalent resolved viscous and inviscid aligned-tank seeds. The CUDA process published frame 1 at SM120 after five accepted attempts and zero retries or capacity replays. At 100 Pa s the first particle's absolute z speed was 9.825845698540285e-5 m/s, versus 9.975234865692224e-5 m/s in the control. The maximum accepted viscous momentum residual was 3.5527136788005e-15 kg m/s. The raw resolved packets, cache and protocol streams are in out/m4-viscous-cuda-61f5d6cf. The earlier attempt using CPU-resolved packets was rejected by executable/backend identity as expected.

Reproduce in the locked environment with ctest --preset windows-release-cuda -R 'FullCellViscosityMatchesCpu|CudaMacViscosity3D|^SimulateCli.cuda_execution$' --output-on-failure and ctest --preset windows-debug-cpu -R '^SimulateCli.cpu_execution$' --output-on-failure. Receipts are out/m4-arena-viscosity-tests.log, out/m4-arena-viscosity-operators.xml, out/m4-arena-viscosity-memcheck.log, out/m4-arena-viscosity-cuda-cli.log and out/m4-arena-viscosity-cpu-cli.log. Review checked arena bounds and stream lifetimes. clang-format 19.1.5 passed.

## Native liquid EOS receipt

The native evaluator computes Hencky or exponent-seven Tait J, log J, density, tangent modulus and sound speed at gauge pressure. It preserves the exact-infinite branch and checks the Tait tensile domain. Review identified loss from reconstructing a subnormal reference modulus via the Tait pressure scale. A fused tangent and a regression now preserve the representable reference state.

The Windows Debug CPU filter passed 13 EOS and compliance cases. The independent Python EOS suite passed all 15. Reproduce in the locked native environment with ctest --preset windows-debug-cpu -R 'LiquidEos3D|Compliance' --output-on-failure and pytest reference/tests/test_eos.py. Receipts are out/m4-liquid-eos-tests.log and out/m4-liquid-eos-reference.log. clang-format 19.1.5 passed. Scene material selection, pressure predictor advection, drift sensor, acoustic oracles, and resolved-process verification remain open.

## Resolved EOS compliance receipt

Resolved scene materials can now select finite reference sound speed and Hencky or Tait. Missing settings retain the exact legacy packet, verified against the independent canonical FlatBuffer golden. An explicit EOS requires the legacy modulus to be infinite. The scheduler evaluates the tangent from the matching cell's carried pressure, freezes it for the projection on either backend, and checks EOS domain before accepting the projected pressure. The two-frame CPU fixture observed nonzero carried pressure and verified K0 + 7p. An injected inadmissible pressure failed in the pressure stage and preserved exact committed state bytes.

The focused CPU and CLI filter passed 40 tests. All 16 schema tests passed. The CUDA scheduler, viscosity and CLI filter passed 16 cases. The RTX 5060 Ti Tait fixture matched CPU pressure, particle position and particle velocity at both frames, with carrier velocity error at most 8.33e-17 m/s. Compute Sanitizer reported zero errors for that case. Logs are out/m4-resolved-eos-cpu-tests.log, out/m4-resolved-eos-schema.log, out/m4-resolved-eos-cuda-tests.log and out/m4-resolved-eos-cuda-memcheck.log. clang-format 19.1.5 passed across tracked native sources.

Reproduce with the locked Windows Debug CPU preset and ctest -R 'LiquidCache3D|LiquidEos3D|LiquidSubstep3D.Tait|SimulateCli.cpu_execution', and the locked Windows Release CUDA preset with ctest -R 'CudaLiquidSubstep3D|CudaMacViscosity3D|SimulateCli.cuda_execution'. Advected full-grid history, pressure-dependent mobility, drift sensors, checkpoint continuation and acoustic-scene gates remain open.

## Committed pressure-history advection receipt

Finite EOS pressure assembly now backtraces every frozen liquid cell center in the committed MAC carrier field, including cells without particle samples. It averages two eligible opposite face speeds, uses the available face if only one is eligible, and uses zero for a missing component. It interpolates committed liquid-cell pressure with renormalized trilinear weights. Missing donors fall back to old local pressure or ambient gauge pressure for a newly wet cell. The advected pressure supplies both the frozen tangent and the compliance history term. This is first-order uniform-grid backtracing. It uses committed liquid occupancy to mask donors and has no swept-wall or phase-aware tracing.

The independent signed-block fixture recovered 7 Pa from a 0.25-cell departure between 4 and 8 Pa, rejected a solid donor, and checked ambient fallback. The two-frame Tait scheduler test verified the frozen tangent and exact serialized continuation after encoding and decoding the first accepted frame. Seven focused Debug CPU cases and five real-device Release CUDA cases passed. Compute Sanitizer reported zero errors for the Tait scheduler. Receipts are out/m4-pressure-advection-windows-debug-cpu-tests.log, out/m4-pressure-advection-windows-release-cuda-tests.log and out/m4-pressure-advection-memcheck.log. Reproduce with the locked Windows presets and ctest filters 'LiquidSubstep3D\.(PressureHistory|Tait)|LiquidEos3D' and 'CudaLiquidSubstep3D\.(Tait|FullCellViscosity)|CudaMacViscosity3D'. clang-format 19.1.5 passed. EOS-dependent mobility, particle drift correction and the acoustic-scene gates remain open.

## EOS pressure mobility receipt

Finite EOS scenes now freeze density from the advected predictor. An open internal face uses the reciprocal of the arithmetic mean of its two liquid densities, the series mobility for equal half-cell distances. Free-surface links use only the adjacent liquid density. The same link coefficient enters pressure assembly, correction and the coupled viscous face inertia. Legacy scenes preserve reference density. The active capillary bound tightens with the minimum predictor density, capped at reference density, and an oversized attempt retries transactionally.

An independent two-cell test checked 0.25 m³/kg for a shared face with densities 2 and 6 kg/m³, the adjacent-cell free-surface mobilities, signed velocity correction and invalid-density rejection. The two-frame Tait scheduler test also checks predicted density. The expanded Hencky scene rejected an oversized capillary attempt and preserved exact committed bytes. All 32 focused CPU and 21 real-device CUDA cases passed. Compute Sanitizer reported zero errors for the Tait scheduler, whose maximum CPU/CUDA pressure difference was zero. Logs are out/m4-eos-mobility-windows-debug-cpu-tests.log, out/m4-eos-mobility-windows-release-cuda-tests.log and out/m4-eos-mobility-memcheck.log. clang-format 19.1.5 passed.

Reproduce under the locked Windows presets with ctest -R 'LiquidBoundaryGeometry3D|LiquidSubstep3D\.(PressureHistory|Tait|ExpandedEos)|LiquidEos3D' on CPU and ctest -R 'CudaLiquidBoundary3D|CudaLiquidSubstep3D|CudaMacViscosity3D' on CUDA. Particle drift sensors and acoustic qualification remain open.

## Native cell sensor and drift-rate receipt

The native host sensor scatters cell-centered trilinear tent weights to selected cells, uses same-side particle reference volume and retains an explicit no-donor state. The scalar drift rate compares sensed log J with the EOS target, clamps the correction in physical units and returns zero in unsensed cells. Review exposed a loss of a significant light donor when a dominant weight rounded its blend to one. The implementation now interpolates with the smaller weight ratio and checks both particle orders.

Analytic tests covered unequal reference volumes, signed lattice neighbors, wrong-side exclusion, unsensed cells, two grid spacings, two correction times, the clamp and an EOS-matched compressed state. All 11 focused Windows CPU tests and all 15 Python EOS reference tests passed. Reproduce with the locked environment using ctest --preset windows-debug-cpu -R 'LiquidEos3D|LiquidSubstep3D\.(PressureHistory|Tait|ExpandedEos)' --output-on-failure and pytest reference/tests/test_eos.py. Receipts are out/m4-eos-sensor-tests.log and out/m4-eos-sensor-reference.log. clang-format 19.1.5 passed. The next integration needs resolved correction controls, scheduler scratch and sealed-component balancing of sensor sources separately from physical sources.

## Finite-EOS scene drift receipt

Resolved finite-EOS materials can select a correction time and maximum rate. The optional fields preserve historical canonical bytes. Active drift uses pre-reserved host scratch that grows with the scheduler, samples same-side reference volume, and adds the correction to the physical source before CPU or CUDA pressure assembly. Unsensed cells retain the physical source and zero-rate controls skip sampling. Review identified that nonzero-side particles would otherwise be silently excluded. Active drift now requires side-zero initial particles and emission templates until phase-aware cell ownership is available.

The CPU scheduler test observed the clamped correction at the one sensed cell and the exact physical source in every unsensed cell. The two-frame Tait test used nonzero drift on the RTX 5060 Ti and matched CPU pressure and particle state. All 21 focused CPU tests, 16 schema tests and two focused real-device CUDA tests passed. Compute Sanitizer reported zero errors. Reproduce with the locked Windows presets and ctest filters 'LiquidCache3D.*Resolved|LiquidEos3D|LiquidSubstep3D\.(FiniteEos|Tait)' and 'CudaLiquidSubstep3D\.(Tait|FullCellViscosity)'. Receipts are out/m4-eos-drift-windows-debug-cpu-tests.log, out/m4-eos-drift-windows-release-cuda-tests.log, out/m4-eos-drift-schema.log and out/m4-eos-drift-memcheck.log. clang-format 19.1.5 passed. Incompressible drift and sealed-component source balancing remain open.

## Sealed band drift balance receipt

The native balance routine takes the sensed cells of one sealed component and finds a common shift before clamping. It rebases around the volume-weighted median and checks the normalized residual of the rounded output rates. Unsensed core cells and physical sources are outside the inputs. The reviewed correction solves in physical rate units to preserve raw-rate differences much smaller than the clamp. Unrepresentable normalized balance terms cause a checked failure.

The native tests cover unequal volumes and saturation, offsets smaller than one raw-rate ulp, rates 600 orders of magnitude below the clamp, and rejected balance-term underflow. All 12 focused Windows CPU cases and 15 Python EOS reference tests passed. Reproduce in the locked environment with ctest --preset windows-debug-cpu -R 'LiquidEos3D|LiquidSubstep3D.FiniteEos' --output-on-failure and pytest reference/tests/test_eos.py. Logs are out/m4-sealed-drift-tests.log and out/m4-sealed-drift-reference.log. clang-format 19.1.5 passed. Scheduler component classification, separate physical compatibility enforcement and end-to-end incompressible drift remain open.

## Incompressible scheduler drift receipt

Incompressible side-zero scenes can now resolve drift controls. A bounded host pressure assembly classifies the physical-only graph and checks physical source/wall-flux compatibility before any sensor source is added. Open components use the local clamped correction, while each sealed component balances its sensed cells using the bounded common shift. Unsensed core cells keep the original source. Both backends share this preparation, using pre-reserved scratch and a separate candidate pressure-cell array.

The CPU scheduler test closed the pressure links of an aligned tank fixture. It observed corrections of -0.3 and +0.3 s⁻¹ at the two sensed cells and zero in the two core cells. An incompatible physical source was rejected before the projection callback and preserved exact serialized state. The open CUDA fixture matched CPU pressure and particle state on the RTX 5060 Ti. Compute Sanitizer reported zero errors. The historical FlatBuffer golden and all 26 focused CPU and seven CUDA cases passed. clang-format 19.1.5 passed. The first CUDA incremental link used a stale Staging::Create signature. Recompilation of the CUDA scheduler test recovered the build, and the original linker output remains in out/m4-incompressible-drift-windows-release-cuda-build.log.

Reproduce in the locked Windows presets with ctest -R 'LiquidSubstep3D\.(IncompressibleDrift|FiniteEos|Tait)|LiquidEos3D|MacPressureWorkspace3D|CanonicalResolvedBytes' on CPU and ctest -R 'CudaLiquidSubstep3D\.(OpenIncompressibleDrift|Tait|FullCellViscosity)|CudaMacPressure3D' on CUDA. Receipts are out/m4-incompressible-drift-windows-debug-cpu-tests.log, out/m4-incompressible-drift-windows-release-cuda-tests.log and out/m4-incompressible-drift-memcheck.log. Mixed open/sealed, growth with drift active, and acoustic-scene qualification remain open.

## Discrete acoustic standing-wave receipt

A fixed sealed column checks the independent cosine pressure and sine face-velocity eigenmode of the staggered incidence. The oracle uses the discrete frequency, backward-Euler damping and phase. It advances 16 steps at each of two spatial resolutions (8 and 16 cells) and two timesteps (0.01 and 0.025 s), feeding accepted pressure and velocity back into the next solve. The error bound accumulates the pressure-row defect in the contractive acoustic energy coordinates. A separate fixture gate limits pointwise waveform error to 0.05% of the 2 Pa initial amplitude. The original 4e-6 Pa pointwise limit was tighter than the accepted solver residual justified. Both negative runs are retained under out/m4-standing-wave-*-overstrict-negative.log.

The CPU and RTX 5060 Ti CUDA operators passed all 64 projections with 123 iterations on each backend. Maximum pressure error was 8.80e-4 Pa and maximum impedance-scaled velocity error was 8.76e-4 Pa. The receipts record approximately 4.08 ms CPU fixture wall time, 123 ms CUDA fixture wall time and 21.39 ms CUDA solver event time. Compute Sanitizer reported zero errors. All 18 CPU MacPressure3D and five CUDA MacPressure3D cases passed. clang-format 19.1.5 passed. Reproduce in the locked Windows presets with ctest -R BackwardEulerStandingWave. XML, regression and memcheck receipts use the out/m4-standing-wave prefix. Review checked the signs, damping, frequency and defect bound.

This oracle qualifies the fixed-coefficient discrete pressure operator. Full scheduler acoustic scenes, pressure-pulse, hydrostatic-compression, water-hammer and compliant-piston oracles remain open.

## Discrete acoustic pressure-pulse receipt

The sealed-column pressure oracle now initializes a single-cell 2 Pa pulse one quarter of the way along the column. It independently reconstructs the constant mean and all discrete cosine pressure and sine velocity modes, including the sealed-end reflection. Both backends pass the accumulated physical-defect bound for 16 steps at each combination of 8 or 16 cells and 0.01 or 0.025 s. The pulse fixture initially inherited the standing-wave 0.05% pointwise limit and failed it on both backends. The passing pulse uses a separate 0.1% engineering limit for this full-spectrum input. The physical-defect certificate and the standing-wave gate are unchanged. The failed runs are in out/m4-pressure-pulse-*-standing-wave-limit-negative.log.

Across 64 pulse projections, each backend used 126 pressure iterations. Maximum pressure error was 0.001311 Pa and maximum impedance-scaled velocity error was 0.001329 Pa. The fixture measured about 3.29 ms CPU wall time, 48.68 ms CUDA wall time and 21.72 ms CUDA solver event time. All 19 CPU and six CUDA pressure regressions passed, Compute Sanitizer reported zero errors, and clang-format 19.1.5 passed. Reproduce with the locked Windows presets and ctest -R 'BackwardEuler(StandingWave|PressurePulse)'. The XML, logs and memcheck output use the out/m4-pressure-pulse prefix.

Review verified modal normalization and the residual bound. The pulse is a fixed-grid discrete operator check. A one-cell pulse changes physical width with resolution, so the two cases do not establish continuum convergence. Full scheduler propagation, hydrostatic compression, water hammer and compliant piston remain open.

## Hydrostatic EOS scheduler receipt

The hydrostatic oracle integrates dp/d(depth) = rho0*g/J(p) independently for Hencky and exponent-seven Tait. It seeds pressure and particle compression in a one-metre open column and advances the normal scheduler for 0.02 s at 0.25 and 0.125 m cell widths. The initial one-cell-wide coarse tank was rejected in deposit because it had no transverse interior transfer face. The passing tank is 0.5 m wide at both resolutions. Negative receipts are out/m4-eos-hydrostatic-initial-cpu-negative.log, out/m4-eos-hydrostatic-initial-cuda-negative.log and out/m4-eos-hydrostatic-stage-negative.log.

Both backends used 18 total pressure iterations across the four cases. Maximum pressure error was 3.276031 Pa, maximum volume-ratio error was about 2.7e-5 and maximum particle speed was about 4.79e-4 m/s. Fixture wall time was about 2.15 s on Debug CPU and 1.07 s on Release CUDA. All 17 focused CPU and four CUDA checks passed, and Compute Sanitizer reported zero errors on the RTX 5060 Ti. clang-format 19.1.5 passed. Reproduce with the locked Windows presets and ctest -R EosHydrostaticCompression. XML, regression and memcheck receipts use the out/m4-eos-hydrostatic prefix.

The fixture limits scale with cell width and the expected one-sided surface pressure error. Review verified the continuum formulas and initial density. This tests short-time preservation of a seeded compressed equilibrium. It does not establish convergence from an uncompressed state or long-time hydrostatic stability.

## Compliant piston pressure-operator receipt

The sealed one-cell fixture supplies a prescribed wetted flux of 0.02 m³/s through one quarter of the moving wall and decreases the chamber volume analytically. It verifies positive pressure rise, no gauge, the scalar tangent recurrence and the physical defect for Hencky and Tait. It halves the timestep from 1/16 s through 1/64 s and checks convergence against the independently integrated EOS compression. Each refinement reduced the final relative error by more than 45%.

Both backends used 224 pressure iterations over 224 projections. The worst finest-resolution relative pressure error was 0.001352 and the worst constraint defect was 1.738402e-9 s⁻¹. The fixture measured about 4.72 ms CPU wall time, 230 ms CUDA wall time and 41.31 ms CUDA solver event time. All 20 CPU and seven real-device CUDA pressure regressions passed. Compute Sanitizer reported zero errors and clang-format 19.1.5 passed. Reproduce with the locked Windows presets and ctest -R CompliantPistonConverges. XML, regression and sanitizer receipts use the out/m4-compliant-piston prefix.

The fixture qualifies the pressure operator and EOS with prescribed moving-wall flux. It does not measure wetted geometry, swept volume or closed-chamber scheduler behavior.

## Discrete water-hammer receipt

The one-metre column starts with uniform positive flow on the interior faces and both ends close instantaneously. The independent oracle projects that initial velocity onto the sealed-column sine modes and evolves each coupled pressure/velocity mode with the backward-Euler phase and damping. The fixture checks the full discrete waveform and accumulated physical-defect bound. It also checks that the mean pressure in the last eighth of the column at 0.25 s approaches the 2 Pa Joukowsky plateau on 16, 32 and 64 cells at fixed Courant number.

The finest plateau relative error was about 4e-6 on both backends. Maximum discrete pressure and impedance-scaled velocity errors were 4.14e-4 Pa and 5.30e-4 Pa. Each backend used 450 pressure iterations across 224 projections. The fixture measured about 52.8 ms CPU wall time, 330 ms CUDA wall time and 195.9 ms CUDA solver event time. All 21 CPU and eight real-device CUDA pressure regressions passed. Compute Sanitizer reported zero errors and clang-format 19.1.5 passed. Reproduce with the locked Windows presets and ctest -R WaterHammerValveClosure. The XML, regression and sanitizer receipts use the out/m4-water-hammer prefix.

Review confirmed the mode normalization, closure sign and plateau location. This is a linear, instantaneous closure oracle for the fixed-coefficient pressure operator. Reservoir boundaries, finite valve motion, cavitation and scheduler-driven closure remain outside its evidence.

## Native trapped-air constitutive receipt

The native fixed-inventory evaluator implements the exponent-1.4 polytropic law, a single gauge boundary lift with signed liquid curvature, the radial timestep limit and the equivalent-radius resolution cutoff. The submerged-sphere initializer includes atmospheric, hydrostatic and Laplace pressure in the absolute inventory. Analytic tests checked equilibrium cancellation, compression and expansion, the resolution threshold, near-equilibrium gauge precision and exponent-scaled timestep arithmetic.

All 14 focused native cavity and EOS checks and six Python cavity reference checks passed. The Release CUDA build and three focused real-device EOS/pressure regressions passed. clang-format 19.1.5 passed. Reproduce under the locked environment with ctest --preset windows-debug-cpu -R 'LiquidCavity3D|LiquidEos3D' and pytest reference/tests/test_cavity.py. Receipts use the out/m4-native-cavity prefix. Review found no numerical defect in the checked constitutive slice.

Resolved cavity discovery, geometric volume, persistent identity and lineage, boundary lift routing, scheduler timestep enforcement, disappearance receipts and Q-CAVITY qualification remain open.

## Cavity gas-pressure boundary receipt

Pressure link assembly now accepts optional gas gauge pressures keyed by sampled air cell. Open air faces use the selected air-side pressure in either orientation and add the existing signed liquid-side capillary lift once. Unlisted air neighbors retain ambient gas pressure. The cavity evaluator exposes its gas gauge pressure before capillarity, including the near-equilibrium correction. Checked staging rejects malformed or liquid-side records, overlarge spans and aliases with unpublished pressure or CUDA scratch.

The shared CPU and CUDA fixture supplied 19 and -3 Pa gas pressure with a -1 Pa capillary jump, observed Dirichlet pressures 18 and -4 Pa, and retained 6 Pa on the four ambient faces. The one-cell solve recovered 38/6 Pa and passed the physical constraint gate. Duplicate and liquid-center overrides were rejected. All 28 focused CPU and eight real-device CUDA cases passed. Compute Sanitizer reported zero errors and clang-format 19.1.5 passed. Reproduce under the locked presets with ctest -R 'LiquidBoundaryGeometry3D|LiquidCavity3D' on CPU and ctest -R 'CudaLiquidBoundary3D|CudaLiquidSubstep3D\.(EosHydrostatic|Tait)' on CUDA. Logs use the out/m4-cavity-link prefix.

Review checked orientation, extent, aliasing and the single capillary lift. The caller still supplies the cavity mapping. Discovery, clipped volume, persistent lineage, timestep routing and Q-CAVITY remain open.

## Air-component labeling receipt

The bounded native labeler traverses certified face passages between sorted signed-lattice gas cells. It reports deterministic component membership and summed input gas volume. A missing open neighbor is unclassified unless an atmosphere outlet is certified. A component with a certified outlet is vented even if another part of its frontier is unclassified. Closed supplied boundaries yield an enclosed component. The caller provides disjoint label, queue and component scratch.

The CPU fixture checked mixed enclosed, vented and unclassified components, partial gas volumes, asymmetric passages, scratch overlap and signed endpoints. All six focused CPU cases and the existing real-device CUDA boundary regression passed. Reproduce with the locked Windows presets using ctest -R 'LiquidCavity3D|CavityGasPressure' on CPU and ctest -R '^CudaLiquidBoundary3D.GeometryAndCapillaryMatchOracle$' on CUDA. Logs use the out/m4-air-components prefix. clang-format 19.1.5 passed and review found no graph-labeling defect.

The labeler consumes supplied gas geometry. Clipping the gas phase and certifying face passages from the sampled field, persistent cavity lineage, scheduler publication and Q-CAVITY remain open.

## Gas cut-cell measures receipt

The native gas measurement reuses the liquid geometry tetrahedra and shared face diagonals to clip the strictly positive-distance phase against the frozen obstacle. It returns geometric gas volume and six gas apertures. Tank walls retain their closed-face convention, and gas on both sides of a slab in one cell is rejected. An all-zero interface plateau contributes no volume or face area.

Analytic fixtures checked 0.75 m³ gas in a planar cut, equal areas on both sides of a shared face, a simultaneous halfspace cut and an open tank with closed walls. All 32 CPU cavity and boundary geometry checks and the real-device CUDA geometry regression passed. The gas measure itself currently runs on the host. Reproduce with the locked Windows presets and ctest -R 'LiquidBoundaryGeometry3D|LiquidCavity3D' on CPU and ctest -R CudaLiquidBoundary3D.GeometryAndCapillaryMatchOracle on CUDA. Logs use the out/m4-gas-measures prefix. The initial declaration-order build failure is retained in the declaration-negative logs. clang-format 19.1.5 passed.

Positive gas volume and face aperture alone do not certify one connected gas region inside a cut cell. The geometry-to-component path still needs that certification and exterior classification before it can supply persistent cavity inventories to the scheduler.

## Gas cut-cell connectivity receipt

The host gas measurement now checks the connectedness of positive-volume clipped tetrahedra before returning one gas node for a cell. It uses the same six tetrahedra as liquid geometry, with edges only across positive-area gas patches on a shared triangle. Clipping honors the frozen tank or the applicable solid side. Disconnected positive pieces return FailedPrecondition without publishing measures.

The analytic fixture rejected opposite gas corners and zero-area edge contact, then accepted a positive-area bridge. All 33 CPU boundary and cavity cases and all six real-device CUDA boundary regressions passed. The new connectivity certificate runs on the host. Reproduce under the locked presets with ctest -R 'LiquidBoundaryGeometry3D|LiquidCavity3D' on CPU and ctest -R CudaLiquidBoundary3D on CUDA. Receipts use the out/m4-gas-connectivity prefix. The initial implicit corner-bit conversion failed the warning-as-error build and is preserved in out/m4-gas-connectivity-cpu-conversion-negative.log. clang-format 19.1.5 passed.

The next dependency is constructing a bounded sampled gas domain, certifying its atmosphere outlets and connecting the cell measures to component labeling. Persistent inventories and scheduler qualification remain open.

## Sampled gas domain receipt

The bounded host adapter converts ordered corner samples and a frozen obstacle into positive-volume gas graph nodes. Certified in-cell connectivity is required before a node is emitted. Open gas apertures lead to represented neighbors or unclassified frontiers. An explicit atmosphere outlet must have positive gas aperture and exactly one sampled incident cell. Missing samples and inconsistent outlet declarations fail rather than fabricating trapped gas.

An independent eight-cell fixture used a single positive lattice corner. Twenty-four equal tetrahedral corner cuts produced 0.5 m³ in one enclosed component. Cropping the last cell left 0.375 m³ and an unclassified frontier. Certifying one of its open outer faces changed the classification to vented, while the same declaration on the intact interior face was rejected. All 34 CPU cavity and boundary geometry cases and the real-device CUDA boundary regression passed. Reproduce with the locked Windows presets and ctest -R 'LiquidBoundaryGeometry3D|LiquidCavity3D' on CPU and ctest -R CudaLiquidBoundary3D.GeometryAndCapillaryMatchOracle on CUDA. Receipts use the out/m4-sampled-air-domain prefix. clang-format 19.1.5 passed.

The scheduler must still supply a sampled domain with certified atmosphere outlets, persist inventories across accepted topology changes, and feed resulting pressure and timestep bounds into the normal path.

## Cavity pressure and timestep preparation receipt

The host adapter consumes labeled gas cells and component-aligned inventories. Resolved enclosed components with inventories supply gauge pressure at matching air-side pressure centers and contribute to the minimum radial timestep bound. The boundary link adds the signed capillary jump once. The adapter reports subresolution inventories and rejects malformed labels.

The CPU fixture checked polytropic pressure and the radial bound against independent calculations, excluded liquid centers and inactive components, and rejected short output and orphaned labels. The locked Windows presets passed eight CPU checks with ctest -R 'LiquidCavity3D|CavityGasPressure' and one real-device CUDA regression with ctest -R CudaLiquidBoundary3D.GeometryAndCapillaryMatchOracle. Receipts use the out/m4-cavity-boundaries prefix. clang-format 19.1.5 passed. Independent review checked component eligibility, ordered output, capillary routing and aliasing.

The adapter accepts caller-aligned inventories. Persistent matching across accepted steps, scheduler inputs and publication, and Q-CAVITY remain open.

## Polytropic topology transfer receipt

Inventory transfer apportions the additive quantity V_initial p_initial^(1/1.4). It combines shares in caller order and represents the result using the largest contributing reference pressure. This is the chosen equal-pressure polytropic mixing closure. It preserves the original inventory on a split followed by reunion. Overlap assignment, accepted-step identity and loss accounting still belong to the caller.

The CPU fixture checked a 1:3 split, reunion, unequal reference pressures and invalid fractions against the pressure law. Nine CPU tests passed with ctest --preset windows-debug-cpu -R 'LiquidCavity3D|CavityGasPressure'. Six real-device boundary regressions passed with ctest --preset windows-release-cuda -R CudaLiquidBoundary3D. This transfer is host code. Build and test receipts use the out/m4-cavity-inventory-remap prefix, including the pre-implementation test build. clang-format 19.1.5 passed.

## Cavity overlap measurement receipt

Consecutive gas snapshots on the same MAC lattice now produce ordered component-pair weights. The measure sums the smaller gas volume in each coincident cell. This is a cell occupancy estimate and cannot resolve subcell motion or displacement with no coincident gas cells. Edges include any vented and unclassified recipients so downstream lineage can account for those transitions.

The CPU fixture checked a simultaneous split and merge, duplicate edge reduction, unmatched cells, invalid component counts and insufficient scratch. Ten focused CPU tests and six real-device CUDA boundary regressions passed with ctest --preset windows-debug-cpu -R 'LiquidCavity3D|CavityGasPressure' and ctest --preset windows-release-cuda -R CudaLiquidBoundary3D. The overlap routine runs on the host. Receipts use the out/m4-cavity-overlap prefix. clang-format 19.1.5 passed.

Inventory assignment, identity and loss accounting consume these weights in the next lineage step. Scheduler integration and Q-CAVITY remain open.

## Cavity candidate lineage receipt

Candidate lineage distributes each tracked parent's inventory in proportion to outgoing occupancy weights. The largest resolved enclosed child can keep the parent's ID. A merge keeps the strongest nominated ID, with the lower ID breaking ties. Other children and newly initialized cavities receive fresh IDs. The result accounts separately for vented, subresolution and unmatched inventory. An unknown recipient of tracked gas fails candidate preparation. New resolved cavities require an explicit initial inventory.

The CPU fixture exercises simultaneous split and merge, a new cavity, all three loss counters, an unknown frontier and missing birth initialization. Eleven CPU cases passed with ctest --preset windows-debug-cpu -R 'LiquidCavity3D|CavityGasPressure'. Six real-device boundary regressions passed with ctest --preset windows-release-cuda -R CudaLiquidBoundary3D. Candidate lineage runs on the host. Receipts use the out/m4-cavity-lineage prefix. clang-format 19.1.5 passed.

The scheduler still needs to own the accepted snapshots, provide initialization and exterior certificates, apply the candidate pressure and timestep bounds, and accumulate loss diagnostics on acceptance.

## Scheduler cavity pressure and retry receipt

The shared scheduler accepts an optional host provider evaluated on each attempt using step-start geometry samples. Its radial bound enters the existing transactional retry path. Its gas pressures enter the real CPU or CUDA projection links, where the capillary term is applied once. The accepted-step measurement records a finite radial bound.

Fourteen CPU tests passed with ctest --preset windows-debug-cpu -R 'Cavity|ExplicitCapillaryOversize'. The explicit attempt fixture verifies rollback, retry and publication at 17 Pa. Twenty-one real-device tests passed with ctest --preset windows-release-cuda -R 'CudaLiquidBoundary3D|CudaLiquidSubstep3D'. The frame fixture verifies retry, SM120, CPU parity and 17 Pa plus the analytic capillary jump. clang-format 19.1.5 passed. Receipts use out/m4-cavity-scheduler. The initial incremental CUDA build skipped translation units after the public header layout changed and failed in the scheduler tests. Rebuild with cmake --build --preset windows-release-cuda --clean-first --parallel 8 --target fluidity_cuda_operator_tests. The initial CUDA fixture omitted the capillary term in its expected pressure, recorded in the fixture-negative log.

The tests supply gas pressure directly. The production scheduler still needs to build the gas domain, maintain accepted lineage and initialize cavity inventory. Q-CAVITY remains open.

## Sampled cavity step preparation receipt

The host preparation API takes a sampled domain, a prior accepted snapshot and a birth initializer. It clips and labels gas, measures overlap, assigns candidate lineage, and prepares pressure and the radial limit in bounded scratch. Only new resolved enclosed cavities need initialization. Candidate state stays in caller scratch until the caller accepts the step.

An integrated spherical fixture initialized a resolved cavity at 104 kPa absolute, reduced the sampled radius from 3.2 to 3.0 m, and verified the same ID, a single initialization, the polytropic pressure, and a tighter radial limit. Fourteen CPU checks passed with ctest --preset windows-debug-cpu -R Cavity. Seven real-device regressions passed with ctest --preset windows-release-cuda -R 'CudaLiquidBoundary3D|PreparedCavityBoundaries'. The new preparation function runs on the host. clang-format 19.1.5 passed. Receipts use out/m4-cavity-step and include the initial fixture compile error for the distinct corner index type.

Production still needs to supply the gas sampling domain and atmosphere certificates, own accepted snapshots through retries and replays, and connect this preparation to the scheduler. Q-CAVITY remains open.

## Scheduler-owned cavity snapshot receipt

Opt-in cavity settings activate bounded scheduler-owned gas snapshots. The particle-derived domain fills its support bounding box to sample the gas-only interior. Trial snapshots publish with the accepted liquid step. Capacity replay restores the saved frame-start cavity state. The accepted measurement exposes active pressure counts and per-step losses. Gas scratch and three snapshot buffers are reserved up front from the larger configured capacity.

The particle-shell CPU fixture found one resolved cavity and injected a capacity failure after the first accepted step. Replay rebuilt the initial inventory once, then preserved it through subsequent accepted steps. Fifteen CPU tests passed with ctest --preset windows-debug-cpu -R 'Cavity|CapacityReplay'. Twenty-one CUDA scheduler and boundary regressions passed after a clean windows-release-cuda build with ctest --preset windows-release-cuda -R 'CudaLiquidBoundary3D|CudaLiquidSubstep3D'. The owned particle-shell fixture ran on CPU. clang-format 19.1.5 passed. Receipts use out/m4-owned-cavity, including initial fixture failures from an excessive face reservation and unsampled interior cells.

The opt-in API still depends on scene-provided atmosphere certification and birth initialization. Checkpoint serialization and production scene loading remain open, as does full Q-CAVITY qualification.

## Unobstructed exterior certification receipt

The bounded host gas domain can certify a positive gas aperture on an outer bounding-box plane when all liquid is contained inside the box and there are no obstacles. The scheduler selects this mode for its bounded unobstructed field path when there is no explicit outlet provider. Interior holes in a supplied sample set stay unclassified. Obstacle scenes continue to use explicit outlet certification.

The sampled sphere fixture opened a gas channel after compression. It verified vent classification, complete inventory release, removal of the gas-pressure override and an unbounded radial timestep. Fifteen CPU checks passed with ctest --preset windows-debug-cpu -R Cavity. Twenty-one real-device CUDA regressions passed with ctest --preset windows-release-cuda -R 'CudaLiquidBoundary3D|CudaLiquidSubstep3D'. Exterior certification runs on the host. clang-format 19.1.5 passed. Receipts use out/m4-cavity-exterior.

## Resolved cavity scene receipt

The resolved material wire has two optional cavity pressure fields. Together they enable scheduler-owned trapped air and give the absolute atmosphere and the hydrostatic gauge pressure for spherical birth equilibrium. The scheduler adds the Laplace term from scene surface tension and clipped equivalent radius. A decoded scene needs no host initialization callback. The hydrostatic value is scene-supplied and constant across births. The optional pair preserves the disabled path for old scenes.

Twenty-five CPU checks passed with ctest --preset windows-debug-cpu -R 'Cavity|LiquidCache3D.Resolved'. The fixture round-trips the pressure settings, checks the fingerprint, and advances a decoded particle shell with nonzero cavity pressure. A clean windows-release-cuda build produced fluidity_simulate and fluidity_cuda_operator_tests. Twenty-one real-device checks passed with ctest --preset windows-release-cuda -R 'CudaLiquidBoundary3D|CudaLiquidSubstep3D'. The executable schema exposed both fields. clang-format 19.1.5 passed. Receipts use out/m4-cavity-resolved.

The current scene settings do not serialize the accepted cavity snapshot. Resuming a cavity run through a disk checkpoint still needs state codec support. Full Q-CAVITY remains open.

## Cavity snapshot checkpoint codec receipt

Liquid state and its optional FlatBuffers extension now carry canonical gas cells, component labels, polytropic inventory and the next cavity ID. State validation checks label coverage and unique tracked IDs. Wire validation checks the same invariants before materialization. Transaction copies reserve cavity records only for enabled states and include them in equality and capacity checks.

The checkpoint fixture round-tripped a tracked and an untracked component, cloned the decoded state at resume capacity and rejected duplicate cavity IDs. Fifty CPU tests passed with ctest --preset windows-debug-cpu -R 'LiquidCache3D|LiquidState3D|Cavity'. Twenty-one real-device regressions passed with ctest --preset windows-release-cuda -R 'CudaLiquidBoundary3D|CudaLiquidSubstep3D'. Both presets were rebuilt cleanly for the state layout change. The CUDA build also produced fluidity_simulate. clang-format 19.1.5 passed. Receipts use out/m4-cavity-checkpoint.

The codec fixture supplies the cavity state directly. The scheduler's live owner still needs to read and publish the snapshot through LiquidState3D for process-resume continuity.

## Scheduler cavity checkpoint continuity receipt

The liquid transaction now owns the accepted cavity snapshot. The scheduler uses bounded working arrays to prepare the next snapshot and writes the candidate into the same trial state as particles, cells and clocks. Capacity replay and failed attempts use the transaction's existing rollback. Cavity-enabled resume requires a saved snapshot after any accepted step.

The decoded particle-shell scene ran to a frame boundary with nonzero cavity pressure, encoded and decoded a checkpoint, and continued to byte-identical state compared with uninterrupted execution. The cavity ID stayed unchanged. Fifty CPU state, codec and cavity checks passed with ctest --preset windows-debug-cpu -R 'LiquidCache3D|LiquidState3D|Cavity|CapacityReplay'. The full LiquidSubstep3D CPU suite passed all 85 checks. Twenty-one real-device regressions passed after a clean windows-release-cuda build with ctest --preset windows-release-cuda -R 'CudaLiquidBoundary3D|CudaLiquidSubstep3D'. The clean build also produced fluidity_simulate. clang-format 19.1.5 passed. Receipts use out/m4-cavity-resume.

The checkpoint equality fixture ran on CPU. The CUDA suite exercised the shared scheduler and the prepared-pressure route. Full Q-CAVITY qualification remains open.

## Real CUDA cavity checkpoint receipt

The shared particle-shell fixture exercises the scheduler's discovered gas domain and pressure lift on SM120. One resolved cavity imposed a finite radial bound and a nonzero pressure RHS. After a checkpoint round trip into a fresh CUDA backend, the cavity ID and full checkpoint digest matched uninterrupted CUDA execution.

Seventeen focused CPU tests passed with ctest --preset windows-debug-cpu -R Cavity. Four CPU surface occupancy tests passed with ctest --preset windows-debug-cpu -R '^SurfaceOccupancy3D\.', including the supplied-occupancy inner-shell removal and closed-cavity orientation cases. Twenty-two CUDA checks passed with ctest --preset windows-release-cuda -R 'CudaLiquidSubstep3D|CudaLiquidBoundary3D'. clang-format 19.1.5 passed. Receipts use out/m4-cavity-device. Compute Sanitizer reported zero errors with compute-sanitizer --tool memcheck --error-exitcode 99 out/build/windows-release-cuda/fluidity_cuda_operator_tests.exe --gtest_filter=CudaLiquidSubstep3D.DiscoveredCavitySurvivesCheckpointResume.

The occupancy fixtures use supplied classifications. Production cache-to-surface occupancy and full Q-CAVITY qualification remain open.

## Oriented surface boundary measurement receipt

Surface measurement now computes local-origin signed volume for each connected boundary. The deep-liquid fixture identifies the raw inward shell and verifies its removal without moving the exterior. The closed-cavity fixture retains one inward and one outward boundary. Production reconstruction preserves these counts when it combines separate material surfaces. Its separated-drop fixture measures two outward boundaries.

Twenty-one CPU mesh, occupancy, cache and reconstruction tests passed with ctest --preset windows-debug-cpu -R 'SurfaceMesh3D|SurfaceOccupancy3D|SurfaceReconstruction3D|SurfaceCache3D'. After adding the reconstruction aggregation, 15 focused CPU checks passed with ctest --preset windows-debug-cpu -R 'SurfaceReconstruction3D.SeedsSeparatedComponents|SurfaceOccupancy3D|SurfaceMesh3D|SurfaceCache3D'. The final build included fluidity_simulate. clang-format 19.1.5 passed. Receipts use out/m4-oriented-surface.

Motion frames already carry the cavity snapshot via accepted FLST states. Production reconstruction still needs to use that snapshot to admit and retain measured cavity boundaries. Q-CAVITY remains open.

## Cached cavity surface reconstruction receipt

Production reconstruction reads resolved cavity IDs from the motion frame. For a cached resolved cavity it samples the complete Simulate sign field through the nested interfaces. Extraction and volume correction require one outward boundary and one inward boundary per resolved cavity. Tracking refuses sampled changes to the cavity ID set.

The CPU fixture discovers a cavity in a quiescent particle shell, rebases that accepted state to frame zero, round-trips the motion wire, and reconstructs and encodes the surface. It retains both signed boundaries and matches the material volume within 0.5 percent. The production simulator built and 22 CPU tests passed with ctest --preset windows-debug-cpu -j 4 -R 'SurfaceReconstruction3D|SurfaceOccupancy3D|SurfaceMesh3D|SurfaceCache3D'. clang-format 19.1.5 passed. Receipts use out/m4-cached-cavity-surface.

The fixture rebases the snapshot because the normal frame-zero publication precedes the first cavity discovery. Initial scene discovery and a full deep-tank production qualification are still required for Q-CAVITY.

## Initial cavity snapshot receipt

A fresh cavity-enabled scheduler prepares gas and lineage at the initial obstacle pose before publishing frame zero. It uses the host field sampler and creates the transaction and backend from that initialized state. Restored checkpoints retain their saved snapshot. Capacity replay now restores the already discovered frame-zero cavity without reinitializing its gas.

The surface fixture now encodes the actual committed frame-zero motion seed and reconstructs the cavity without rebasing the clock. Eighteen CPU cavity tests passed with ctest --preset windows-debug-cpu -j 4 -R Cavity. Twenty-two real-device tests passed with ctest --preset windows-release-cuda -j 1 -R 'CudaLiquidSubstep3D|CudaLiquidBoundary3D'. The CUDA checkpoint fixture explicitly checks cavity state before the first step. A parallel CUDA run reported resource exhaustion in eight cases. All passed in the serial rerun. Both simulator targets built and clang-format 19.1.5 passed. Receipts use out/m4-initial-cavity.

The initial snapshot must fit the configured initial capacity before stepping. Full deep-tank production reconstruction and Q-CAVITY qualification remain open.

## Deep-tank surface geometry receipt

Production surface reconstruction accepts a stationary open tank with a planar fill. It intersects the full liquid sign field with the five tank walls, derives the material volume from the physical bounds and clamps carrier velocity into the admissible wall halfspaces. The particle band therefore supplies the carrier without turning the unsampled core into air.

An 8 m cube fixture had 296 particles on its boundary band and no particles in the 6 m interior cube. A frame-zero Simulate motion payload reconstructed one outward boundary and no inward shell, with measured volume within 0.5 percent of 512 m³. The separate closed-cavity fixture retained one inward boundary. These establish native production geometry evidence for both Q-CAVITY conditions at frame zero. A 1 m surface lattice failed volume correction after rounding the tank corners. A 0.5 m lattice passed. The coarse failure is preserved in the negative receipt.

The simulator built and 23 CPU checks passed with ctest --preset windows-debug-cpu -j 4 -R 'SurfaceReconstruction3D|SurfaceOccupancy3D|SurfaceMesh3D|SurfaceCache3D'. clang-format 19.1.5 passed. Receipts use out/m4-deep-tank-surface. This tank case is stationary and planar. Moving tank contact and complete M4 end-to-end qualification remain outside this receipt.

## Native phase-local mixture receipt

The bounded host mixture operator reduces current-volume contributions by stable substance ID and applies the additive-volume density law. Materials carry a phase ID. Cross-phase contributions fail before publishing fractions. Validation also covers sorted inputs, representability, output capacity and aliasing. Reduction preserves donor order within each substance.

The native fixture matches the independent aqueous oracle at 25 percent clear water, 75 percent dyed water, and 1015 kg/m³. Oil stays in a separate phase. Invalid inputs leave fraction storage unchanged. Eighteen CPU mixture, remap and EOS checks passed with ctest --preset windows-debug-cpu -R 'LiquidMixture3D|LiquidRemap|LiquidEos3D'. All five Python multiphase reference checks passed with python -m pytest reference/tests/test_multiphase.py -q -p no:cacheprovider. Three existing real-device CUDA regressions passed for Tait, viscosity and cavity. Mixing runs on the host. Receipts use out/m4-mixture, including the pre-implementation build failure.

Particle constituent state, conservative transfer and remap, scene material resolution, and inspect/render publication remain open.

## Particle composition state receipt

Particles and resolved source templates now carry optional explicit phase and stable substance IDs with current-volume fractions. The native record reserves eight inline fractions to preserve bounded, trivially copyable particle state. Active fractions must be positive, strictly ordered and sum to one within FP64 tolerance. Legacy particles use no explicit composition, and optional wire fields preserve their existing canonical bytes.

The native codec fixture round-trips a two-constituent particle and source, rejects invalid sums and duplicate IDs, and preserves the legacy scene golden. A source replay fixture retained exact fractions through emission, retry and capacity replay. Thirty-nine CPU checks passed with ctest --preset windows-debug-cpu -j 4 -R 'LiquidCache3D|LiquidState3D|LiquidMixture3D|SourceRetryAndWholeFrameReplay'. A clean CUDA build and 22 serial real-device scheduler and boundary checks passed. The persistent-substep fixture explicitly verifies that CUDA advection retains both fractions. Both simulator builds and clang-format 19.1.5 passed. Receipts use out/m4-particle-composition.

The new fields are transported with the particle state. Phase-local grid reduction, constituent remap, scene substance definitions, and inspected/rendered fractions are still open.

## Phase-local constituent grid transfer receipt

The bounded host transfer deposits f J V0 onto cell centers with a trilinear tent. It selects one explicit phase at a time and retains separate substance volume channels in material order. Every positive stencil weight requires a represented cell, so cropping the supplied support fails instead of dropping inventory. This operator is ready for material-field assembly and inspection consumers.

Fixtures verified independent weights at all eight cell centers, conservation under unequal reference volume and compression, same-phase mixing to 75 percent clear water and 25 percent dyed water, and a separate oil channel. Twenty CPU tests passed with ctest --preset windows-debug-cpu -j 4 -R 'LiquidMixture3D|LiquidRemap|LiquidEos3D'. The real-device particle-composition regression passed. The grid transfer itself ran on the host. clang-format 19.1.5 passed. Receipts use out/m4-constituent-grid, including the initial fixture's incorrect hand-calculated z weights.

Material registry, scheduler and remap integration, and inspected/rendered composition remain open.

## Resolved liquid constituent registry receipt

Resolved scenes now carry an optional, ordered substance registry with stable IDs, phase membership and positive material densities. The scheduler owns its copy. Native and wire validation reject explicit particles or sources whose substances are missing or belong to a different phase. Legacy implicit particles require no registry.

The CPU codec fixture round-tripped two aqueous substances and a separate oil phase, verified canonical bytes, and rejected a cross-phase source. The CUDA scheduler fixture rejects a missing registry and then carries registered fractions through real-device advection. Forty-one CPU checks passed with ctest --preset windows-debug-cpu -j 4 -R 'LiquidCache3D|LiquidMixture3D|SourceRetryAndWholeFrameReplay'. Twenty-two serial real-device scheduler and boundary checks passed under windows-release-cuda. Both simulator targets built and clang-format 19.1.5 passed. Receipts use out/m4-constituent-registry.

The registry supplies the material identities and density data. Composition-dependent solver coefficients and published inspect/render fractions remain open.

## Published point composition receipt

The point cache now carries an optional stable phase ID and active constituent volume fractions copied from accepted particle state. Particle and point codecs share the same bounded wire handling. Inspect exposes explicit records in particle order under particle_composition with particle_id, phase_id and substance_id/volume_fraction pairs. Legacy point frames omit the optional fields.

The real-process CLI fixture published a mixed aqueous point and a separate oil point, then verified the exact inspected IDs and fractions. Seventy-one CPU cache and mixture tests passed with ctest --preset windows-debug-cpu -j 4 -R 'LiquidCache3D|LiquidCacheStore3D|LiquidMixture3D'. SimulateCli.inspection and SimulateCli.public_interface passed with the rebuilt fixture. Twenty-two real-device CUDA regressions and clang-format 19.1.5 passed. Receipts use out/m4-inspect-composition.

Inspect currently reports accepted particle fractions. Solver-side mixing and evaluated surface fractions still require integration.
