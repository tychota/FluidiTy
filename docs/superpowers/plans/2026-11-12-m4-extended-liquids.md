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

## Miscible density in pressure projection receipt

The shared scheduler now samples density from current-volume-weighted constituent mixtures in selected pressure cells. It uses the existing cell-centered tent and provides sampled density to CPU and CUDA pressure-link assembly. Unsensed cells retain the scene reference density. The local mixture density also supplies the reference density for the configured EOS law and sound speed. The capillary timestep check uses the lighter predictor density when applicable. Active explicit particles from more than one phase fail before pressure assembly until the shared-interface projection is available.

The single-cell fixture verified 1007.5 kg/m³ for equally weighted donors, the resulting inverse density on all six open links, and a Tait rest tangent of 4030 Pa at sound speed 2 m/s. Twenty-six CPU tests passed with ctest --preset windows-debug-cpu -j 4 -R 'Mixture|Eos|Tait|Drift|SourceRetryAndWholeFrameReplay'. Twenty-two real-device tests passed with ctest --preset windows-release-cuda -j 1 -R 'CudaLiquidSubstep3D|CudaLiquidBoundary3D'. The persistent CUDA substep carries the same registered composition as the CPU. clang-format 19.1.5 passed. Receipts use out/m4-mixture-pressure. Negative receipts retain the initial fixture with seven selected cells and the stale CUDA link failure resolved by a clean build.

This qualifies single-phase mixture pressure coefficients. Constituent remap, evaluated surface fractions, and immiscible interface projection remain open.

## Shared immiscible pressure link receipt

The production CPU and CUDA link builders accept a certified internal phase crossing in frozen face order. Each crossing has two liquid endpoints with explicit densities and a positive aperture. Its mobility uses the series density resistance along the center segment, and one internal pressure jump applies signed normal viscous stress minus surface tension times curvature. A malformed or external crossing fails validation.

The two-cell fixture measured mobility 1/850 m³/kg in both orientations, pressure differences of -2 and +2 Pa, and zero corrected interface velocity. Thirty-four CPU checks passed with ctest --preset windows-debug-cpu -j 1 -R 'LiquidBoundaryGeometry3D|Mixture'. Twenty-two real-device CUDA checks passed with ctest --preset windows-release-cuda -j 1 -R 'CudaLiquidBoundary3D|CudaLiquidSubstep3D'. Five Python multiphase oracle checks passed. Compute Sanitizer reported zero errors on CudaLiquidBoundary3D.GeometryAndCapillaryMatchOracle. clang-format 19.1.5 passed. Receipts use out/m4-shared-interface.

The caller supplies crossing geometry and stresses. Production phase-field construction, pair settings, the internal capillary timestep bound and shared render geometry still need integration.

## Internal capillary timestep receipt

The internal liquid crossing applies the two-density capillary limit sqrt((rho_tail + rho_head) h³ / (4 pi sigma)). Scaled FP64 arithmetic avoids overflow or underflow in the squared limit. The link builder accepts the limit itself and returns retryable Aborted above it. Zero surface tension imposes no capillary limit.

The shared CPU/CUDA fixture verified both density orientations, acceptance at the bound, rejection at the next representable duration, and the zero-tension path. The CPU range fixture checked density-sum overflow and a thin-cell limit with a representable square root. Forty-seven CPU checks passed with ctest --preset windows-debug-cpu -j 1 -R 'LiquidBoundaryGeometry3D|Cavity'. Twenty-two serial real-device regressions passed under windows-release-cuda. Compute Sanitizer reported zero errors on CudaLiquidBoundary3D.GeometryAndCapillaryMatchOracle. clang-format 19.1.5 passed. Receipts use out/m4-internal-capillary.

The scheduler still needs to generate internal crossings from phase fields and resolved pair settings. The current pressure builder enforces the bound once those crossings are supplied.

## Resolved phase-pair surface tension receipt

The resolved scene now carries one canonical tension record per configured pair of registered phases. Pair order is strict and both endpoint phases must exist. The scheduler owns its copy and lookup returns the same record in either orientation. The optional wire vector participates in scene fingerprints and preserves legacy encodings when absent.

The fixture round-tripped the water–oil pair, checked 0.031 N/m and symmetric lookup, and rejected reversed authoring order, an unknown phase and negative tension. Forty-two CPU checks passed with ctest --preset windows-debug-cpu -j 4 -R 'LiquidCache3D|LiquidMixture3D|SharedLiquidInterface'. Twenty-two serial real-device CUDA regressions passed under windows-release-cuda. Both simulator targets built. The production schema exposes liquid_phase_pairs and clang-format 19.1.5 passed. Receipts use out/m4-phase-pairs.

The pair table is ready for phase-field crossing assembly. The scheduler still needs to construct those crossings and use the pair settings in projection.

## Particle-derived two-phase projection receipt

The scheduler now supports an active configured pair in the common-radius particle-union field. The difference of nearest phase distances labels pressure centers and locates each crossing along its center segment. Density uses only same-phase donors, with nearest same-phase extrapolation outside tent support. The existing centered curvature operator evaluates the phase field at the crossing. Shared links apply pair tension and the internal capillary bound. This slice uses the split-viscosity approximation of zero normal viscous jump. A missing pair, a third active phase, legacy donors in the two-phase scene, active drift correction, or a phase with no resolved pressure cell refuses the step.

The planar water–oil fixture verified the shared crossing at one-half of the segment, zero signed curvature, densities of 1015 and 820 kg/m³, and a capillary retry. A co-located unresolved pair left committed state unchanged. The broad CPU scheduler, geometry and mixture suite passed 140 cases before the final guard. Eight focused CPU checks and 23 serial real-device CUDA checks passed after it. CPU and CUDA matched on the particle-derived crossing and retry. Compute Sanitizer reported zero errors on that CUDA case. clang-format 19.1.5 passed. Receipts use out/m4-particle-phase-crossings.

The phase reconstruction and density staging are host-side. Curved, moving and shared render interface qualification, constituent remap, and the full clear/dyed-water and oil workflow remain open.

## Uniform constituent split and merge receipt

The checked uniform APIC merge conserves each explicit constituent using current J V0 volume, while its kinematic moments retain the existing mass weights. The target keeps the ordered union of stable substance IDs. It refuses incompatible phases, composition modes or capacity. The existing admissibility callback sees the composed target and remains responsible for the selected EOS and support gate. Uniform split copies composition and checks each constituent share remains representable.

The unequal-J fixture merged and split 9.5 m³ and 1.5 m³ of two aqueous constituents and rejected a cross-phase merge before certification. Eighteen CPU checks passed with ctest --preset windows-debug-cpu -j 4 -R 'LiquidUniform|LiquidRemap|LiquidMixture'. Two real CUDA particle and phase scheduler regressions passed. The remap ran on the host. clang-format 19.1.5 passed. Receipts use out/m4-constituent-remap, including the initial checked-iterator failure in prefix insertion.

This qualifies uniform particle adaptation. Grid band/core constituent remap, evaluated surface fractions, shared render geometry and full M4 workflow evidence remain open.

## Reconstructed surface composition receipt

Each reconstructed exterior vertex inherits the phase of its nearest particle. An exact distance tie uses phase ID and then particle ID. A two-cell linear tent weights same-phase donors by accepted current volume. The nearest particle supplies the fractions outside tent support. The surface cache stores explicit phase IDs and stable constituent fractions in an optional vertex-ordinal table, preserving the existing fixed vertex representation.

The reconstruction and cache round-trip fixture checked phase 4 and the expected 25/75 mixture at every exterior vertex. The debug CPU simulator built and all 13 focused tests passed with ctest --preset windows-debug-cpu -j 4 -R 'EvaluatesAndCachesMixed|SurfaceCache3D|SurfaceMesh3D|SurfaceTracking3D'. clang-format 19.1.5 and git diff --check passed. Receipts use out/m4-surface-composition.

Named renderer attributes, shared liquid–liquid render geometry and end-to-end three-source qualification remain open.

## Blender named surface attribute carrier probe

Blender 5.2.3 LTS e13b728c6880 round-tripped a mesh through its Alembic exporter and native Mesh Sequence Cache reader. The evaluated mesh exposed neither the custom point float nor the corner color. A USD round trip preserved the point float and the integer phase. A separate time-sampled USDA mesh produced an evaluated Mesh Sequence Cache after saving and reopening a blend from factory startup. Frames one, two and four returned distinct expected fraction and phase values, and frame three had zero vertices and faces with empty attribute arrays. The extension was absent from this factory-startup probe.

The receipts and fixture are out/m4-alembic-attribute-probe.*, out/m4-usd-attribute-probe.* and out/m4-usd-dynamic-probe.*. Each Python fixture runs with Blender 5.2.3 -b --factory-startup --python-exit-code 1 --python FIXTURE.py from the worktree root.

Use an authenticated Mesh-cache export to native USD for named render fields. Preserve the existing Alembic surface export. Name fraction attributes fluidity_fraction_<stable substance ID>. Represent phase membership with per-phase named fields so full-width stable IDs are carried in names and never narrowed for Blender. Export time-sampled vertex fields with zero values for absent constituents and preserve the cache timeline and orientation. Verify shader consumption in a fresh extension-disabled Blender process on the production carrier.

## Production named surface export receipt

The Blender bridge command export-surface-usd authenticates the contiguous committed Mesh frames and publishes a native USD cache without replacing an existing destination. Point primvars fluidity_fraction_<substance ID> and fluidity_phase_<phase ID> use stable IDs in their names. Phase fields are one-hot at mesh vertices. Missing values are zero. The export uses one decoded mesh at a time and rereads the authenticated frames for each property, so export cost grows with the number of distinct fields. The carrier uses solver frame ordinals at 24 fps and Blender world axes.

The mixed-surface fixture baked frames zero and one through production Simulate and Mesh. Blender 5.2.3 read phase 4 and 25/75 percent substance fractions on every vertex at both frames. A fresh factory-startup process reopened the saved blend, verified the attributes and checked the native material node bindings. The bridge rejected overwriting the carrier and cleaned staging after a missing-frame failure. The first incremental Release run crashed with stale native objects after the vertex layout change. A clean Release build with the pinned Alembic package fixed the crash. All 725 debug CPU tests passed on the preceding candidate after building all registered test executables. clang-format 19.1.5 and git diff --check passed for this slice.

From the worktree root with the debug CPU solver and fixture and the Release Blender bridge built, run the commands below against a fresh output directory. Receipts use out/m4-usd-mixed-retry and out/m4-usd-bridge-clean-release.log.

```powershell
$cpu = 'out/build/windows-debug-cpu'
$bridge = 'out/build/windows-release-cuda/fluidity_blender_bridge.exe'
$blender = 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe'
$run = 'out/m4-usd-qualification'
New-Item -ItemType Directory $run
& "$cpu/fluidity_simulate_fixture.exe" scene "$run/scene.flrs" "$cpu/fluidity_simulate.exe" cpu mixed-surface
& "$cpu/fluidity_simulate.exe" run --stage simulate --scene "$run/scene.flrs" --initial-state "$run/scene.flrs.flst" --cache "$run/simulate" --frame-end 1
& "$cpu/fluidity_simulate.exe" run --stage mesh --cache "$run/simulate" --output-cache "$run/mesh" --surface-spacing 0.125 --surface-sigma 0.25 --frame-end 1
& $bridge export-surface-usd "$run/mesh" 1 "$run/surface.usda"
& $blender -b --factory-startup --python-exit-code 1 --python tools/blender/verify_m4_surface_usd.py -- prepare "$run/surface.usda" "$run/surface.blend"
& $blender -b --factory-startup --python-exit-code 1 --python tools/blender/verify_m4_surface_usd.py -- reopen "$run/surface.blend"
```

The bridge has no artist-facing attachment operator for this carrier yet. The fixture validates saved node bindings and evaluated geometry attributes. Rendered shader response, a shared water–oil interface, and the three-source workflow remain open.

## Shared two-phase lattice partition receipt

The two-phase extractor partitions a common piecewise-linear exterior into A-to-air and B-to-air triangles. It clips one A-to-B sheet to the liquid and uses the same indexed seam on all incident sheets. The edge and face keys use the existing six-tetrahedron lattice subdivision. Coincident plane segments and degenerate geometry return errors. The manufactured oblique cut of a sphere closed both phase volumes with opposite uses of the same interface triangles. Each volume had Euler characteristic two and zero open edges, nonmanifold edges or vertices, winding defects, degenerate faces or self-intersections. Their summed volume matched the unpartitioned exterior to 1e-12 m³. The fixture also checked the vertex budget.

Ten CPU tests passed with ctest --preset windows-debug-cpu -j 4 -R 'SurfaceMesh3D|EvaluatesAndCachesMixedSurfaceFractions'. clang-format 19.1.5 and git diff --check passed. Receipts use out/m4-phase-partition. The clean Release build from the preceding candidate passed 23 focused real-device CUDA checks in out/m4-usd-clean-release-cuda-tests.log.

Production integration must sample the existing particle phase field, preserve seam incidence through tracking and volume correction, qualify both closed phases after correction, and carry the shared faces and phase pair through the cache and renderer. This receipt covers the raw lattice partition.

## Shared phase sheet cache receipt

The optional Mesh frame record carries an ordered stable phase pair, one label per exterior triangle, and one oriented A-to-B internal sheet. Internal triangles address the existing exterior indices at the seam and append positions only for interior vertices. Read and write enforce combined record budgets, reconstruct both closed phase boundaries from the same shared sheet, and reject open, nonmanifold, intersecting or wrongly wound boundaries. The fixed exterior vertex representation and legacy payloads are preserved.

A tetrahedron split by an oblique plane round-tripped byte-for-byte with one interior vertex and one shared seam. Wrong exterior labels, flipped shared winding and an out-of-range index were rejected. The combined vertex budget also refused the fixture on decode. Five focused CPU tests passed with ctest --preset windows-debug-cpu -j 4 -R 'SurfaceCache3D|PartitionsTwoClosed|EvaluatesAndCachesMixed'. clang-format 19.1.5 and git diff --check passed. Receipts use out/m4-shared-surface-cache.

The codec is ready for the production reconstruction to preserve the shared seam through correction and for USD to publish the internal sheet.

## Production shared surface reconstruction receipt

The Mesh reconstruction now samples the same nearest-phase field as pressure projection. It extracts both labeled exterior patches and one shared sheet on a common lattice anchored to the lowest particle ID. Exterior volume correction moves the shared seam indices directly. The cache validator then certifies both closed phase boundaries. The moving path retains exterior IDs when indexed connectivity and phase labels agree and displacement is bounded by one cell width. A change in the anchor, connectivity or phase appearance requires seam lineage and currently refuses the frame.

The production reconstruction fixture advanced two immiscible particle phases through an accepted CPU scheduler frame and rebuilt the phase sheet from its MotionInterval. The sheet retained connectivity, every exterior vertex retained its ID, and geometric velocity matched the prescribed 0.24 m/s translation to 1e-8 m/s. The corrected seed exterior met the 0.5% volume gate, and the encoder validated the closed phase boundaries for both frames. Initial negative receipts show connectivity drift on a fixed world lattice and an invalid manufactured carrier without accepted velocity faces.

The debug CPU simulator built. Six focused CPU cases passed with ctest --preset windows-debug-cpu -j 4 -R 'PublishesSharedSheet|EvaluatesAndCachesMixed|SurfaceCache3D|PhaseCrossing'. clang-format 19.1.5 and git diff --check passed. Receipts use out/m4-production-shared. USD publication of the sheet, topology-changing phase tracking and the three-source render gate remain open.

## Shared USD geometry and native material receipt

The production USD bridge writes one combined mesh. It appends shared A-to-B triangles once, using the exterior vertex indices at the seam. Uniform primvars fluidity_face_phase_<ID> identify the exterior phase and fluidity_interface_<A>_<B> identifies the shared faces. Point fractions remain keyed by stable substance ID. After volume correction, vertices incident to only one exterior phase use donors from that certified phase. Seam vertices use nearest-phase composition and the material uses face masks to distinguish the two exteriors.

The three-substance seed fixture baked through production Simulate and Mesh for frames zero and one. Inspect confirmed stable phase and substance IDs on the three particles. Blender 5.2.3 read 396 aqueous, 384 oil and 220 shared triangles on each frame. It found 54 edges incident to all three sheets, with two faces on every other edge and no duplicate triangles. The aqueous interior of the exterior patch measured 25/75 clear/dyed water and zero oil. The oil patch measured pure oil. A fresh factory-startup Blender process reopened the blend and rendered 444 dye-colored pixels and 754 oil-colored pixels. Disconnecting the dye input yielded zero dye-colored pixels. The pixel check uses the Standard view transform with a 64 × 64 Cycles emission render.

From the worktree root with the CPU solver and fixture and the Release Blender bridge built, run these commands against a fresh output directory. Receipts use out/m4-shared-usd-production. clang-format 19.1.5 and git diff --check passed.

```powershell
$cpu = 'out/build/windows-debug-cpu'
$bridge = 'out/build/windows-release-cuda/fluidity_blender_bridge.exe'
$blender = 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe'
$run = 'out/m4-shared-usd-qualification'
New-Item -ItemType Directory $run
& "$cpu/fluidity_simulate_fixture.exe" scene "$run/scene.flrs" "$cpu/fluidity_simulate.exe" cpu shared-surface
& "$cpu/fluidity_simulate.exe" run --stage simulate --scene "$run/scene.flrs" --initial-state "$run/scene.flrs.flst" --cache "$run/simulate" --frame-end 1
& "$cpu/fluidity_simulate.exe" check --cache "$run/simulate" --frame 1
& "$cpu/fluidity_simulate.exe" run --stage mesh --cache "$run/simulate" --output-cache "$run/mesh" --surface-spacing 0.25 --surface-sigma 0.25 --frame-end 1
& $bridge export-surface-usd "$run/mesh" 1 "$run/surface.usda"
& $blender -b --factory-startup --python-exit-code 1 --python tools/blender/verify_m4_shared_surface_usd.py -- prepare "$run/surface.usda" "$run/surface.blend"
& $blender -b --factory-startup --python-exit-code 1 --python tools/blender/verify_m4_shared_surface_usd.py -- reopen "$run/surface.blend"
```

This fixture uses a native seed with three pure-substance particles. Authoring three emitting sources and their interaction in Blender, artist-facing USD attachment, phase topology changes and final qualification remain open.

## Blender-authored three-source scene receipt

Blender source boxes now carry stable substance and miscibility-phase IDs, rest density, and optional center emission with a configured particle volume and rate. The authoring adapter uses deterministic box-local seed quadrature. The Interactions control supplies the pair tension. The v3 packet resolves the registry, pair table, seed compositions and emitter templates into the native scene and state. An emitter with an unknown substance and a reversed pair were rejected before output publication. The single-liquid controls still passed the pinned Blender free, plane, slab and tank bake regressions.

The headless scene authored three boxes through the extension and set the water–oil tension to 0.031 N/m. The CPU solver published three initial particles and one accepted child from each emitter in frame one, with clear and dyed water in phase 4 and oil in phase 5. Production Mesh and USD carried 398 aqueous, 382 oil and 218 shared faces per frame with 54 three-sheet seam edges. The evaluated aqueous patch had both water constituents and zero oil. A fresh factory-startup Blender 5.2.3 process reopened and rendered 433 dye-colored and 748 oil-colored pixels. Removing the dye link yielded zero dye-colored pixels.

With the Release bridge and debug CPU solver built, run the following from the worktree root. Receipts use out/m4-blender-authored-nonzero and out/m4-blender-authored-final. The full debug CPU run on the preceding shared USD candidate passed all 728 cases. clang-format 19.1.5 and git diff --check passed.

```powershell
$solver = 'out/build/windows-debug-cpu/fluidity_simulate.exe'
$bridge = 'out/build/windows-release-cuda/fluidity_blender_bridge.exe'
$blender = 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe'
$run = 'out/m4-authored-source-qualification'
& $blender -b --factory-startup --python-exit-code 1 --python tools/blender/verify_m4_liquid_sources.py -- $bridge $solver $run
& $solver run --stage mesh --cache "$run/simulate" --output-cache "$run/mesh" --surface-spacing 0.25 --surface-sigma 0.25 --frame-end 1
& $bridge export-surface-usd "$run/mesh" 1 "$run/surface.usda"
& $blender -b --factory-startup --python-exit-code 1 --python tools/blender/verify_m4_shared_surface_usd.py -- prepare "$run/surface.usda" "$run/render.blend"
& $blender -b --factory-startup --python-exit-code 1 --python tools/blender/verify_m4_shared_surface_usd.py -- reopen "$run/render.blend"
```

The headless workflow still invokes the Mesh and USD bridge commands explicitly. Artist-facing surface attachment, broader moving and curved phase qualification, adaptive constituent remap and final acceptance remain open.

## Artist-facing surface attachment receipt

Import & Export now exposes Attach Surface Cache for a committed Mesh directory. It verifies the cache through the bridge, imports a project-relative native USD carrier, and compares the last evaluated geometry counts against the authenticated bridge receipt before replacing the live binding. Reattaching identical bytes reuses the existing object, mesh and carrier. Relocation and static/animated replacement preserve the object and assigned material. The action restores frame and selection and leaves a working binding in place when validation fails.

Pinned Blender 5.2.3 exercised the public operator against the Blender-authored three-source Mesh cache. Same-cache attach, relocation, animated-to-seed replacement, seed-to-animated replacement, and refusal of a corrupted payload all passed. A fresh factory-startup process reopened the resulting blend with the extension absent, found the shared sheet and material, and measured 433 dye and 748 oil pixels, with zero dye pixels in the control. The bridge built and formatting checks passed. Receipts use out/m4-surface-attachment-static.

```powershell
$blender = 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe'
$run = 'out/m4-attach-qualification'
New-Item -ItemType Directory $run
& $blender -b --factory-startup --python-exit-code 1 --python tools/blender/verify_m4_shared_surface_usd.py -- attach out/m4-blender-authored-nonzero/mesh out/build/windows-release-cuda "$run/attached.blend"
& $blender -b --factory-startup --python-exit-code 1 --python tools/blender/verify_m4_shared_surface_usd.py -- reopen "$run/attached.blend"
```

Mesh generation still uses the separate native stage command. This receipt qualifies public attachment to its committed output.

## Constituent inventory through band and core ownership

Merged pinned main b540f2d locally as 54efed6 to adopt the accepted M3 band/core handoff API. The native core inventory carries one phase and sorted extensive constituent current volumes. Band retirement sums J V0 times each fraction, and reseeding reconstructs fractions from the core owner and checks the actually stored particle inventories. Adaptive restriction accumulates each constituent in child order. Prolongation distributes the parent fractions across the limited child current volumes and certifies each restricted total. An incompatible wet neighbor supplies the parent state at the phase boundary, and mixed-phase restriction or ownership transfer refuses the operation.

The full-cell ownership round trip conserved 6 m³ clear and 2 m³ dyed water through band retirement, eight-child prolongation, restriction and reseeding with oil present in the stencil. A separate nonuniform refinement fixture preserved 3 m³ and 9 m³ across child current volumes of 1.375 and 1.625 m³. Malformed constituent totals, cross-phase owners and the ninth constituent were rejected. Twenty CPU checks passed with ctest --preset windows-debug-cpu -j 4 -R 'LiquidBandHandoff3D|LiquidRemap3D|LiquidUniform'. clang-format 19.1.5 and git diff --check passed. Receipts use out/m4-constituent-band.

Main and origin/main were both at b540f2d. M3 family-remap integration fdc3c28 and topology/exterior-ownership integration 1af7b0f remain outside main. The verified M4 APIs can carry composition through full single-phase cells. Production adaptive scheduling, checkpoint ownership and coarse/fine traction still need the accepted M3 execution path and its qualification receipts.

## Full-suite and capillary requalification started at bed51fd

The complete Debug CPU build and ctest --preset windows-debug-cpu -j 6 passed 734 of 734 tests. A Release incremental build produced inconsistent Mesh result layouts after header changes. Its Ninja compiler-dependency prefix was mojibake while compiler output used French inclusion lines. The affected incremental results are retained in out/m4-bed51fd-windows-release-cuda-tests.log. Reconfigure and build in a fresh directory before using Release results. The clean Release build and suite receipts use out/m4-bed51fd-release-clean.

The public capillary runner now launches the frozen 216-frame oscillating-drop-l2 fixture, authenticates committed frame records through the native reader, and checks step traces, centered volume-weighted quadrupole, same-sign extrema, frequency, damping and volume drift. It stores raw process streams and binary hashes. A synthetic damped sinusoid with angular frequency 2 rad/s measured 2.000005 rad/s and damping 0.0499374, against 0.04994. The full CPU run, clean Release test suite, and pinned Lean independent proof run were still active when this receipt was recorded.

```powershell
$build = 'out/build/m4-bed51fd-release-clean'
uv run --project reference --locked python -B tools/native/tests/run_capillary_drop.py --solver "$build/fluidity_simulate.exe" --fixture "$build/fluidity_liquid_acceptance_fixture.exe" --backend cpu --output out/m4-bed51fd-drop-cpu
```

## Blender Mesh build and resume receipt

Build Surface and Resume Surface now run the committed Bake cache through the native Mesh stage using Blender's background process, progress and cancellation controls. The panel exposes the output path, sampling spacing and Gaussian width. Attachment opens that output directory by default. The headless fixture built and resumed frames zero and one from the Blender-authored clear-water, dyed-water and oil bake. It compared the upstream publication bytes, rejected reuse without Resume, exercised relocation and a corrupt replacement, and rendered the saved attachment in a fresh extension-disabled process. The shared face count, 54 seam edges and dye and oil render response matched the earlier receipt. Free, plane, slab and tank simulation bakes still passed after the job-runner refactor.

Receipts use out/m4-build-surface-final and out/m4-build-surface-domain-regression.log. To reproduce from the qualified authored Simulate cache:

```powershell
$blender = 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe'
$run = 'out/m4-build-surface-qualification'
New-Item -ItemType Directory $run
& $blender -b --factory-startup --python-exit-code 1 --python tools/blender/verify_m4_shared_surface_usd.py -- build-attach out/m4-blender-authored-nonzero/simulate out/build/m4-bed51fd-release-clean "$run/attached.blend"
& $blender -b --factory-startup --python-exit-code 1 --python tools/blender/verify_m4_shared_surface_usd.py -- reopen "$run/attached.blend"
```

## Capillary, proof and validator results on bed51fd / c60a546

Both frozen 216-frame drop runs passed on the clean Release solver. CPU measured 1.9245825615 rad/s and CUDA measured 1.9245825497 rad/s against 2.0238577025 rad/s, with 4.90524% relative error, damping 0.135843, and relative volume drift below 0.000418. All 1080 accepted steps on each backend were capped, and the largest accepted-to-bound ratio was 0.23633. Receipts use out/m4-bed51fd-drop-cpu and out/m4-bed51fd-drop-cuda. The same clean Release build passed all 795 CTest cases, including the exact static-sphere rest check at both declared resolutions. The Debug CPU build passed all 734 cases. Independent Lean proof verification passed and saved out/m4-bed51fd-proof/evidence.json.

WSL passed all 1331 Python tests from c60a546, including real symlink provenance. Its preceding 360 failures were traced to a synthetic CUDA report omitting mandatory viscosity, compliance and pressure cases. The repair kept those cases and added the production phase-crossing and cavity regressions to the required operator set. All three passed on the real device and under Compute Sanitizer with zero errors. Receipts use out/m4-c60a546-python.log and out/m4-bed51fd-phase-cavity-memcheck.log.

## Resolved constituent viscosity receipt

The optional substance coefficient is serialized in the resolved registry. Absent coefficients use the scene viscosity. The declared miscible model is the arithmetic current-volume-weighted viscosity, sampled with the existing cell tent and same-phase fallback. The coupled CPU and CUDA projector accept a viscosity per frozen pressure cell. The legacy scene-wide coefficient still uses the scalar path. Full-cell and full-face or sealed-face geometry remains the supported quadrature. Curved cut-cell and adaptive strain, resolved immiscible production traction, and final candidate qualification remain open.

The CPU fixture resolved a 25/75 mixture of 40 and 120 Pa s. All four pressure cells received 100 Pa s, including cells using nearest-donor continuation, and the committed state matched the uniform 100 Pa s control byte for byte. Forty-four focused Debug CPU cases passed with ctest --preset windows-debug-cpu -j 4 -R 'ConstituentViscosity|SchedulerConsumesResolvedViscosity|LiquidCache3D|PhaseCrossing|LiquidMixture'. The CPU simulator built and clang-format 19.1.5 passed. The real-device heterogeneous-viscosity fixture was in a fresh Release build at the time of this entry. Receipts use out/m4-constituent-viscosity.

## Planar immiscible coupled viscosity receipt

A stationary tank with a planar free surface now supplies exterior geometry independently of its internal particle-derived phase field. The CPU scheduler fixture verified two water–oil pressure crossings, both at the center of their links, zero planar curvature, and the separate 40 and 120 Pa s cell coefficients entering the coupled solve. The accepted step damped tangential motion with a momentum residual below 1e-5 kg m/s. Eight CPU cases passed, including the high-contrast extension and Couette manufactured checks.

Five real-device CUDA tests passed, including the expanded stationary-tank parity fixture with uniform, miscible and immiscible material controls. It checked agreement with the CPU committed state and unchanged device allocation/free counts during each step. Receipts use out/m4-planar-phase-viscosity. clang-format 19.1.5 and git diff --check passed. The full-cell qualification does not cover curved cut-cell, moving no-slip wall, or adaptive strain quadrature.

## Authored immiscible viscosity receipt

Blender sources expose an optional dynamic viscosity in Pa s. The stationary tank can use an explicit planar fill height. Packet v4 carries both settings while v3 remains the default for the earlier source workflow. The resolver validates the viscosity, tank and seed height before publication. In Blender 5.2.3, the headless fixture authored a planar tank with distinct 40 and 120 Pa s substances and a 0.031 N/m interface. The production CPU solver accepted a traced frame and all coupled momentum residuals were below 1e-5 kg m/s. The previous v3 three-emitter fixture passed through the same updated bridge. Receipts use out/m4-blender-viscous-tank and out/m4-blender-v3-viscosity-regression.log.

```powershell
$blender = 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe'
$build = 'out/build/m4-constituent-viscosity-release'
& $blender -b --factory-startup --python-exit-code 1 --python tools/blender/verify_m4_liquid_sources.py -- viscous "$build/fluidity_blender_bridge.exe" "$build/fluidity_simulate.exe" out/m4-authored-viscosity-qualification
```

This fixture uses the full-cell planar tank path. Its phase interaction and coefficients are qualified through Simulate. Curved and adaptive viscous quadratures remain outside the verified geometry.

## Viscosity scratch allocation regression

The full Debug suite at a47d8a0 caught one allocation/free on the inviscid attempt path. MSVC checked iterators allocate a proxy even for an empty vector. The scheduler now reserves constituent viscosity records during construction and transactional growth, includes them in its staging byte budget and borrows a span during the attempt. Legacy scenes reserve no coefficient records. The four failed allocation tests, the constituent and planar immiscible viscosity cases, and the workspace filter passed in the 75-case rerun. Receipts use out/m4-viscosity-scratch-cpu and the original failures remain in out/m4-a47d8a0-cpu-tests.log.

## Shared-surface retessellation receipt

The production Mesh path can now rebuild a shared phase sheet when its triangle connectivity changes. After volume correction it matches predicted old vertices to reciprocal nearest new vertices within half a source cell width and with the same incident exterior phase labels. Matches retain IDs and age, and unmatched vertices receive fresh IDs with zero initial geometric velocity. The cache still checks closure and intersection for both phase regions. The unchanged-connectivity path retains the previous indexed correspondence for bounded motion.

The accepted CPU deformation fixture changed the oil velocity by 0.75 m/s. The next Mesh frame changed connectivity, retained some exterior IDs, allocated fresh IDs for the other vertices, and encoded successfully with a valid shared sheet. Reconstruction replay from the same predecessor and accepted motion produced identical bytes. The translation fixture still retained all IDs and measured 0.24 m/s. Six focused CPU cases passed with ctest --preset windows-debug-cpu -j 4 -R 'PublishesSharedSheet|SurfaceCache3D|PartitionsTwoClosed|EvaluatesAndCachesMixed'. clang-format 19.1.5 and git diff --check passed. Receipts use out/m4-retessellated-phase-replay. The previous 642579f Debug CPU build also passed all 736 cases.

This qualifies retessellation of a continuing phase pair. Phase appearance/disappearance, changes in the number of connected liquid components, and curved cut-cell and adaptive viscous traction require further evidence.

## Enclosed curved shared geometry receipt

The common partition was checked with a 0.73 m phase sphere wholly inside a 1.63 m liquid sphere. At both 0.5 and 0.25 m lattice spacing, the internal sheet had one positive closed component, Euler characteristic two and zero open edges, nonmanifold defects, degenerate triangles, orientation defects or self-intersections. It shared no vertices with the air boundary. Halving spacing reduced inner volume error to less than 40 percent of the coarse error, with fine error below 8 percent against the analytic sphere.

Three CPU cases passed with ctest --preset windows-debug-cpu -j 4 -R 'EnclosedCurvedPhaseSheet|PartitionsTwoClosed|PublishesSharedSheet'. The receipt covers the geometric extraction of a prescribed curved field. It does not qualify the dynamic pressure/strain of a curved internal interface or material sampling on its two sides. Receipts use out/m4-curved-sheet.

## Two-sided shared interface composition receipt

Mesh evaluates the same phase-local current-volume tent on both sides of every shared-sheet vertex, including vertices wholly inside the liquid. The optional checked cache table stores both samples in vertex order and rejects incomplete coverage or a substance assigned to both phases. USD keeps one indexed sheet and publishes fluidity_interface_fraction_<substance ID> as point fields, with zero away from that sheet. Exterior fraction fields retain their previous names and sampling behavior.

Five focused CPU tests passed with ctest --preset windows-debug-cpu -j 4 -R 'PublishesSharedSheet|SurfaceCache3D|EnclosedCurvedPhaseSheet'. The authored three-source Simulate and Mesh run, native USD export, and fresh Blender reopen verified both frames. The shared and interior-only vertices contained 25/75 percent clear/dyed water on phase 4 and pure oil on phase 5, with 218 shared faces and 54 seam edges per frame. To reproduce, run the authored source and Mesh commands from the three-source receipt, export USD, run the existing prepare command, and run verify_m4_shared_surface_usd.py with sides and the saved blend in a fresh Blender 5.2.3 process. Receipts use out/m4-interface-composition-production and out/m4-interface-composition-final-tests.log. clang-format 19.1.5 and git diff --check passed.

## Curved internal capillary pressure receipt

A stationary planar tank encloses a particle-derived spherical phase of radius 0.8 m, represented by a central phase-A donor and 1000 phase-B donors on a 1.6 m shell. The 0.5 m grid and accepted 0.001 s step used production field sampling, link assembly, pressure solve and publication. The manufactured 3.1 N/m jump measured 24 links, 2.54381 percent signed curvature RMS error and 7.43946 Pa mean core pressure against 7.75 Pa analytic. Both curved and planar crossing tests passed. Run ctest --preset windows-debug-cpu -V -R 'MeasuresLaplacePressureOnCurvedParticlePhase|DiscoversParticlePhaseCrossing'. Receipts use out/m4-curved-phase-pressure.

The otherwise identical 0.031 N/m probe measured the same curvature and published zero core pressure. At this timestep and density the pressure solver's absolute flux and cell-defect floors can accept the zero initial iterate. The negative result is retained in out/m4-curved-phase-pressure-small-signal.log. The manufactured stronger signal qualifies jump sign and magnitude above that floor. Small-signal relative accuracy, curved viscous traction and adaptive traction remain open.

## Small internal capillary jump accuracy receipt

For a nonzero open internal pressure jump with a positive assembled signal norm, pressure convergence caps the integrated residual at 1e-3 of the signal. This is in addition to the existing global and cell-volume-scaled defect gates. The bounded CPU and CUDA pressure path, owning CUDA path and coupled Schur solve use the same criterion. The coupled path scales by its initial measured flux residual.

The original 0.031 N/m curved-phase case now measures 0.0755401 Pa against 0.0775 Pa analytic on the inviscid CPU path and 0.0749891 Pa with coupled viscosity, using the same 24 crossings and 2.54381 percent curvature RMS. This replaces the zero-pressure result reported in the preceding curved-interface receipt. Seventeen focused Debug CPU tests passed. Seven Release cases passed, including six real-device tests and the CUDA mixed-link oracle with its forcing scaled by 1e-8. The CUDA pressure matched the scaled reference within the fixture's tolerance. Receipts use out/m4-small-jump-pressure-final-tests.log and out/m4-small-jump-pressure-cuda-tests.log. clang-format 19.1.5 and git diff --check passed.

## Candidate 0c06750 and remaining geometry dependencies

The Debug CPU build and all 738 CTest cases passed on 0c06750. The commands were uv run --project reference --locked --group native cmake --build --preset windows-debug-cpu -j 7 and ctest --preset windows-debug-cpu -j 6. Receipts are out/m4-0c06750-cpu-build.log and out/m4-0c06750-cpu-tests.log.

The frozen pressure face aperture measures the portion outside solid, and the pressure-cell volume measures the intersection of liquid and nonsolid. Neither records the liquid intersection of the edge-centered dual cells used to integrate shear. Full-cell strain uses one cell-center normal quadrature and one shear quadrature for each complete four-cell edge stencil. Uniform cut viscosity needs consistent cell and edge quadratures and a boundary treatment for incomplete stencils. Reuse the frozen liquid corner samples and obstacle clipping to integrate the appropriate stress volumes. Validate the pressure-derived inertia and adjoint with the same frozen geometry. Test affine strain, zero strain under rigid motion, moving-wall work, and traction convergence before admitting these scenes. The existing runtime guard reports the unsupported quadrature when a positive-viscosity trial reaches a partial cell or face.

The M3 remap integration branch at 9282854 adds a production dependency record, with conservative family remap at fdc3c28. The outstanding native entry points are a graded trial topology with canonical coarse/fine face ownership, exterior and moving-wall links in the scalable adaptive pressure graph, and transaction/checkpoint publication of accepted adaptive state. The M4 remap oracle can exercise constituent conservation independently. Qualifying adaptive viscosity and composition through accepted Simulate frames requires those M3 entry points and their replay evidence.

## Shared-surface topology review follow-up

Independent review of 0c06750 identified that valid per-frame closure can hide an untracked phase event. A regression constructs a valid predecessor with the same exterior and one extra closed oil pocket. The old matcher returned OK when the pocket disappeared. The matcher now compares exterior and per-phase component counts, orientation counts and Euler characteristic before transferring IDs through retessellation. A mismatch returns FailedPrecondition and requires lineage. The existing ordinary deformation still retains IDs and replays byte for byte. These aggregate invariants detect the tested change but do not establish component lineage, including events with equal topology counts.

Six focused CPU tests passed with ctest --preset windows-debug-cpu -j 4 -R 'PublishesSharedSheet|SurfaceCache3D|EnclosedCurvedPhaseSheet|PartitionsTwoClosed'. The pre-fix regression and passing run are out/m4-shared-topology-red.log and out/m4-shared-topology-green.log.

On the preceding 0c06750 source, the clean Release CUDA suite passed 799 tests, including 62 GPU-labeled cases. WSL passed all 1331 Python checks from a source archive. The independent proof checker passed and retained its evidence JSON. Blender 5.2.3 authored the three sources and accepted one child per emitter, built and resumed Mesh, relocated the attachment and rejected a corrupt replacement. A fresh process measured 218 shared faces and 54 seam edges on both frames, both shared-side compositions, and 433 dye and 748 oil pixels. The dye-disconnected control had zero dye pixels. Receipts use out/m4-0c06750-cuda-tests.log, out/m4-0c06750-python.log, out/m4-0c06750-proof and out/m4-0c06750-blender.

## Reviewed 9ef5fb8 qualification receipts

The reviewed topology fix passed all 738 Debug CPU and 799 clean Release CUDA CTest cases. The independent follow-up review found no remaining regression in supported retessellation. Blender 5.2.3 reran source authoring, accepted emission, Mesh Build/Resume, attach/relocation, fresh-process rendering and two-sided composition. Both frames retained 218 shared faces and 54 seam edges, with 25/75 clear/dyed water on one side and pure oil on the other. The reopened render measured 433 dye and 748 oil pixels. Disconnecting dye gave zero dye pixels. Receipts use out/m4-9ef5fb8-cpu-tests.log, out/m4-9ef5fb8-cuda-tests.log and out/m4-9ef5fb8-blender.

The preceding 0c06750 source passed the frozen 216-frame capillary drop on CPU and CUDA. Frequency error was 4.90524 percent on both, damping ratio 0.135843, peak relative volume drift 0.000417287 or less, and all 1080 accepted steps per backend were capillary-bounded. The maximum accepted-to-bound ratio was 0.236328. The real-device probe reported GeForce RTX 5060 Ti, compiled and executed SM 120 and successful launch, copy and synchronization. Compute Sanitizer reported zero errors on the three phase crossing and cavity tests. Receipts use out/m4-0c06750-drop-cpu, out/m4-0c06750-drop-cuda, out/m4-0c06750-device.log and out/m4-0c06750-memcheck.log. The subsequent code diff is confined to Mesh/cache topology and its regression.

## Isolated uniform cut-strain slices

The cut-strain worktree branches from 9ef5fb8 so the reviewed candidate and qualification binaries remain intact. Commit 3256a58 clips the original liquid tetrahedra against the frozen obstacle and the eight octant boxes. The analytic oblique cut, halfspace and tank results, zero plateau, and non-affine liquid-plus-gas conservation passed. Commit 3d43671 uses the sum for normal strain and the incident quarters for complete edge shear. The shear test prescribed unequal quarters and viscosities with a 9/16 m³ support and 4 Pa s weighted harmonic viscosity. The measured tensor quadrature was 9 kg m²/s. All 77 focused Debug CPU tests passed. From the cut worktree, run ctest --preset windows-debug-cpu -j 4 -R 'LiquidBoundaryGeometry3D|Viscosity|MeasuresLaplacePressureOnCurvedParticlePhase'. The build and test receipts are out/m4-cut-strain-build.log and out/m4-cut-strain-green.log in that worktree.

Next, use the scheduler's same-epoch corner samples and obstacle to assemble the cut rows, compare their summed cell volumes with the frozen pressure volumes, and use only owned pressure velocity slots. The complete-edge operator needs a boundary closure for incomplete shear stencils. The acceptance checks need to establish affine strain, rigid-motion null modes where admitted, pressure adjoint consistency, traction convergence, and moving-wall work. Retain the production refusal for cut geometries until their closure is implemented and measured. The adaptive projection and accepted ownership path remain dependent on M3.

## Uniform cut-free-surface production receipt

The isolated cut implementation was incorporated into the local M4 branch through 58b8d5c. Both native backends use the pressure freeze's corner samples and compare summed stress octants with the frozen liquid volume. Normal rows use that volume. Each positive edge shear row uses two pressure-owned derivative pairs reached through open internal pressure links. The closure reproduces affine strain, annihilates rigid motion in its manufactured fixture and rejects disconnected or unavailable support before publication. A failed thin support and a solid cut both retain the exact committed checkpoint bytes. The real CUDA arena reserves at most three normal and twelve shear rows per pressure cell, with no allocation or free during the fixture's attempt.

The resting sphere passed at three and five cells per radius. The moving pure-shear sphere at 10 Pa s and 1/120 s retained 0.9176789 of the inviscid control's particle kinetic energy, within its fixed 0.95 ratio gate. CPU/CUDA differences were zero in position, 1.39e-17 m/s in particle velocity and 6.02e-14 Pa in pressure. The earlier mixed rotation/shear probe retained about 0.9824 of the control and failed the same gate. Its negative result is retained. The motion control was then changed to a symmetric affine shear at the same viscosity and timestep. Compute Sanitizer passed the moving cut sphere with zero errors.

The equivalent isolated revision 9ea071c passed all 741 Debug CPU and 803 Release CUDA CTest cases. Blender requalified the authored planar immiscible viscosity and three-source Mesh, native attachment, material render and interface-side fractions. Receipts in the cut worktree are out/m4-cut-cpu-final-tests.log, out/m4-cut-cuda-full-tests.log, out/m4-cut-cuda-pure-shear.log, out/m4-cut-cuda-tests.log, out/m4-cut-cuda-memcheck.log and out/m4-cut-blender. The cut tests can be rerun with ctest --test-dir out/build/windows-release-cuda -V -R 'CutViscousSphereDampsShearAndMatchesCpu|FullCellViscosityMatchesCpuWithoutDeviceAllocation'.

The stationary full-face tank now admits fractional free-surface cells after checking that the solid removed no liquid from those quadratures. A 2.65 m fill at 1 m spacing and 10 Pa s matched hydrostatic pressure within 0.03 Pa with particle speeds below 1e-6 m/s. Shifting the wall to cut cells was refused without changing the checkpoint. Debug CPU passed both rollback and hydrostatic fixtures. The real CUDA counterpart and full candidate suites were launched. Moving and genuinely cut solid wall closure, and production adaptive viscosity/composition, remain outside the verified scope.

## c444fe6 complete uniform-candidate receipts and upstream refresh

The fractional full-face tank produced identical CPU/CUDA pressure and position, and particle velocity differed by at most 4.04e-28 m/s. Device allocation and free counts were unchanged during the accepted attempt. The complete candidate passed 742 Debug CPU, 805 clean Release CUDA, and 1331 locked Linux Python tests. Independent Lean F7 and the audited inventory passed. Compute Sanitizer reported zero errors on the real cut sphere and fractional tank. Blender 5.2.3 reran authored immiscible viscosity, three-source emission, Mesh Build/Resume, attach/relocation, corruption refusal, fresh-process material render, and two-sided composition. Both 216-frame capillary drops passed with 4.90524 percent frequency error, damping ratio 0.135843, peak volume drift at most 0.000417287 and all 1080 accepted substeps within the capillary bound per backend.

Receipts are out/m4-fractional-tank-cpu-full.log, out/m4-fractional-tank-cuda-full.log, out/m4-fractional-tank-cuda-focused.log, out/m4-c444fe6-python.log, out/m4-c444fe6-proof/evidence.json, out/m4-c444fe6-memcheck.log, out/m4-c444fe6-blender, and out/m4-c444fe6-drop-cpu and out/m4-c444fe6-drop-cuda. The capillary runner and Blender commands are recorded above. Use out/build/m4-fractional-tank-release for this exact binary pair.

The read-only upstream refresh pinned main at 549fb3f. It includes bounded adaptive family remap, exterior pressure subfaces, geometric face ownership, matched-error pressure timing, CUDA allocation instrumentation and the CLI11 parser. Its design changes are confined to the CLI and language/dependency clauses. The M3 production integration record still requires accepted graded trial topology, physical aperture/wall/source assembly at production scale, transaction/checkpoint publication and real CPU/CUDA adaptive replay. Bounded adaptive pressure remains capped at 4096 cells. M2's quarter-scale swell publishes and replays frames zero and one, then the frame-two Mesh scaffold encloses 1.32813 m³ against 1.76750 m³ and fails the 0.5 percent volume correction. The uniform M4 authored gate was measured independently. The integration preview from c444fe6 to 549fb3f reported no textual conflicts. Receipts are out/m4-upstream-heads.log, out/m4-upstream-fetch.log and out/m4-upstream-integration-preview.log.

## e1352c9 against pinned main 549fb3f

The local current-main candidate includes the Cartesian cut-wall slice. Its pressure centers must remain inside the clipped tank. The normal derivative uses the actual wall distance, and the planar moving-piston path uses the wetted flux and sealed-slot wall velocity from the same frozen pose. The initial stationary cut-wall probe at 0.01 s missed the 0.03 Pa pressure check by 0.002072 Pa. It remains recorded in the isolated cut worktree. The stationary hydrostatic test at 0.02 s passed the same pressure bound. The manufactured wall-traction and backward-Euler work balance passed with a nonzero -4 Pa analytic stress at three wall offsets, two spacings and two timesteps, including 0.01 s. The production moving test checked the 0.75 m wall distance and 15 kg m²/s normal weight, 0.3975 m³/s wetted inflow and corrected free-surface outflow within one percent. The real CUDA test matched CPU and made no device allocations or frees during the attempt.

The integrated candidate passed 751 Debug CPU, 815 clean Release CUDA, and 1331 Linux Python tests. There were no skipped cases. The pinned independent F7 checker passed. Four sanitized real-device viscosity and phase cases reported zero errors. Blender 5.2.3 reran authored viscosity, three-source Bake, Mesh Build/Resume and attachment, and reopened the native material and two-sided composition. Both 216-frame drops passed the 5 percent frequency gate with 4.90524 percent error, damping ratio 0.135843, volume drift below 0.000418, and all 1080 accepted steps within the capillary bound per backend. The integration review verified that the shared piston-flux reducer preserves the adaptive explicit exterior graph incidence.

Build the locked Windows presets with the repository's qualified CPU and CUDA toolchain receipts. The local current-main worktree used conan.lock with CLI11 2.7.2, clang-format 19.1.5, ctest --preset windows-debug-cpu -j 6, and ctest --test-dir out/build/windows-release-cuda --output-on-failure -j 1. The earlier capillary and Blender reproduction commands take this build directory as their binary input. Receipts in this worktree use out/m4-e1352c9-cpu-tests.log, out/m4-e1352c9-cuda-tests.log, out/m4-e1352c9-python.log, out/m4-e1352c9-proof/evidence.json, out/m4-e1352c9-memcheck.log, out/m4-e1352c9-blender, and out/m4-e1352c9-drop-cpu and out/m4-e1352c9-drop-cuda.

## Remaining completion gates

The scheduler's allocation-free host attempt contract remains unmet for positive viscosity. An MSVC Debug CRT-hook probe wrapped Attempt and status destruction on the 27-cell fractional tank. The inviscid control allocated and freed zero blocks. The accepted 10 Pa s step allocated and freed 623 blocks and requested 657753 bytes in aggregate. The probe was on isolated revision dbe846e and is retained as out/m4-positive-viscosity-host-allocations.log in the cut worktree. The measured paths include strain and octant construction, the owning pressure preparation and the coupled GCR and velocity-block work buffers. Reserve their storage at scheduler creation or capacity growth, borrow it through both backends, and rerun the host audit on accepted, retried and rejected steps. The device-allocation receipt remains valid separately.

The native adaptive Simulate dependency requires accepted graded trial ownership, the production physical exterior pressure graph, and transaction/checkpoint publication and replay on CPU and CUDA. Once that path exists, run the existing bounded adaptive remap, pressure-boundary and face-ownership tests as controls, assemble the coarse/fine strain and inertia on the same owned subfaces, and qualify traction and constituent inventory across accepted topology changes and checkpoint replay. M2's wave Mesh failure remains a separate end-to-end dependency for the full wave campaign.

## Isolated pressure history and host scratch follow-up

The pressure-history commits 30c4b54 and dbe846e reuse the checked gauge-shifted initializer. The stopping scale is still measured at zero pressure. Anchored and gauge-shifted converged history accept with zero pressure iterations. The 1e-8 Pa jump preserves its original forcing norm, and stale nonzero history is cleared for exactly zero forcing. The real CUDA check passed two moving-piston frames with absolute target times of 0.01 and 0.02 s. Receipts in the cut worktree use out/m4-coupled-history-green.log, out/m4-coupled-history-signal.log and out/m4-coupled-history-cuda-final.log.

Commit 2cb0cd9 factors the inner velocity solve into checked caller scratch. The Debug CRT audit counted zero allocation or free on successful solve, alias rejection and iteration exhaustion including status destruction. Commits c7d8d16 and 4f47da8 reuse the GCR basis, compensated flux sums and CPU inner buffers across an owning projection. A 64-cell analytic column converged in 67 iterations and exercised two GCR restarts. The accepted fractional tank now uses 99 host allocations and frees and requests 136592 aggregate bytes. The inviscid control still reports zero. The latest remeasurement remains a failing scheduler gate. Receipts use out/m4-bounded-velocity-green.log, out/m4-bounded-velocity-rejection.log, out/m4-bounded-velocity-cuda.log, out/m4-reused-cpu-velocity-tests.log, out/m4-reused-cpu-velocity-cuda.log, out/m4-gcr-restart-final.log and out/m4-reused-cpu-host-allocations.log. The isolated commits have focused evidence and have not been folded into the fully qualified e1352c9 candidate.

The failing audit can be reproduced in a detached Debug worktree at dbe846e. Apply out/m4-positive-viscosity-host-allocations.patch from the cut worktree, build fluidity_native_tests and run ctest --preset windows-debug-cpu -V -R 'PositiveViscosityHostAllocationProbe'. Reverse that same patch after collecting the receipt. The positive case is expected to fail the zero-allocation assertion.

The next host slices are to expose checked bounded pressure-system preparation without an unused multigrid hierarchy, then build strain rows and cut octants in caller spans. Reserve rows, flux sums, the fixed GCR basis and inner velocity storage only for positive-viscosity scenes at creation and transactional growth. Bind the prepared spans and final solution to the scheduler's trial publication. On CUDA, borrow the preallocated host weighted rows and the arena's pinned result rather than allocate an owning result at each inverse. Re-run the accepted, retry and late-rejection CRT tests, with the inviscid and device allocation controls, before closing this gate.

Bounded preparation commits 0534e44, 3e0680e and 90f9776 reuse the checked original pressure assembly without constructing the inviscid hierarchy. The arena rollback and owning-system comparison passed, and the coupled solve now accepts the same pressure view. At 531fdff both scheduler backends borrow their existing grid and reserved pressure arena. The 10 Pa s host audit now counts 69 allocations and frees with 119365 aggregate requested bytes. The inviscid control remains zero. Sixty-two focused Debug CPU and nine mixed CPU/CUDA cases passed, including seven GPU cases. The receipts in the cut worktree use out/m4-coupled-bounded-system-tests.log, out/m4-coupled-borrowed-arena-tests.log, out/m4-coupled-borrowed-arena-cuda.log and out/m4-coupled-borrowed-arena-allocations.log.

The remaining host work is to size and borrow remapped rows, the GCR basis and its candidate/response flux buffers, wall velocities, and the CPU and CUDA inner preparation within the reserved arena. Publish final velocities and pressures from the trial workspace instead of constructing an owning result during the step. Convert the strain and cut-volume setup to caller spans, then run the same CRT audit on accepted and rejected viscous attempts. The verified e1352c9 code and the focused 531fdff development slice remain on separate local branches.

Isolated commit 99af6c1 moved the coupled face and cell work buffers and GCR basis into the reserved arena. Commit ec87c65 added the remapped strain rows and CPU inverse work buffers. The 27-cell fractional tank now counts 41 allocations and frees and 48110 aggregate requested bytes, down from 69 and 119365 bytes at 531fdff. The inviscid control still counts zero. The row change passed 70 focused Debug CPU tests and 30 Release cases, including 14 GPU cases. Independent review checked arena sizing, the 15-row scheduler bound, standalone row extents and result lifetimes. Receipts in the isolated worktree use out/m4-coupled-fixed-buffers-* and out/m4-coupled-rows-*. The scheduler audit still fails its zero-allocation assertion. Coupled row capacity now participates in the general pressure reservation, including inviscid workspaces. Measure that footprint and scope the extra capacity to eligible scenes before final performance qualification.

Isolated commit efa9835 moved pressure history, defects and accepted face velocities to the pressure arena and reduced the same fractional-tank audit to 16 allocations. Commit bb5a4e1 moved material records, cut octants, Cartesian wall distances, pressure face slots and assembled strain rows to that arena. The 10 Pa s and inviscid controls each measured zero host allocations and frees. The permanent Debug CRT test covers acceptance and injected kAccept retry and fatal rejection. Both rejected attempts preserve identical checkpoint bytes. Final focused checks passed 71 Debug CPU and 14 real CUDA tests. Independent review identified that early geometry failure bypassed scratch accounting. The commit records usage on every exit after borrowing scratch. Reproduce with ctest --preset windows-debug-cpu -V -R PositiveViscosityAttemptsAllocateAndFreeNothing. Receipts in the isolated worktree use out/m4-coupled-view-*, out/m4-geometry-scratch-audit.log and out/m4-geometry-final-*. CUDA still owns prepared host rows, diagonals and downloaded velocities in the viscosity inverse. Physical-error statuses also need an allocation audit beyond the injected late failures.

At isolated commit 0730021 the CUDA viscosity inverse forms weighted rows and its diagonal in reserved host scratch, then returns the synchronized pinned velocity span. The coupled solver consumes it before the next launch and copies accepted velocities into its projection span. The Debug CUDA test on SM120 passed with zero CRT allocations and frees and unchanged CUDA ownership counters for the accepted 10 Pa s step and injected late retry and fatal rejection. Reproduce with ctest --test-dir out/build/windows-debug-cuda -V -R CudaLiquidSubstep3D.PositiveViscosityAttemptsAllocateAndFreeNothing. This Debug build uses the locked Debug Conan dependencies and the qualified Windows CUDA toolchain receipt. The same commit passed 61 focused Debug CPU and 16 real Release CUDA cases. Receipts in the isolated worktree use out/m4-cuda-host-scratch-*. The real numerical and geometry failure paths still need allocation auditing, and viscosity-only arena sizing still needs to be scoped to eligible scenes.

Isolated commit fc2c93f selects coupled workspace capacity at scheduler creation and capacity-growth replay from positive scene or constituent viscosity. Standalone inviscid pressure and liquid workspace queries retain their original reservation. The exact-capacity accounting check and forced late positive-viscosity capacity replay passed. The clean focused Debug build passed 40 CPU and four real CUDA cases, including both allocation audits. See out/m4-scoped-workspace-tests.log in the isolated worktree. An earlier broad run at 0730021 omitted the CLI and two auxiliary executables after a target-only clean build. The 19 resulting missing-executable failures are retained in out/m4-cuda-host-scratch-cpu-full.log. The replacement run builds all targets before invoking the full suite.

Isolated commits fe8840e and 77fd993 preserve code-only statuses for bounded viscosity geometry, strain, and solver errors. New CPU and real Debug CUDA audits exercise an actual unsupported Cartesian cut and actual coupled pressure exhaustion with a one-iteration limit. Both allocate and free zero host blocks, preserve committed state, and report the expected pressure-stage disposition. CUDA device ownership counters also remain unchanged. Ninety focused cases passed (73 CPU and 17 real CUDA), then five targeted cases passed after the octant status review correction. The all-target Debug suite at fc2c93f passed 843 of 844 cases. The CUDA wetted-piston overflow test reported three CRT allocations there and on one immediate reproduction. It passed twelve consecutive isolated repetitions on the rebuilt viscosity-failure candidate. No root cause was established. Receipts are out/m4-scoped-workspace-full.log, out/m4-debug-piston-overflow-reproduction.log, out/m4-debug-piston-overflow-final-repeat.log and out/m4-physical-failure-*. All eighteen isolated host-reservation commits are now replayed onto pinned upstream main as local code candidate da91132. Clean complete CPU and Release CUDA suites were started on that candidate. The adaptive explicit exterior wall-flux branch remains present in the integrated pressure workspace.

### Integrated source da91132

The exact integrated source passed 760 Debug CPU and 823 Release CUDA CTest cases from complete clean builds. Release includes the CLI interruption campaign, adaptive controls, 67 GPU-labeled cases and eight liquid acceptance cases. Clean Debug CUDA passed the integrated zero-host-allocation checks for accepted and injected late viscosity outcomes, actual geometry rejection and pressure exhaustion, and the earlier wetted-piston overflow. Upstream's CUDA CRT hook accounts for the attempt thread. The earlier isolated process-wide count is still unexplained. The pinned independent Lean checker and 1331 Linux Python tests also passed.

The 0.031 N/m curved phase fixture supersedes the earlier zero-pressure receipt. Its 24 production crossings have 2.54381 percent RMS curvature error. Against 0.0775 Pa analytic pressure, the core measured 0.0755401 Pa inviscid and 0.0749891 Pa with 0.001 Pa s viscosity. Both pass the existing 25 percent fixture threshold. This measures curved capillary pressure. The design's viscosity traction promotion gates use planar extension and Couette.

Blender 5.2.3 passed authored nonzero immiscible viscosity, the three-source Bake, Mesh Build/Resume, attachment, relocation and corruption refusal. A fresh process rendered 433 dye and 748 oil pixels, with zero dye pixels in the disconnected control. The two frames each had 218 shared faces, 54 seam edges, 25/75 aqueous composition on the water side and pure oil on the other. Receipts use out/m4-da91132-*. The capillary campaigns and Compute Sanitizer were launched separately. The upstream refresh still reported 549fb3f as main. Production adaptive Simulate, its checkpoint replay and the wave surface failure remain open.

The da91132 Compute Sanitizer run passed all five real-device cases and reported zero errors. The completed 216-frame CUDA drop measured 1.9245825497 rad/s against 2.0238577025 rad/s, for 4.90524 percent frequency error. Peak relative volume drift was 0.0004172863 and all 1080 accepted steps were capillary-bounded. Run tools/native/tests/run_capillary_drop.py with the clean Release fluidity_simulate and fluidity_liquid_acceptance_fixture binaries and --backend cuda. Receipts are out/m4-da91132-memcheck.log and out/m4-da91132-drop-cuda. The CPU run was still active. Independent integration review checked adaptive explicit exterior incidence. Curved discontinuous-viscosity traction refinement remains unqualified. The predeclared design traction matrix uses planar extension and Couette.

The integrated CPU drop also completed all 216 frames. Its angular frequency was 1.9245825615 rad/s, with 4.90524 percent relative error, damping ratio 0.1358429, and peak relative volume drift 0.0004172863. All 1080 accepted substeps were capillary-bounded. The command is the same as the CUDA run with --backend cpu. The receipt is out/m4-da91132-drop-cpu.

### Additional bounded traction and remap evidence

Test-only commit 83a77fb shares the predeclared planar extension and moving-wall Couette matrices with the real SM120 arena. Both matrices use resolutions 16 and 32, timesteps 0.005 and 0.0025 s, and viscosity ratios 1, 10 and 1000. The isolated Release CUDA arena passed all 12 extension cases. Its worst normalized normal RMS was 4.35943e-5, normal maximum 5.96813e-5, and vanishing tangential maximum 8.21211e-4. Debug CUDA passed all 12 finite-strip Couette cases with worst normalized normal and tangential errors 1.19307e-6 and 8.65521e-7. Both tests require unchanged device and pinned reservations and frees. The CPU periodic Couette matrix remains the periodic reference. Reproduce with ctest --test-dir out/build/windows-release-cuda -V -R 'CudaMacViscosity3D.CoupledProjectionRecovers'. Isolated receipts use out/m4-arena-cuda-traction-tests.log and out/m4-couette-matrix-tests.log. The integrated CPU matrix passed all six coupled regression cases.

Test-only commit e1e73d2 exercises the bounded topology family entry with a 25/75 clear-water/dye parent and adjacent oil. It checks slope veto against the oil stencil donor, exact constituent amounts through refinement replay and coarsening, retained oil owners and rejection of cross-phase child restriction. All 15 Debug remap and band tests passed. Run ctest --preset windows-debug-cpu --output-on-failure -R 'AdaptiveLiquidRemap3D|LiquidRemap3D|LiquidBandHandoff3D'. Receipt is out/m4-adaptive-constituents-tests.log.

Upstream still reports 549fb3f. Local adaptive-width e458042 and remap-integration 9282854 add bounded policy and handoff evidence. Production qualification requires the M3 scheduler to publish graded trial ownership, the assembled physical subface pressure graph and replayable accepted topology. After that handoff, build coarse/fine strain against the same owned apertures, run the fixed extension and Couette limits, and check constituent totals and accepted/replayed digests on both backends. The separate full wave campaign still requires the M2 frame-two mesh volume correction.

The integrated 83a77fb Release CUDA run passed all five CudaMacViscosity3D tests, including all 24 arena traction cases and their device/pinned ownership checks. The exact receipt is out/m4-83a77fb-cuda-traction-tests.log. All 15 bounded remap and band tests also passed in Release at e1e73d2, with receipt out/m4-adaptive-constituents-release-tests.log. Both follow-ups change tests only. The production files are unchanged from fully qualified uniform candidate da91132.
