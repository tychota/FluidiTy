# FluidiTy M6 qualification

M6 remains open. This branch starts at upstream main
`549fb3f844841b79d14b7beb9ecfb6db9727c976`. The evidence below applies to
the bounded native operators and fixtures.

## Gate results

| Gate | Evidence | Status |
|---|---|---|
| Q-SOLID uncut elastic work | `MacTransfer3D.UncutElasticWorkBalancesAtTranslatedCorner` uses translated MAC supports that straddle eight blocks, a quadratic velocity field with an analytic gradient, and a positive-determinant Neo-Hookean state. Grid power is −3 W and deformation power is +3 W at the spec's scaled relative/absolute tolerance of 1e-6. The singular-deformation regression rejects an unresolved positive determinant. | Pass for the fixture |
| Explicit cut work | `MacTransfer3D.MaskedElasticWorkReportsDerivativeMismatch` checks the renormalized deposition gradient against 800/529 per second and the complete gradient against 2 per second. It checks dt times the stress contraction of their difference, measured grid stress work, impulse and torque, exclusion of incompatible faces, and unchanged forces when compatible support is empty. | Pass for geometry-derived translating-slab tags and supplied complete ghost fields |
| Blade work and continuous contact | `ElasticBladeContact3D.MovingThinBladeAccountsForOnePhysicalReaction` detects the slab crossing at 31/64 of the step with both sweep endpoints outside the original relative slab interval. It constrains both velocities using their incoming normal speeds. Stored momentum changes by (6, −1.5, 0) N s and the blade does 12 J of work. The reaction moment uses the sweep's certified hit position. | Pass for constant-translation analytic slabs |
| Persistent cut sides and replay | Slab tests generate both authored normal signs. `RigidBody3D.InteractionCheckpointReplaysBodyAndBladeSideAfterRejectedAttempts` serializes an assigned side, rejects malformed checkpoints and conflicting blade attempts, and compares next accepted checkpoint bytes after restart against uninterrupted execution. | Pass for one blade affinity per particle and the standalone interaction checkpoint |
| Isolated rigid integration | `RigidBody3D.AdvancesRotatedAsymmetricInertiaAndOffCenterForce` exercises world force, off-center torque, rotated unequal principal moments, the gyroscopic term, normalized quaternion and the shared pose drift. Separate fixtures check large-origin sweep bounds, invalid-inertia/duration rejection and byte replay. | Pass for isolated principal-axis bodies |
| Production elastic and cutter scene | The cut work fixture derives compatible masks from time-sampled slab geometry. The elastic operator still takes caller-supplied stress and complete velocity samples. The bounded blade path does not implement a mesh cutter, affinity propagation through a full solid, or a production angular sweep. The branch needs elastic P2G/G2P scheduling, integration of the geometric tags with same-side ghosts, accepted deformation/affinity state, and a resolved Cutter role and scene with production replay. | Open |
| Production checkpoint | The size-prefixed FLSI v1 codec stores ordered rigid records, previous pose, assigned particle sides and optional elastic particle records including accepted deformation and MAC transfer state. The solver still needs one accepted checkpoint transaction that includes full material state, liquid pressure history, rigid history and exactly-once reaction bookkeeping. | Open |
| Two-way floating body | Existing production liquid inputs accept prescribed obstacle translation. They need body-driven geometry and return pressure/contact loads to a dynamic body in the same accepted step protocol. | Gated on production liquid–rigid coupling |

## Floating-body entry and exit evidence

1. Supply a resolved dynamic body with stable ID, positive mass and inertia,
   pose, surface motion and a supported swept geometry. Feed the body
   pose and boundary velocity into production liquid pressure, ghost
   completion and continuous contact. Use geometric union or reject
   ambiguous overlapping blocked regions.
2. Integrate pressure traction `−Σ p_f A_blocked,f n_body,f` with moment
   arms about the center. Record pressure, contact and external loads
   separately. Apply contact reactions once. Version the declared one-step
   partitioned lag, add the added-mass diagnostic, and retain a coupled
   solver gate for light bodies if the partitioned update fails.
3. Run a resolved floating-body scene through the production scheduler.
   Freeze the scene's equilibrium and trajectory tolerances before the
   run. Archive draft/force balance, force and torque traces, body and
   liquid momentum budgets, pressure residuals, and time-step sensitivity
   for the ordinary and light-body cases. Check interrupted/restarted
   output against uninterrupted output on the same artifact, including
   a rejected step between accepted steps. Require the light-body evidence
   or the coupled-solver result before marking the gate passed.

## Verification

| Command or executable | Result |
|---|---|
| Windows Debug CPU build and `ctest --preset windows-debug-cpu --parallel 6` | 674/674 passed, including schema conformance |
| Windows Release CUDA `fluidity_cuda_operator_tests.exe` on RTX 5060 Ti, driver 610.88, SM120 | 42/42 passed |
| Release CUDA `ctest --preset windows-release-cuda -R '^CudaLiquidBoundary3D\.'` | 6/6 passed |
| Compute Sanitizer memcheck on `CudaLiquidBoundary3D.*` | 6/6 passed, 0 errors |
| `python -m pytest reference/tests tools/native/tests -q` | 1,191 passed, 1 skipped |
| Tracked native `clang-format --dry-run --Werror` with 19.1.5 and `git diff --check` | Passed |

The CPU build used the local pinned Python/Conan dependencies with an M6
build directory. The locally installed uv shim was below the repository
version floor. Release CUDA used an M6 Conan install from the pinned lock.
The CUDA results qualify existing device operators affected by the shared
contact change. The new elastic and rigid operators are host paths.

## Integration decisions

- M4's integrated local candidate was `codex/m4-current-main` at `86399fd`
  during qualification. M6 uses verified upstream main.
- M5's inspected local tip was `codex/m5-granular-snow` at `e9e5074`.
  It had material laws and the pair operator without a production liquid–rigid scheduler.
- M5 owned the shared `ResolveDistinctBodyContact3D` API with stable
  64-bit body IDs and equal/opposite impulses. M5 can reuse the uncut
  force operator from `3131f9c`. The M6 slab path uses the existing
  prescribed-wall sweep. Both milestones still need the production
  coupling and checkpoint transaction.
- Isolated rigid bodies use positive diagonal moments in an authored
  principal-axis frame. The step evaluates angular acceleration at the
  start and uses the updated world velocities for the drift and sweep.
- Independent review identified large-origin sweep rounding, torque at a
  reconstructed rather than certified hit, unaligned checkpoint bytes and
  unresolved determinant signs. Each issue was reproduced in a failing
  fixture and corrected before the final checks.

On the subsequent slab face-tag change, the rebuilt Windows Debug CPU executable passed all 31 focused transfer, wall-contact, blade and rigid tests. The fixture covers both exact blade faces, interior anchors and invalid sample time.

The subsequent elastic deformation stage uses the shared old/new same-side ghost gather and advances the dimensionless deformation gradient from the true new-time velocity derivative. The fixture poisons velocities on the excluded side, verifies the affine gradient and the left multiplication of F, and rejects a collapsing trial. All 32 focused native tests passed. The production scheduler still has to commit the candidate with the rest of the accepted material state.

Elastic checkpoint replay now feeds the restored particle, side and deformation into the next gather and compares the next record bytes to uninterrupted execution. The new vector is optional so rigid-only v1 records remain readable. The 12 focused transfer and rigid tests and the FLSI schema-conformance check passed.

The elastic acceptance stage now stages deformation, continuous slab contact, PIC transport, APIC fit and physical reaction for every particle before publishing the batch and incrementing the checkpoint ordinal. It starts from supplied MAC histories and assigned sides. The focused fixture checks an accepted state update and unchanged checkpoint bytes after a collapsing trial. A separate regression rejects a committed side that disagrees with the sweep start even when the blade moves away. All 32 focused native tests passed.

The dry translating-slab grid step now builds both MAC histories from compatible particle deposits, adds explicit Neo-Hookean stress forces and publishes through the existing acceptance stage. It discards old donor mass when a face changes side or enters the slab. The test checks uniform transport, restoring response to nonzero stress, byte-identical two-step replay, and unchanged checkpoint bytes after rejection. All 33 focused native tests passed. Integrated cut-work reporting, timestep qualification, and resolved Cutter scene entry remain open.

The dry grid step reports applied grid stress work, deformation work, cut work computed from the exact completed gather samples and compatible deposition derivatives, and work associated with omitted stress forces. The stressed replay fixture measures nonzero cut work and checks the work identity on retained faces. Blade impulses and blade work remain in the physical contact records. Diagnostics must be finite before particle state is published.

The dry elastic step now enforces the spec's continuous Neo-Hookean acoustic limit before deposition, using the maximum absolute row sum of F F-transpose to bound the largest squared stretch. The configured discrete factor defaults to 0.5. Fixtures check the analytical identity and stretched-state speeds, rejection of an uncertified tangent, and byte-preserving rejection of an oversized attempt. All 48 focused native tests passed. Stability and convergence of the discrete factor in the resolved Cutter scene remain open.

A native two-particle axis-aligned moving slab scene now exercises the full dry grid path. One 0.125 s step and two 0.0625 s steps both reached the same endpoint impact with 1 N s normal impulse and 0.5 J blade work, retained the positive side, and measured zero cut work from the undeformed state. Restart at the fine midpoint reproduced the accepted checkpoint bytes and impact work. An exploratory oblique single-particle scene failed because the 2h donor search could not reconstruct a ghost. This is evidence of a remaining support limitation, and the analytic scene does not close general Cutter authoring or discrete stability qualification. All 49 affected tests passed.

The oblique slab regression now passes with an interior impact at fraction 0.8 and 0.78125 J of blade work. A deficient 2h elastic ghost gather retries with 3h same-side, same-component donors. Full-rank local fits retain the 2h neighborhood. All 50 affected native tests passed. General Cutter authoring and production liquid-rigid qualification remain open.

The dry Cutter step can initialize an empty affinity table from the authored starting slab pose using certified projections. The oblique regression covers both signs and shows that overlap or a subsequent failed deposit leaves the initial table empty and the accepted ordinal unchanged. All 50 affected native tests passed. Blender Cutter authoring, general mesh geometry and liquid-rigid qualification remain open.
