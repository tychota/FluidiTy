# Use projected liquid histories for pressure-supported particle gather

> Use superpowers:subagent-driven-development. Require actual RED/GREEN, an independent reviewer and a separate verified commit. This companion correction leaves the frozen physical qualification plan and its outcomes intact.

**Goal**

Prevent force-only face velocities outside the pressure solve from contaminating new-time particle gathering and ghost continuation when the particle is supported by a projected liquid cell.

**Architecture**

The pressure geometry retains its exact frozen cell and incident-face inventories. Derive a bounded new-time gather validity view from the existing final velocity, mass status and new wall tags after successful projection. A particle whose checked post-emission, pre-advection world position at t^n lies in a frozen pressure cell uses that view. All other particles retain the existing checked gather view. Existing same-side continuation supplies missing new samples. Keep old normalized history, published carrier values and validity, complete support, the true derivative-basis L and J update, and pressure RHS/correction unchanged.

**Tech stack**

C++20, existing checked sparse MAC and bounded scheduler, existing explicit CUDA gather, locked Conan 2 and CMake Ninja presets, uv reference tooling.

**Spec**

[FluidiTy design](../specs/2026-11-07-fluidity-design.md) §§5.1, 5.2, 6.1 and [M1 physical acceptance](2026-11-09-real-3d-liquid-acceptance-and-throughput.md) Task 4.

## Fixed evidence and limits

- Exact b2dec6a CPU piston failed independent tight receiving by 26.7498% and lost 2.1415% material volume. Exact pressure residuals pass; 98.9% of material loss is from the initial top layer. Force-only velocities on mass-valid y=.5625 m faces outside the pressure inventory were directly consumed and used as continuation donors in the first three steps. Undoing only their added gravity in a scratch replay isolated a large compression contribution without establishing a whole-trajectory correction.
- Scratch globally nonempty-pressure masking evolved the full frozen CPU trajectory and restored material drift to +0.02686%, but has a mixed disconnected-component gap. Use the containing-cell local rule only. The diagnostic mixed scene preserved four pressure-free particle-step receipts exactly, with 16 pressure-supported particle steps taking the correction and 720 direct-to-continued changes. In the frozen local CPU trajectory all 245,760 step starts had pressure support, all 240 steps settled, and all 246,784 emitted trajectory rows matched the global experiment byte for byte. These are bounded diagnostic results, with actual RTX qualification pending.
- The scratch trajectory does not qualify the piston. Fixed-radius reconstruction still reports 8.24–8.69% receiving deficit and independent wall versus solver flux differs by 1.73%. The exact determinant-one affine control also shows 8.34–8.38% fixed-radius representation deficit. Preserve both constraints and the raw official failure. A material mesh endpoint is a supplemental diagnostic, and its saturation/swept requirements are open.
- This change does not claim to fix wall flux, surface representation, full physical acceptance or throughput. Do not change frozen scene resolution, pressure equations, target thresholds, fixed-radius field, source/compression, frame dt policy, volume estimate, device configuration or proof claims. No J clamp, pressure-divergence substitution, force undo, CPU fallback, unsupported compiler override or driver/firewall policy change.

## Task 1 — Separate projected new gather eligibility from carrier mass status

**Files**

Modify native/src/liquid_substep_3d.cc and its staging/reservation files for a bounded scratch view. Reuse checked existing signed lattice and pressure cell ordering; use a private helper if a local checked containing-cell computation is needed. Extend native/tests/test_liquid_substep_3d.cc and native/tests/test_cuda_liquid_substep_3d.cc with independent native and actual GPU fixtures. Touch public header contracts only where this semantic separation needs to be stated.

- [ ] Reproduce RED on the real CPU scheduler and actual RTX using a bounded pressure-active free-surface fixture with positive-mass faces beyond the pressure solve. Observe the old normalized, force, pressure membership and new gather statuses and the resulting L/J without emulating GPU. Fail the specific regression when force-only new samples remain direct/donors. Preserve a known exact zero-mass case and first-step pressure rows.
- [ ] Add a nonempty global pressure solve containing a disjoint unpressurized particle cloud. Require exact baseline old/new gather fields, diagnostics, positions and J for unsupported particles. Include globally empty pressure, pressure-active and no-pressure free-fall, no-donor status, same-side wall histories and an analytic full-support non-affine discrete divergence-free control that has nonzero true particle trace.
- [ ] Certify half-open signed containing cells [index,index+1) from the particle position after emission and before advection at t^n. Each index can use the full signed int32 range, including int32 max when its high plane at max+1 is certifiable. Reuse compensated plane comparisons without inheriting tank/halo bounds. The actual frozen cells are queried in (z,y,x) order. With nonempty pressure, an uncertifiable containing cell returns OutOfRange before publication. Cover negative/exact boundary, translated non-dyadic grid, both signed index limits and unrepresentable cases. Skip containing-cell work when pressure is empty.
- [ ] Reserve a face-capacity new-gather validity array during staging creation and checked growth. Include it in reservation accounting and resize it during transfer preparation. Populate it only after successful projection and after sampling new-time tags. Keep every frozen pressure-incident face, structurally ineligible wall face and kLowMass status unchanged. Mark only otherwise usable pressure-exterior faces missing. Match the established t^n+1 tags and side semantics, do not alter either wall array or raw published face validity. Pass the view only for particles with certified pressure support. Unmodified histories and per-particle scalar order survive no-pressure, retry, rollback and checkpoints. Do not retain frozen or gather spans across topology replacement, capacity replay, owner movement or the next gather bank reuse.
- [ ] Exercise actual CPU and RTX GREEN with independently expected complete-basis PIC, retained velocity delta, all nine entries of L, fallback flags and the true-gradient exponential J update on frozen inputs. Check old/new wall poses and side tags, no per-attempt allocations, unchanged low-mass and native no-donor failures, error atomicity, retries, replay and exact state/point reproducibility on the same corrected artifact. Confirm zero direct new donors outside projection on the affected supported particles. Ensure that all controls contribute to the test exit status.
- [ ] Run the four locked Windows presets, isolated named-pipe Release Compute Sanitizer, actual RTX 5060 Ti CC12.0 explicit 120-real and 120-virtual image and portable code review. Report actual bounded material/physical observations and fixed gates still open. Commit separately and independently review before new long exact physical runs.

## Review focus

- New eligibility must not change pre-force old PIC/FLIP history or suppress a genuine no-pressure body acceleration.
- A global pressure solve is not local support for a disconnected particle. Exact signed containing cell, pressure membership, and time level matter.
- Keep no-donor native failure, low-mass fallback and held wall normals. No artificial velocity or capillary-step change.
- Store new gather validity independently from immutable/committed carrier status and give it bounded lifetime, copy isolation, memory accounting and rollback.
- Particle trace L and material volume are not synonyms for pressure-cell residual, wall flux or reconstructed receiving volume.
