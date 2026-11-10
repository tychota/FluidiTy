# Real 3D Liquid Acceptance and Throughput Implementation Plan

> Use superpowers:subagent-driven-development for execution. Each task needs a fresh implementer, independent review and a separately verified commit. Retain RED and GREEN evidence for behavioral changes.

**Goal**

Qualify the uniform 3D liquid solver on real CPU and CUDA scenes, with frozen physical targets, independently measured results, deterministic replay and measured end-to-end throughput.

**Architecture**

A test-only scene author and verified cache reader use the public Simulate CLI and versioned records. The numerical solver supplies bounded tank walls and measurement receipts where the existing interfaces are insufficient. Independent Python analysis computes physical targets and evaluates published data. A failed target remains an open gate and its raw result stays available.

**Tech stack**

C++20, CUDA C++20, CMake and Ninja, Conan 2, Python 3.13 with uv, FlatBuffers and BLAKE3. Use the pinned independent Lean toolchain to reverify the existing proof inventory.

**Spec**

[Canonical FluidiTy design](../specs/2026-11-07-fluidity-design.md) sections 2.3, 5.1, 6.1, 6.3, 11, 12, 14 and 15, [M1 roadmap](2026-11-08-m1-uniform-3d-liquid-roadmap.md), and [formal contracts](../../references/formal-contracts.md).

## Global constraints

- Start at origin/main `f90737400a46cdf8023190f19e0c16b33ec494b1`. PR #13 tested `c400a8ad3e7b53e226137413adc7929f68dabbb4` and merged tree `fcae33a728f228b4cb51afe36bc3a151174e6771` exactly. All six hosted CPU jobs passed.
- Follow AGENTS.md for clear domain names, documented API contracts, readable guards and useful numerical comments. Run clang-format 19.1.5 on tracked hand-written native sources.
- Keep the bounded uniform single-liquid, zero-compliance baseline. No Blender, AMR, band ownership, mesh input, production fusion, new materials or published throughput promises.
- Keep existing CPU/CUDA, schema and v1 fixture contracts. Additive schema changes must pass flatc conformance and preserve older canonical bytes, explicit defaults and fingerprints. Scene authoring uses exact executable and initial-state bytes.
- Use actual hardware with explicit `120-real;120-virtual`, precise existing FP flags, and the Windows named-pipe sanitizer setup. No fallback, skip credit, unsupported compiler override or firewall changes. Record actual tool and dependency revisions.
- All thresholds below are declared before scene execution. Retain exploratory failures and their physical causes. Any fixture correction must preserve the physical target and have a separate documented justification and independent review. Never tune a target to make an observed result pass.
- Keep machine paths, raw logs, cache payloads, generated files and downloaded artifacts ignored under `out/m1/acceptance`. Keep the progress ledger and reviews in `.superpowers/sdd/2026-11-09-real-3d-liquid-acceptance-and-throughput`. Tracked reports contain portable procedures and measured summaries.

## Scene authority and measured quantities

The qualification uses the existing public CPU/CUDA execution path. The independent evaluator reads accepted manifests, points and retained checkpoints. It never uses a test-only physics substitution. Use the existing analytic sphere or particle-sphere union as appropriate. A bounded open-top tank with one translating wall and a horizontal analytic liquid surface covers the controlled tank fixtures. General mesh obstacles belong to M1-B.

| Fixture | Frozen physical setup | Required measurement |
| --- | --- | --- |
| Free fall | One particle at (2.5, 2.5, 2.5) m, h=1 m, open analytic sphere centered at the initial point with radius 1.1 m, ρ=1000 kg/m³, σ=0, g=(0, −9.81, 0) m/s², no obstacle or source. Maximum dt 0.01 s, frames 0–6 at 24 fps. Energy blend 0, 0.6 and 1. | Each accepted frame has acceleration error ≤0.1% of g using stored velocity differences and actual accepted times. Check transverse velocity against 1e-10 m/s and report advecting displacement separately. The constant-field algebra checks all e∈[0,1] with consistent old/new history. |
| Hydrostatic column | Open-top tank x,z∈[−0.25,0.25] m, bottom y=0 m, analytic horizontal surface y=1 m, h=0.125 m, g=9.81 m/s², ρ=1000 kg/m³, σ=0. Deterministic center-seeded particles, one per filled cell, zero initial velocity, V0=h³ and J=1. Run to frame 240. | Maximum particle speed at frame 48 is strictly <1e-3 m/s. Report cell pressure versus ρg(1−y), ambient gauge and depth-band errors. Bound accounted volume drift at every frame through 10 s by 1%. |
| Piston | Tank x∈[−0.5,0.5] m, z∈[−0.25,0.25] m, floor y=0 m, initial free surface y=0.5 m, h=0.0625 m, ρ=1000 kg/m³, σ=0, g=9.81 m/s². The low-x wall moves into fluid at 0.015625 m/s for 2 s. The particle sphere union follows accepted particles. Repeat with maximum dt 0.005 s for temporal refinement. | Compare independently reconstructed displaced volume, independently integrated physical piston flux, and actual signed solver wall flux. Each physical relative mismatch ≤1%, denominator floor 1e-8 m³. Verify sealed incompatible rejection separately using native pressure/scheduler fixtures. |
| Static sphere | R=0.125 m, ρ=1000 kg/m³, σ=1 N/m, no gravity, ambient gauge zero. Use analytic distance with h=R/3 and R/5 and deterministic lattice centers inside the sphere. Total reference volume is 4πR³/3. Observe frames 1–24 with pressure at retained checkpoints and maximum speed from point frames. | Median pressure for liquid cells within R/2 differs from 2σ/R by ≤10% at coarse resolution and ≤5% at fine resolution. Report maximum error and require fine maximum parasitic speed no greater than coarse plus 1e-8 m/s, and fine speed ≤0.05√(σ/(ρR)). |
| Oscillating drop | Same R, ρ and σ, ρ_out=0, h=R/5. Volume-preserving l=2 seed with y semiaxis 1.03R and x,z semiaxes R/√1.03, zero initial velocity, transported particle field. Run frames 0–216, maximum dt 0.01 s and the capillary bound. | Measure centered volume-weighted quadrupole. Fit interpolated same-sign extrema over [0.25,9] s. Require at least two periods, finite positive frequency and a measured damping ratio ≤0.20. Frequency is within ±5% of √(8σ/(ρR³)). Report frequency and fit residual, and fail if the signal has insufficient extrema. |
| Energy-blend dam break | Open-top tank x∈[0,2] m, z∈[0,0.25] m, floor y=0 m, initial block x∈[0,0.5], y∈[0,0.5], z∈[0,0.25] m, h=0.0625 m, ρ=1000 kg/m³, g=9.81 m/s², σ=0. Energy blend 0, 0.6 and 1. Frames 0–48. | Compare accepted positions, volume, kinetic energy, 95th percentile front position and peak speed at frames 0, 12, 24 and 48. Require a measured endpoint difference above 0.01 m in front position or 1% of initial potential energy in kinetic energy at a shared time. Report curves without assuming monotonic response. |

Use world grid origin (0,0,0) m, lexicographic (z,y,x) center seeds, strictly inside the sphere or box, zero initial velocity and slopes, J=1, IDs from 1, affine enabled, PIC uninitialized, seed zero, and mass ρV0. Use V0=h³ for rectangular liquid columns, equal V0=(4πR³/3)/particle_count for the spheres, and V0=1 m³ for the single free-fall particle. The static and dynamic spheres are centered at (0,0,0) m. The dynamic drop starts from the unperturbed center seeds and applies the declared determinant-one transform. Use the existing resolved defaults for quantization, friction, pressure budget and mass threshold. All jobs use 24 fps and maximum dt 0.01 s except the stated piston refinement. Validate actual particle count, 128-block ceiling, 4,096 pressure cells and 24,576 owned faces in the preflight. The piston has 1,024 seeds. If an extent fails, retain its receipt and explicitly review any bounded capacity adjustment before a run. Keep the physical resolution and targets fixed.

For piston and dam break, use the transported sphere union with fixed radius 0.9h. Evaluate the piston independently from the start-of-step particle centers, the wall pose and exact accepted dt. For each horizontal (x,z) coordinate compute the sphere-intersection intervals in y, clip to the tank and the reference surface and merge overlapping intervals. Integrate interval lengths with nested adaptive quadrature, seeded at sphere support breakpoints. At the moving piston plane this becomes one integration in z. Check the reported spatial error and a tighter repeat against 0.25% of the measured flux. The independent planar control at a 0.2580645h rise must detect the 3.125% error that coincident 8/16 midpoint grids miss.

Compute receiving volume as change in occupied volume above the initial surface plus swept wetted volume above that surface. Record initial/final void below the reference plane and swept dry wall volume below it. Require their summed absolute contribution ≤0.25% of the full flux before using the saturated receiving-region formula. Independently compare the full wetted swept volume and solver signed piston flux. Both use start-of-step geometry and exact accepted durations. Require the two maximum-dt configurations to agree on the displaced-volume and flux estimates within 0.25% of the primary flux. A planar oracle with the declared 0.03125 m stroke gives 0.0078125 m³ of swept volume below the initial plane.

The oscillating-drop radius is the declared physical R. The particle sphere-union support radius is a separate discretization parameter. Before either physical run, use the independently clipped volume estimator and deterministic bisection on [0.5h,0.9h] to match the transformed seed's union volume to 4πR³/3 within 0.5%. Tighten integration to 0.25% volume and verify the initial l=2 quadrupole sign and size against the analytic ellipsoid within 10%. Freeze the selected support radius as an identical resolved binary value for both backends. Keep the l=2 frequency target independent of the support fit. If initialization cannot meet these checks, qualify the reconstruction with an explicit separate fix before the dynamics.

Measure particle volume as ∑V0J using committed state and the equivalent volume encoded in point radius. Keep mass, center-selected pressure-cell volume, continuous reconstructed volume, swept boundary volume, source volume and compression separate. All declared scenes have zero source and zero physical compression. For transported points use stable-ID overlap to track sphere-connected components across frames and report losses below and above h³/8. Check complete ordered point ID inventories and each ID’s 4πr³/3 against V0J at retained checkpoints. Verify finite values and uniqueness. After frame 240 hydrostatic checkpoints are 230/240. After frame 24 static sphere checkpoints are 10/20. Read speed at hydrostatic frame 48 from verified FLPT.

Use the existing pressure gates ‖b−Ap‖₂ ≤ 2e-6 m³/s + 2e-6‖b‖₂, maximum |r|/V and physical constraint defect ≤ 2e-5 s⁻¹, θ≥0.05 and at most two pressure dt retries. For σ>0, each accepted dt must be no greater than √(ρh³/(4πσ)). Do not equate an operator fixture with a scene result.

## Review focus

- A cache with a missing frame or corrupt checkpoint could appear to have a good final value. Task 1 verifies every ordinal and rejects gaps, nonfinite records and wrong artifact identity.
- A moving piston near an immobile floor or side could misapply piston velocity on a fixed wall. Task 2 covers a fractional piston cut, fixed faces, corner contact and endpoint limits on CPU and CUDA.
- A retry can discard an accepted-looking substep. Task 3B checks that accepted metrics and geometry witnesses roll back with the abandoned schedule, while a separate attempt ledger still counts actual work.
- Volume error can hide particle loss or moving-region bookkeeping. Tasks 1 and 4 include lost-ID, clipped-volume and independent flux controls.
- A wrong device, truncated receipt or missing performance sample can falsely pass. Tasks 6 and 7 reject partial inventories, nonfinite timers, fabricated device results and memory budget violations.

## Task 1 — Build the verified scene harness and qualify free fall

**Files**

- Create `native/tests/liquid_acceptance_fixture.cc`, `tools/native/liquid_acceptance_metrics.py`, `tools/native/tests/run_liquid_acceptance.py` and `tools/native/tests/test_liquid_acceptance_metrics.py`.
- Modify `cmake/SimulateCli.cmake`. Keep the longer acceptance label separate from the fast CLI and reliability inventories.

**Interfaces**

- `fluidity_liquid_acceptance_fixture write-scene CASE CPU_OR_CUDA SOLVER SCENE` writes a resolved FLRS and SCENE.flst with actual BLAKE3 identities. The three free-fall cases are `free-fall-pic`, `free-fall-default` and `free-fall-retained`.
- `fluidity_liquid_acceptance_fixture read-frames CACHE EXPECTED_FINGERPRINT FIRST LAST` streams one verified JSON object per frame with clock, identity and ordered point records. `read-checkpoint CACHE EXPECTED_FINGERPRINT MAXIMUM_FRAME` returns checked retained particle and pressure state and its ordinal, without modifying the cache. The author emits expected fingerprint, execution ID and seed digest from the exact files it wrote. Require fresh output paths and current successful child receipts. Reconcile their ordinals, times and identities with the manifests.
- Python `evaluate_free_fall(frame_samples)` returns acceleration and transverse errors. `run_liquid_acceptance.py --executable PATH --fixture PATH --backend cpu|cuda --case CASE --output DIRECTORY` launches the actual CLI and retains stdout, stderr and measurements. Error and receipt paths must have explicit, descriptive names.

- [ ] Write failing Python tests for a missing frame, wrong artifact/backend, duplicate ID, NaN and an acceleration deviation above 0.1%. Add real subprocess RED for the fixture commands and for malformed scene selection.
- [ ] Implement the strict fixture commands and independent evaluator. Test the linear-in-e uniform history argument by checking the two affine endpoint values and the known midpoint, with negative control for missing history.
- [ ] Run `uv run --project reference --locked --group native python -m pytest tools/native/tests/test_liquid_acceptance_metrics.py` and the three real CPU and CUDA scenes. Require all expected frames and architecture 1200 for CUDA.
- [ ] Commit `test: measure real uniform 3D liquid free fall` after review.

## Task 2 — Qualify bounded tank geometry and motion

**Files**

- Modify `native/include/fluidity/host/liquid_boundary_geometry_3d.h`, `native/include/fluidity/host/liquid_substep_3d.h`, `native/include/fluidity/base/liquid_boundary_math.h`, the shared boundary geometry and contact math headers, host staging and geometry adapters, and CUDA boundary, geometry staging and contact adapters.
- Modify `schemas/simulate/resolved_scene.fbs` and `native/src/liquid_resolved_scene_3d.cc`. Preserve the checked-in v1 baseline. Add focused native geometry, contact, codec and real CUDA tests next to their existing suites. Update the exact mandatory test/receipt and Release memcheck inventories in `tools/native/validate.py` and its negative controls.

**Interfaces**

- `AxisAlignedLiquidTank3D` has world low/high x, world floor y, world low/high z and low-x wall velocity in SI units. Other walls are fixed and the top is open. Append a typed obstacle variant and optional tank table with fixed field IDs.
- `LiquidHorizontalPlaneSurface3D` has world surface height y and distance `world_y − surface_height`. Append a typed field variant and optional plane table. Keep the older variant tags, schema defaults, and encoded golden bytes.
- Stage the plane in a finite floor-to-surface box clipped in x and z by the tank. Add the exact pressure-face-owner blocks and required exterior air and derivative samples independently of the particle positions. Use signed lattice bounds and check each extent before writing.
- Frozen geometry clips cells and faces against the same five world planes on host and device. Carry a piston-only blocked area or already integrated signed piston flux separately from the fixed-wall blocked fraction. Use exactly that datum in pressure assembly and the matching correction/defect. For ghost and contact, choose wall normal and velocity from the active constraint at the sample or contact, with deterministic earliest time and fixed axis order for tied orthogonal zero-friction constraints. Tank qualification accepts zero friction and checks finite positive width throughout each step.

- [ ] Write native RED for a half-cell piston cut with analytically known volume and face apertures, fixed floor/side velocity zero, single and tied contacts for both velocity fields, bad tank extent and step crossing a fixed wall. Cover high and low oriented faces. Write codec RED for round-trip plus missing variant fields, and actual CUDA RED for matched tank geometry/contact.
- [ ] Implement bounded allocation-free clipping in existing checked paths. Reuse the same start and end wall poses for pressure, ghost construction and contact. Use literal axis-aligned planes so corner cases remain independently verifiable. Preserve every existing halfspace, slab and sphere behavior.
- [ ] Run relevant Debug and Release CPU tests, schema conformance, matched real CUDA tests, image checks and named-pipe memcheck. Require both new modes to decode through the public CLI and reject malformed controls before writes.
- [ ] Commit `feat: support bounded open tanks for liquid qualification` after review.

## Task 3A — Deduplicate and batch bounded field samples

**Files**

- Modify `native/src/liquid_substep_3d_staging.cc`, `native/include/fluidity/host/liquid_substep_backend_3d.h`, and the existing CPU and CUDA dispatch and geometry staging adapters. Use the span overload of `SampleLiquidField3D` in `native/cuda/liquid_geometry_stages_3d.h`. Extend focused geometry/arena tests and exact validation inventories as needed.

**Interfaces**

- Collect, order and deduplicate exact center and corner lattice keys before field evaluation. Provide one bounded typed batch accepting `std::span<const Vec3> world_sample_positions` and writing `std::span<double> signed_distances` in the same order. Validate extents and scratch reservations before access, and publish no partial samples. Reuse bounded buffers and existing CUDA fences.
- The existing scalar route issues about 21,000 fenced field calls for roughly 600 initial dam-break candidate cells. Retain actual startup launch count, field-sampling and wall times from the short fixed fixture. The batch count follows buffer chunk capacity, independently of repeated sample keys.

- [ ] Write RED for duplicated requests, mismatched output extent, field parity and bounded launch/fence counts on real CUDA. Include mixed signed block boundaries and an empty batch.
- [ ] Implement deduplication and batching with the same coordinate and sample order. Require exact CPU old/new equivalence and the existing bounded CPU/CUDA tolerance, unchanged accepted state on failure and no physical-attempt allocation.
- [ ] Run focused preset and real GPU/sanitizer gates. Use measured startup service to choose an explicit standalone long-job timeout before the 500-frame runs.
- [ ] Commit `perf: batch bounded liquid field sampling` after review.

## Task 3B — Report accepted numerical work and measured resource usage

**Files**

- Modify `native/include/fluidity/host/liquid_substep_3d.h`, scheduler and staging sources, `native/cli/simulate_execution.cc` and existing CUDA arena metrics and snapshot paths. Extend the existing acceptance evaluator and validator tests.

**Interfaces**

- Add typed committed-step metrics for pressure iteration count, true residual, RHS norm, volume-scaled residual and defect, accepted dt, capillary bound, signed piston wall flux in m³/s and center-selected pressure-cell volume in m³. Use checked pressure inputs and frozen faces. Emit a distinct attempt ledger with retry/replay outcomes and work time. The existing last-frame diagnostics cannot stand in for job totals.
- Add opt-in `--trace-geometry` for the piston measurement only. Attach ordered accepted particle IDs/world positions and the accepted clock to trace records, and invalidate the records of a rolled-back frame. The runner uses the prior accepted positions or the authenticated initial FLST for the step-start geometry. Require exact pose and accepted dt in the corresponding receipt. Run throughput without this detailed geometry stream.
- Add per-job measurement receipts to Simulate with cold setup, emit/activation and field sampling, deposit and normalization, forces/geometry/pressure, gather/contact/acceptance, snapshot/cache publication, drain and total wall time in seconds. Use monotonic host measurements for service and actual CUDA events collected at existing fences. Report the field phase outside the previous CUDA timing scope. Preserve the numerical call order.
- Reuse the arena ledger to record allocated reservations and logical live extents across both old and new arenas and active snapshots. Evaluate aggregate peak device reservation against min(0.70 × device total, 0.80 × device free at job start) before allocation and at growth. Track opaque runtime/graph overhead and observed free-device changes separately from attributable reservations, and report physical state, scratch and output bytes.

- [ ] Write RED for metrics lost or duplicated after a capacity rollback, invalid or missing pressure/phase fields, an aggregate allocation budget exceeded by old plus replacement owners, and a fabricated CUDA receipt.
- [ ] Implement metrics across the existing committed observer and rollback boundary. Keep optional measurement work outside physical attempt allocations and retain the no-allocation controls. On rejected work retain actual diagnostic time/count without publishing provisional metrics.
- [ ] Run transaction, cache, pressure and GPU tests on all four presets, including replay equality with measurements and the existing native bounded real device growth path. The public default scene leaves growth disabled.
- [ ] Commit `feat: report accepted liquid work and aggregate arena usage` after review.

## Task 4 — Qualify hydrostatic volume and piston displacement

**Files**

- Extend the acceptance fixture, independent metrics and runner from Task 1. Add portable scene parameters and physical target tests in `tools/native/tests/test_liquid_acceptance_metrics.py`.

**Interfaces**

- Add cases `hydrostatic-column` and `open-tank-piston` using Task 2 geometry and Task 3B observed values.
- Add Python `evaluate_hydrostatic_column(frame_samples, checkpoint_samples)`, `evaluate_volume_budget(frame_samples, expected_flux_samples)` and `evaluate_piston_displacement(frame_samples, accepted_step_samples)`. Use the fixed setup, formulas and thresholds above.

- [ ] Write RED for wrong depth/time, speed exactly at the strict limit, hidden missing IDs with unchanged total volume, unaccounted source/compression, and a wrong piston orientation or aperture. Exercise the analytic thin-rise alias control and signed swept-region estimator. Reuse the native CPU sealed scheduler rejection without retry and the independent real CUDA sealed pressure rejection as separate controls.
- [ ] Run real CPU and CUDA hydrostatic and piston scenes through the public CLI. Record all outcomes against the frozen thresholds. Diagnose physical failures with independently checked initial/boundary geometry and pressure receipts. Make a separately reviewed regression fix before accepting a previously failing target.
- [ ] Require all observations through frame 240 for volume, frame 48 for speed, and frame 48 for piston. Preserve GPU architecture and actual finite measurements. Keep each correction and result reproducible.
- [ ] Commit `test: qualify hydrostatic volume and piston flux` after review.

## Task 5 — Qualify capillarity and the energy-blend dam break

**Files**

- Extend the existing acceptance fixture, independent metrics, runner and their tests. Write measured portable results and selected point snapshots under the ignored acceptance output root.

**Interfaces**

- Add `static-sphere-coarse`, `static-sphere-fine`, `oscillating-drop-l2` and three `dam-break-*` cases.
- Python `evaluate_static_sphere`, `evaluate_drop_frequency` and `compare_dam_break_energy_blends` use the specified modes, sampling windows, geometry and tolerances. For the frequency fitter use deterministic local parabolic peak interpolation and median same-sign periods. The damping ratio is δ/√(4π²+δ²) for log amplitude decrement δ. Use positive finite amplitude magnitudes and at least two complete periods from same-sign extrema, including negative peaks. Compare measured angular frequency in rad/s and report the damping-corrected frequency separately.

- [ ] Write RED with an independent sampled damped sinusoid, wrong frequency beyond ±5%, insufficient extrema, a one-frame velocity spike, and capillary time-step violation. Add endpoint controls for duplicated dam-break data and incompatible geometry.
- [ ] Run every scene on CPU and real CUDA. Verify pressure, parasitic current convergence, actual dynamic frequency, the capillary step bound, and real energy-blend differences at shared frames.
- [ ] Retain a look-book of the fixed frames with consistent axes/units and the kinetic energy/front/volume measurements. The analysis only consumes verified native output. Keep incomplete or failed physical results distinct from approved operator gates.
- [ ] Commit `test: qualify capillary scenes and energy-blend behavior` after review.

## Task 6 — Measure default-scene reliability and end-to-end throughput

**Files**

- Extend `tools/native/tests/run_liquid_acceptance.py` and its tests with exact workload inventory. Add `docs/native/m1-liquid-acceptance.md` for the final portable procedures and results.

**Interfaces**

- The default workload is the transported tank configuration from the dam-break default energy case with the same initial block and no further source, σ=0, e=0.6. Run 500 completed frames each on Windows Release CPU and CUDA. The two separately fresh runs count 1,000 nonzero committed default-scene frames. Frame zero, previews, failed or repeated frames do not count.
- The named measured workloads are the default workload and the fixed fine-resolution dynamic drop. Report cold, steady and drain wall intervals, frames/substeps and particle updates per second, stage and snapshot times, pressure iterations, peak reserved and live bytes and total output bytes per workload/backend. Report the 32 s/frame candidate reference allocation only in its declared scope.

- [ ] Write RED for a shortened or duplicate workload, invalid or skipped device, warning, unresolved retry, lost output, nonfinite data and understated aggregate GPU memory. Verify interrupted same-artifact recovery on the measured default workload as well as the retained 50-trial regression campaigns.
- [ ] Run the two 500-frame jobs with strict inventory, measure the fixed workloads and preserve raw receipts. Track warnings and failure outcomes separately. Apply the ≤1% volume budget separately through the first 240 frames of the transported default workload and report its component inventory.
- [ ] Verify local four-preset locked suite, standalone tests, both existing 50-trial campaigns, Release named-pipe Compute Sanitizer, image audit and exact proof archive. Rerun all declared scene/backend pairs on the final exact release artifacts. Record observed limits and any still-open physical target.
- [ ] Commit `test: qualify uniform liquid reliability and throughput` after review.

## Task 7 — Verify and integrate the exact slice

- [ ] Check every measurable M1 exit gate against the approved roadmap. Keep scene, numerical, formal and platform evidence distinct. If a physical target remains open, document it and continue the in-scope fix cycle before claiming M1 complete.
- [ ] Obtain a fresh whole-branch review with descriptive naming, API comments and conditional/readability review. Fix important findings separately and requalify the exact clean candidate.
- [ ] Publish a PR with exact path-free results. Verify the six hosted CPU jobs on the tested head, then merge the authorized green PR and fast-forward clean primary main. Preserve worktree evidence and both branches.
- [ ] Continue to M1-B only after the required M1 scene and throughput evidence. Run the separate M1c code/spec and architecture audit after M1-B and before M2.
