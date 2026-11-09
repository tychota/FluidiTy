# Analytic Sphere Capillary Equilibrium Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development with fresh implementers and independent review. Retain RED/GREEN, raw measurements and exact artifact identity in ignored evidence.

**Goal:** Preserve the known static equilibrium of the frozen analytic-sphere capillary scenes on the actual CPU and RTX paths.

**Architecture:** Give the boundary freeze an optional exact outward curvature for an analytic field when the scheduler knows one. Keep the existing sampled curvature for transported and other fields. The positive-sigma analytic scheduler opts into an exact initial pressure only for a checked, anchored uniform rest state. Evaluate the original residual and corrected velocities through existing code.

**Tech Stack:** C++20, MSVC, nvcc explicit SM120, Conan 2, Ninja CMake presets, uv locked qualification and existing native acceptance harness.

**Spec:** docs/superpowers/specs/2026-11-07-fluidity-design.md physical/derived geometry split and capillarity, and docs/superpowers/plans/2026-11-09-real-3d-liquid-acceptance-and-throughput.md Task 5. The scene geometry, timesteps, pressure and speed thresholds remain frozen.

## Evidence and constraints

- Actual CPU fine sphere failed at 16.839115/16.820388 Pa and fine/coarse maximum speed .00254097/.00118795 m/s, preserving both original 24-frame runs.
- Exact-library witness using sampled SDF center-to-zero-set curvature transport reduced pressure error below .13% but retained fine .000489575 > coarse .000316146 m/s. Prescribing analytic 2/R reduced speeds to .08145/.00462 micrometers/s but still failed +1e-8 m/s because first-step PCG started from zero Pa. Exact equilibrium witness retains every particle, never edits a pressure output and checks the actual true residual. An analytic source can supply 2/R as input geometry without approximating it with a stencil.
- No broad curvature redesign, solver tolerances, selected cell/topology, apertures, wall flux, material J, pressure equation, transfer, dynamic drop representation, or acceptance criterion changes.
- Keep static scene user-visible ambient gauge zero. Preserve legacy paths and committed checkpoints. Keep actual CPU/CUDA qualification distinct and do not interrupt existing frozen jobs. No push or merge while any M1 gate is open.

## Task 1 — Carry analytic sphere curvature to its open links

**Files:** native/include/fluidity/host/liquid_boundary_geometry_3d.h, native/include/fluidity/host/liquid_boundary_workspace_3d.h, native/src/liquid_boundary_geometry_3d.cc and native/src/liquid_substep_3d.cc. Use existing boundary/scheduler test seams and no new device ABI unless proven necessary.

**Interface:** Add a trailing default-empty optional exact outward interface curvature in inverse meters to owning and borrowed freeze inputs and update the FrozenLiquidCell3D contract. Supplied values may be finite signed or zero. The scheduler supplies finite positive 2/radius only for positive sigma with a checked LiquidAnalyticSphere3D. Freeze applies it only to cells already identified as having open liquid/air links. The override skips only their derivative halo; adjacent air centers and all corners are still required. Shared FreezeCore requires a finite supplied curvature. When sigma and curvature are both nonzero on an active open air link, their rounded product must be finite and normal before geometry publication. An exact zero curvature is allowed. Existing BuildLinks still checks addition with ambient pressure. Otherwise keep sampled curvature and halo validation. Retain the existing capillary clock.

- [ ] Runtime CPU and actual CUDA RED for analytic coarse/fine exact 16 Pa exterior jump, ordinary sampled field unchanged, zero sigma/no open air no capillary lift, bounded invalid scalar and rollback. Verify one curvature lift through assembly and correction.
- [ ] GREEN using existing shared FreezeCore and scheduler geometry metadata, preserving face/volume data and producing identical CPU/RTX input values.
- [ ] Verify scoped and ordinary boundary cases and legacy scene replay. If the RTX is occupied, perform runtime CPU RED first and retain the exact compiled CUDA RED control/source for later real exclusive runs, as for Task 2. Hardware qualification stays open. Commit independently. Do not claim static speed refinement at this intermediate commit.

## Task 2 — Start an anchored uniform rest problem at its exact constant pressure

**Files:** native/include/fluidity/host/liquid_boundary_workspace_3d.h, native/src/liquid_substep_3d.cc, native/src/mac_pressure_workspace_3d.cc and its shared preparation declaration, native/cuda/liquid_pressure_stage_3d_host.cc, and existing CPU/actual CUDA pressure tests and exact validator inventory where needed.

**Interface:** Add a trailing default-false host-only flag to LiquidPressureProblemView3D. The positive-sigma analytic-sphere scheduler sets it on a local copy immediately before native projection. Both bounded CPU and CUDA host preparation pass it to the shared PrepareBoundedMacPressureUpload3D initial-iterate path. After checked Prepare, eligible systems are nonempty and gauge-free, with every prepared compliance, previous pressure and integrated target flux zero, every predicted normal velocity, actual prescribed wall flux and internal jump zero, and one or more identical finite Dirichlet pressures across the whole problem. Checking all faces is deliberately conservative, including dormant closed faces. The shared code may decline a candidate if original-system arithmetic cannot represent evaluation bounds. Seed with the common pressure only in the initial iterate. Keep default bounded and owning behavior, ordinary warm starts, closed-face output, pressure history, RHS, original residual/correction and publication unchanged. No new device ABI or scratch allocations.

- [ ] Runtime CPU and actual CUDA RED: same exact Dirichlet rest problem returns constant pressure, zero corrected velocity, zero physical defect/residual and zero iterations. Include two resolutions, separate disconnected anchored components with same boundary pressure and signed nonzero pressure, and ordinary nonuniform/source/wall/jump/compliance/gauge/no-boundary controls keeping established behavior. Check both flag values on a tiny positive uniform Dirichlet pressure for which the previous default solve accepts its zero initial iterate. Check exact prepared RHS/absolute-term and physical history equality across flag values. Include implicit wall flux and a two-ended explicit nonzero wall flux, a disconnected gauged component, unequal exterior pressures, and dormant closed-face passthrough.
- [ ] GREEN uses existing preparation and original-system checks. Independent reviewer inspects finite input ranges, equality semantics, anchoring, numerical order, and no false special case.
- [ ] Run and retain all frozen static-sphere scenes through frame 24 on actual Windows CPU and RTX, checkpoints 10/20 and every stable ID, then four locked presets, explicit SM120 device images and isolated Release named-pipe memcheck covering an inventoried bounded pressure case. If RTX is occupied by the archived qualification, perform CPU runtime RED before implementation and retain the exact compiled CUDA RED control and source. Run it and the exact CUDA GREEN with real exclusive device access before declaring hardware qualification. A missing test is an open gate. Re-run relevant regression/replay. Commit implementation and later fixes separately.

## Review focus

- Known analytic field only. Zero sampled trace is not an interface marker, and nonanalytic fields must not receive the analytic shortcut.
- Positive-sigma open links preserve bound and jump once. A nonfinite curvature or unrepresentable active capillary lift must fail without publishing.
- Uniform source-free anchored component assumption must be justified by actual accepted prepared inputs. Check zero prescribed integrated wall flux and internal jump, not just velocities.
- Preserve previous pressure as physical data in compliant cases and leave pinning untouched.
- Benchmarks use the same solved case and verified artifacts, not a fixture-only pressure injection.
