# M2 Surfaces and Wave Effects Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal** — Qualify a reproducible geometric surface, then deliver tracked surfaces and wave effects through the Blender bake workflow.

**Architecture** — Separate read-only fixed-cache experiments from native product stages. Simulation owns accepted motion, occupancy and source events, Mesh owns persistent geometry and lineage, Whitewater and Volumes publish independent products, and Blender owns the look.

**Tech Stack** — Locked Python 3.13 / NumPy / SciPy for reference experiments, C++20 / CUDA / FlatBuffers for native stages, Blender 5.2.3 LTS and qualified native playback carriers.

**Spec** — [FluidiTy design](../specs/2026-11-07-fluidity-design.md), especially §§3, 9–12, 14 and 15. [Neural evidence](../../references/neural-surface-evidence.md). [CUDA measurement](../../native/cuda-optimization.md).

## Global Constraints

- Keep M3 adaptivity, resampling and simulation narrow-band changes out of M2.
- Never publish a partial frame. Preserve upstream bytes, fingerprint resolved settings and upstream content, and verify deterministic same-artifact replay.
- Measure an equal-error workload. Pin the revision, executable, fixtures, device and executed architecture. Separate CPU process RSS, live GPU bytes, reserved arenas and disk output. Record median, p95 and max frame latency.
- Use Blender 5.2.3 LTS build `e13b728c6880` for inspection. Preserve cache-only native rendering, 24 fps, SI units, relative paths, and stable attribute names in English and French.
- Keep experiments out of runtime dependencies. No neural model or engine promotion without geometry, determinism, resource and license evidence.
- Require Q-SURF-V ≤0.5% volume error per resolved component against Simulate's target. Require Q-SURF-N area-weighted RMS curvature ≤min(0.1/h, 3 m⁻¹) in the declared rest-tank interior.
- Require Q-GRID ≤2% quality variation on subcell translations, with an absolute error floor and rotation controls. Require Q-IDS ≥90% eligible survival in drop-to-pool outside justified repair and ≥95% in calm motion.
- Require Q-CAVITY to retain genuine enclosed air without a false internal shell. Require Q-WW integrated source difference ≤5% on a frozen field and ≤20% on a scene under h→h/2, with fixed physical source spacing.
- Keep zero default warnings, 50 interruption trials, 1,000 cumulative clean frames, and ≤1% unexplained live-memory growth from frame 10 to 240 as separate reliability gates.

## Approach and entry

Use a downstream geometric reference with the checked native M1 reader. It can run without changing the solver or waiting for Blender. A solver-first implementation would provide occupancy and carrier history sooner but would couple the reference to two unqualified interfaces. A CNN-first implementation would confound reconstruction, missing support and model cost. The geometric route lets each failure be measured independently.

The fixed input is the M1 CUDA `oscillating-drop-l2` publication at `3e0526edb9adda66b4be965ab469d74f5042c990`. The M1 closure review and existing committed-frame/replay tests cover part of M1c. Check the merged M1-B install, real job and cache-only receipts, then audit the exact merged baseline before landing integrated code. Specifically trace byte ownership from accepted steps to published blobs, replay and read-only inspection, check rejected and truncated outputs, and identify which occupancy, interval and event data the current cache actually provides. Record unresolved M1 scene gates separately, including the piston receiving-volume finding. A native Points fixture or in-progress Blender branch cannot close entry.

## Review Focus

- A changed, missing or malformed committed record must fail before reconstruction (Task 1).
- A band-only input needs authoritative interior/cavity occupancy to avoid a false inner shell (Task 4).
- Opposite sheet sides and unresolved thin components must retain distinct identity or have explicit measured loss (Task 4).
- A rejected substep and two opposing births between publications must produce zero and two events respectively (Task 5).
- Moving baked foam must move the mask. Look edits must preserve published mesh data, geometry and identity (Task 7).

## File map

- `tools/native/m2_surface_baseline.py` and `tools/native/tests/test_m2_surface_baseline.py` own the cache adapter, deterministic reference extraction and reporting.
- `tools/native/m2_motion_interval.py` and `tools/native/tests/test_m2_motion_interval.py` own the independent analytic transport experiment.
- `docs/references/m2-fixed-cache-evidence.md` records portable artifact pins, reproduction and measurements. Raw exports and timings stay under ignored `out/`.
- `schemas/mesh/` and `native/include/fluidity/host/surface_tracker_3d.h` own Mesh products and tracking state. `schemas/simulate/` owns the upstream interval/occupancy contracts. Use the existing `cmake/LiquidCacheSchemas.cmake` pattern for schema conformance.
- `schemas/whitewater/`, `schemas/volumes/` and the corresponding native host interfaces own downstream outputs. Extend `native/cli/fluidity_simulate.cc` only at the stage dispatch and protocol boundary. Keep each stage's job loop in its own translation unit.
- `native/include/fluidity/host/ocean_boundary_3d.h` owns wave target and relaxation controls. Wire CPU and CUDA after the history snapshot in their existing Simulate scheduler.
- `blender/fluidity/` owns the asset bindings delivered after M1-B. Keep stage/carrier tests under `tools/blender/`. Confirm the operator and export adapter boundaries against the merged extension.

## Task 1 — Pin and measure the geometric reference

**Interfaces** — `read_pinned_frames(reader: Path, run: Path) -> dict[int, dict]` returns checked publications 0, 48, 96, 144, 192 and 216 after authenticating the full 0–216 range. `extract_surface(field, origin, spacing) -> Mesh` returns indexed, outward triangles on a globally aligned regular lattice. `mesh_metrics(mesh, reference_volume, axes=None) -> dict` reports topology, signed volume and optional sampled ellipsoid radial error.

- [x] Write tests for changed export hash, inverted/empty meshes, a disjoint pair of components and a closed analytic sphere. Require exact reproducible vertices/indices, zero orientation/boundary/nonmanifold edge defects, one component, Euler characteristic 2 and under 2% volume error for the sphere at 0.10 radius spacing.
- [x] Run `python -m pytest tools/native/tests/test_m2_surface_baseline.py -q` and observe the missing baseline.
- [x] Authenticate the M1 reader, solver, resolved scene, seed and archived export with committed SHA-256 pins. Invoke native `read-frames` over 0–216 and compare the byte stream against the archived export, allowing line-ending differences. Validate the execution ID, frame clock and selected particle payloads. Canonicalize the benchmark particle order by stable ID.
- [x] Evaluate the sphere-union field with a fixed support multiplier of 1.35 times the common material radius scale. Compare isotropic and global covariance-metric support (determinant one) at 0.0125 m and 0.00625 m on the same frames. For the geometric control, extract the authored ellipsoid directly at both spacings. Use one fixed tetrahedral decomposition, edge keys from global node IDs and gradient-consistent outward winding. If either union fragments the drop, retain it as a negative control and test a full-volume density kernel with 0.025 m Gaussian sigma, four-sigma cutoff and a fixed 0.5 occupancy threshold.
- [x] Rerun the targeted tests. Run `python tools/native/m2_surface_baseline.py --reader READER --run RUN --output REPORT` twice on the fixed inputs. Compare deterministic geometry digests and numerical metrics. Record frame-zero radial error against the analytic ellipsoid separately from volume error against the material sum at every sample. Include components, Euler characteristic, boundary and nonmanifold edges, wall time and isolated peak process RSS. These experiments cover a full-volume isolated drop.
- [x] Commit the reference, tests, plan and portable evidence.

## Task 2 — Measure MotionInterval independently

**Interface** — `study() -> dict` reports analytic endpoint transport errors for endpoint hold and linear timestamp interpolation with 1, 2, 4 and 8 sampling intervals, separating temporal representation error from RK2 integration error. If 8 misses the predeclared budget, add 16 and retain the failed measurement.

- [x] Add a test for a time-varying translation and a radial sheet folded by divergence-free differential rotation. Use analytic integrals for endpoints and assert interpolated error decreases under refinement and constant velocity transports exactly.
- [x] Run `python -m pytest tools/native/tests/test_m2_motion_interval.py -q` and observe the missing study.
- [x] Implement `study()` using fixed 24 fps output and analytic integral controls. Record RMS/max endpoint errors in metres and cell widths and sample bytes at each rate. Record the chosen provisional error budget before measurement.
- [x] Rerun tests and `python tools/native/m2_motion_interval.py`. Use the results as manufactured controls for Task 4. A particle endpoint cache alone cannot qualify carrier interpolation in the breaking wave.
- [x] Commit the experiment and its distinct evidence.

## Task 3 — Reproduce and compare Zhao CNN separately

**Interface** — An out-of-tree, pinned research invocation consumes matched full and band-limited particle snapshots and writes scalar queries, extracted geometry and per-stage receipts. It adds no installed dependency.

- [ ] Add `tools/native/m2_zhao_experiment.py` and `tools/native/tests/test_m2_zhao_experiment.py`. Test published patch offsets, an independently evaluated quartic density stencil, a normalization contributor outside direct patch support, and the occupied-cell pruning windows. Run `python -m pytest tools/native/tests/test_m2_zhao_experiment.py -q` first against the missing module and again after implementation.
- [ ] Run the pinned ONNX weights in an isolated CPU environment. Use an independent implementation of the published coordinate, normalization and pruning formulas. Compare frozen full-data pruning with recomputed pruning so missing feature support and changed query selection stay separate. Label downstream cuts of M1 particles as synthetic bands. Record the deviation from the upstream TensorRT pipeline, the unresolved terms and any unrun authored input.
- [ ] Resolve software/weight terms, the archive hash and toolchain. Record the blocker if terms or a runnable reproduction are unavailable. Preserve the published stride 2, `h_phi=h_N/2`, fixed batch 1024, 8³ input and all pruning/normalization choices.
- [ ] Run supplied full-volume and analytic controls, then matched FluidiTy snapshots with actual retained support, density variation and reseeding. Separate field error to full-data CNN from zero-set error, topology, lost components, volume, latency, peak memory and input coverage.
- [ ] Compare monolithic and tiled queries with global origin/order, identical batches and matched pruning. Start from the 9 h_N L∞ dependency bound and measure query selection, density, extraction and uncertainty halos separately. Exercise a deliberately missing normalization halo as a negative control.
- [ ] Record all four §9.2 gate decisions. Evaluate license, deterministic topology and complete extraction/export cost within the combined job budget before proposing any promotion.

## Task 4 — Qualify persistent Mesh and replay

**Entry** — M1-B merged at `4cf1715`. The [post-M1-B audit](../../references/m1c-post-blender-audit.md) records the handoff and open M1 physical gates. Task 1 supplies the stateless comparison and Task 2 supplies temporal fixtures.

**Interfaces** — Mesh consumes committed particle and occupancy fields, obstacle closure and a validated timestamped `MotionInterval`. Its checkpoint owns previous geometry, IDs, lineage, frame and backend/settings identity. It publishes a mesh commit keyed by upstream content plus resolved Mesh settings.

- [x] Implement the host time sampler and bounded midpoint transport first. Follow the [native MotionInterval plan](2026-11-10-m2-motion-interval.md). The publication step below adds the accepted-state producer and persistent carrier.
- [ ] Add native fixture coverage in `native/tests/test_surface_tracker_3d.cc` and job cases in `tools/native/tests/test_mesh_replay.py`. Run the selected preset's `ctest -R 'SurfaceTracker3D|MeshCli'` before and after implementation. The first run must reproduce the missing interface and the final run must execute every named case.
- [ ] Add native tests for a deep tank with cavity, wet wall and detached control, advected translating and folding sheets, split and merge, intersection, droplet transition at below 2h and above 3h, and invalid carrier. Require oriented closed manifold output under declared closure and global intersection checks.
- [ ] Publish occupancy and `MotionInterval3D` as optional authenticated Simulate artifacts. Old point-only caches remain inspectable. A Mesh job requiring those inputs fails with a named missing-upstream diagnostic. Fingerprint sample times, valid masks, interpolation policy and upstream content rather than checkpoint filenames.
- [ ] Implement checked motion sampling and RK2 with spatial steps at most 0.5h, rate-limited projection, separation/obstacle/gradient checks and local deterministic repair. Report lost sheet area/volume, repair area, ID survival with denominators and actual raw/corrected mesh volume. Volume adjustment moves eligible vertices with the ±0.25h cap and checked backtracking.
- [ ] Add `AdvanceSurfaceTracker3D(const MeshState3D&, const MotionInterval3D&, const GeometricTarget3D&, const ResolvedMeshSettings3D&) -> absl::StatusOr<MeshAdvance3D>` to `surface_tracker_3d.h` and implement in `native/src/surface_tracker_3d.cc`. The result contains the next MeshState and diagnostics. Publish geometric velocity, sampled fluid velocity, ID, age, wetness, thickness, phase and repair lineage. Derive geometric velocity from correspondence and time in seconds.
- [ ] Test checkpoint seek, cancel/resume, rejected repair, incompatible fingerprints and fail-before-publish. Compare exact published bytes and surviving IDs with uninterrupted runs on the same artifact. Verify upstream bytes before/after.
- [ ] Compare against Task 1 at equal error and against anisotropic/constrained skinning on the adversarial controls. Keep geometric render smoothing and volume diagnostics separate. Commit only after the differential and replay checks.
- [ ] Score Q-SURF-V, Q-SURF-N, Q-GRID, Q-IDS and Q-CAVITY separately. Declare the small-component absolute volume floor and track cumulative lost volume/count. A closed drop with biased raw volume does not close these gates.

## Task 5 — Whitewater and Volumes

**Interfaces** — Accepted simulation produces ordered `(t_s,x,u,n,w_s)` source events with `w_s=V_c(t_s) Φ_c dt_s`. Whitewater consumes those events, MotionInterval and the accepted SDF/obstacles. Its checkpoint stores points, hysteresis, remainders and RNG state. Volumes consumes committed Whitewater and valid source fields and publishes density with lineage.

- [ ] Add `native/tests/test_whitewater_3d.cc`, `native/tests/test_volumes_3d.cc` and `tools/native/tests/test_whitewater_replay.py`. Use `ctest -R 'Whitewater3D|Volumes3D|WhitewaterCli'` on the selected preset for the missing-feature and completed-feature runs. Verify all named cases executed.
- [ ] Add tests for two opposing bursts, rejected trials, exact drag including zero and stiff limits, missing carrier per class, wall collisions, hysteresis and capacity rollback. Compare births and position/velocity/age distributions at two output rates with unchanged physical source spacing.
- [ ] Add `AdvanceWhitewater3D(const WhitewaterState3D&, std::span<const WhitewaterSourceEvent3D>, const MotionInterval3D&, const ResolvedWhitewaterSettings3D&) -> absl::StatusOr<WhitewaterAdvance3D>` in `native/include/fluidity/host/whitewater_3d.h`. Require source events in accepted time/cell order. Return points and next source state atomically on success. Publish point ID, kind, age, velocity and radius.
- [ ] Implement §10 potentials on the fixed physical grid and deterministic quadrature-time births, drag/transport and classification. Keep density generation one-way. Publish named foam/bubble density grids with explicit validity, physical scale and immutable Whitewater lineage.
- [ ] Add `BuildWhitewaterDensity3D(const WhitewaterPointFrame3D&, const VolumeGridSpec3D&) -> absl::StatusOr<WhitewaterDensityFrame3D>` in `native/include/fluidity/host/volumes_3d.h`. Integrate with the Volume manifest's exact Whitewater content identity. Exercise empty frames and reject invalid spacing, unsupported fields and mismatched lineage before publication.
- [ ] Test mass/integral, zero/empty output, checkpoint replay, stage invalidation and byte-identical upstream data after changing generation controls. Record timing and peak ownership per stage.

## Task 6 — Ocean Boundary

**Interfaces** — Resolve wave and absorbing-zone parameters into Simulate settings. Apply the velocity relaxation after the history snapshot, then apply checked level-source transactions. Keep the translating paddle path as an independent scene variant.

- [ ] Add `native/tests/test_ocean_boundary_3d.cc` and run `ctest -R 'OceanBoundary3D|OceanBoundaryCuda3D'` before and after the implementation. Require direct fixtures for zero zone weight and nonpositive relaxation time, not just a scene render.
- [ ] Add rest, sign and timestep subdivision tests for the exponential update, second-order Stokes target with return current, and a measured reflected-wave control. Check generation/absorption with wave gauges, particle-volume and boundary-flux ledgers, CPU/CUDA tolerance and same-artifact replay.
- [ ] Implement the source/absorber after the history snapshot with accepted births/deletions and bounded rollback. Use `-expm1(-dt*w/tau0)` with the §4/companion relaxation weight `w=(exp(xi^3.5)-1)/(e-1)` on normalized zone coordinate xi. Use the rest target for absorption. Record wave height, period, reflection and mass budget against declared fixture limits before running the shot.

## Task 7 — Integrated breaking-wave bake and Blender inspection

**Interfaces** — Bind committed mesh, foam, spray, bubble and volume products through versioned native playback. Foam Projection takes baked foam points, radius/falloff and the existing mesh and writes the named float attribute `fluidity_foam_mask` on that mesh. Whitewater settings own particle generation.

- [ ] Qualify the mesh/point/volume carriers in pinned Blender across changing and empty frames, relocation, relink and a fresh process with the extension disabled.
- [ ] Bake the paddle or Ocean Boundary scene through the public workflow. Inspect committed surface and whitewater frames. Record the scene, device, full process wall time, phase partition, frame statistics, peak host and device memory, source lineage, checkpoint/recovery and any missing qualifications.
- [ ] Keep the reference shot pinned at 8 × 20 × 8 m, 0.035 m cells near camera, 240 frames at 24 fps, 3 m inlet depth sloping to 0 m over 20 m, second-order Stokes swell H≈1 m and T≈4 s, 35 mm camera 2 m from the break at 1920 px, and the RTX 5060 Ti 16 GB target. Treat ≤10,800 s as the full-pipeline target with the separately declared ≤6 h uniform fallback policy. A bounded qualifying scene carries its own dimensions and budget.
- [ ] In headless Blender, evaluate two frames whose baked foam moves between separated patches. Check mask values follow the baked foam points, zero for an empty foam frame, and respond to projection radius/falloff. Check evaluated surface vertex coordinates, triangle indices, published Mesh identity, manifest and blob hashes before and after projection and material edits. Verify material response in a fresh render.
- [ ] Inspect the narrow sidebar in English/French, exercise failure recovery and verify the separately reported native bake and Blender gates. Keep the full reference-shot time/memory verdict separate from a bounded shot and from M3 performance targets.
