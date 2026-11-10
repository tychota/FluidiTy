# M2 Integrated Delivery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal** — Complete the accepted-data, persistent surface, wave-effects and Blender breaking-wave path.

**Architecture** — Extend the accepted Simulate observer with a bounded collector. Publish immutable, content-addressed stage products with exact upstream and settings identity. Mesh, Whitewater and Volumes consume those products through checked native APIs and their own replay state.

**Tech Stack** — Existing C++20/CUDA solver, FlatBuffers and BLAKE3 cache primitives, independent Python fixtures, Lean exact contracts and Blender 5.2.3 LTS.

**Spec** — [Canonical design](../specs/2026-11-07-fluidity-design.md) and [M2 staged plan](2026-11-10-m2-surfaces-and-wave-effects.md).

## Global Constraints

- Retain the canonical M2 quality, replay, language, hardware and reliability thresholds.
- Preserve old point-only caches and their existing fingerprints when effects capture is disabled.
- Publish derived-stage manifests last and bind exact upstream point content, resolved capture settings and previous-stage checkpoints.
- Bound snapshot count, bytes, numerical work and frame queues. Capture only accepted states and clear provisional intervals on frame replay.
- Record every quality or resource failure. Neural promotion remains declined by the recorded experiment.

## Review Focus

- Old point caches remain readable and identical default settings retain their original identity.
- Rejected steps and capacity replays cannot leak carrier samples or births into a published interval.
- Missing spatial carrier or occupancy cannot become a valid zero or an invented liquid interior.
- Interrupted derived-stage publication preserves earlier products and retries compare existing bytes.
- Surface styling in Blender changes the named foam mask while the baked surface geometry and identity remain fixed.

## Task 1 — Accepted source products and shared stage publication

**Files** — Create `native/include/fluidity/host/effects_cache_3d.h`, `native/src/effects_cache_3d.cc`,
`native/include/fluidity/host/motion_capture_3d.h`, `native/src/motion_capture_3d.cc`,
`schemas/effects/stage_manifest.fbs` and `schemas/effects/motion_frame.fbs`. Extend schema generation,
the Simulate resolved options and accepted observer. Add cache and capture native tests and CLI replay tests.

- [ ] Test manifest lineage, corruption, interrupted publication and no-replace replay through the existing native filesystem fault seam.
- [ ] Test a bounded collector on accepted states, duplicate epochs, capacity rollback, endpoint inclusion, invalid coverage and signed coordinates.
- [ ] Implement `EffectsCache3D` with exact source/settings/build identities and separate payload content. Reuse the existing native lease, bounded byte read and atomic publication operations.
- [ ] Implement `MotionCapture3D` with at most 17 samples per 24 fps interval initially. Store accepted timestamps and compact blocks/cells/faces, plus endpoint material targets. Record actual sample gaps and bytes. Keep source event statistics per accepted substep independently of carrier decimation.
- [ ] Add optional resolved capture settings and wire the observer and rollback path. Verify default FLRS bytes and old caches, then interruption/resume and CPU/CUDA source parity.

## Task 2 — Geometric Mesh, persistence and replay

- [ ] Add a native geometric density reference, globally consistent tetrahedral extraction, manifold/orientation and intersection checks, per-component material targets and measured vertex correction with backtracking.
- [ ] Add persistent MeshState, checked MotionInterval advection, bounded projection and deterministic local repair/lineage. Test calm correspondence, drop-to-pool, wet wall, cavity, folding and thin-sheet controls.
- [ ] Publish mesh fields and checkpoint state under exact source lineage. Add CLI Mesh replay and cancellation tests and independent Python quality scoring at two resolutions and translated/rotated origins.

## Task 3 — Whitewater and Volumes

- [ ] Implement the §10 source statistics, accepted birth events, deterministic remainders, quadrature-time births, exact drag and hysteretic classes.
- [ ] Test opposite bursts, rejected events, output-rate comparisons, zero/stiff drag, missing carrier and obstacle collision. Publish replayable points with source lineage.
- [ ] Implement conservative foam and bubble density output with validity, native volume export and empty-frame tests.

## Task 4 — Ocean Boundary and shot integration

- [ ] Implement resolved second-order Stokes and absorbing-zone inputs after the history snapshot, with transactional particle level control and return current.
- [ ] Verify timestep subdivision, reflection, source volume and CPU/CUDA parity before running the integrated scene.
- [ ] Run the declared bounded breaking-wave bake, then the full reference workload when supported by measured uniform capacity and time. Preserve the M3 adaptivity boundary.

## Task 5 — Blender workflow and final gates

- [ ] Qualify native mesh/whitewater/volume carriers, stage controls and cache-only reopening in Blender 5.2.3 LTS.
- [ ] Implement Foam Projection and test moving and empty baked foam, projection controls, material response, exact surface geometry and cache identity.
- [ ] Verify English/French workflow and recovery and run the complete gate matrix across Python, native CPU/CUDA and applicable Lean contracts. Review the whole branch, fix important findings, open the M2 PR and merge only after its required checks pass.
