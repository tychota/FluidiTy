# M1-B First Blender Bake Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring committed 3D M1 point frames into a portable Blender scene and ship a bounded offline first-bake workflow.

**Architecture:** A separate native bridge links the checked M1 cache codec and the qualified Ogawa writer. The extension uses explicit subprocesses for authoring and simulation, verifies cache content through the bridge, and saves a relative Alembic binding and native Points node group in the scene.

**Tech Stack:** C++20, M1 FlatBuffers and BLAKE3, Alembic 1.8.8 Ogawa, Blender 5.2.3 LTS Python and Geometry Nodes.

**Spec:** [FluidiTy design, §§14–15](../specs/2026-11-07-fluidity-design.md).

## Global Constraints

- Windows x64, Blender 5.2.3 LTS build `e13b728c6880`, 24 fps and one Blender unit per meter.
- Bake target: GeForce RTX 5060 Ti, CC 12.0 and driver 610.88; CPU workflow results are separate.
- Frame zero is time zero; every subsequent sample is at frame ordinal / 24 seconds. Use the manifest commit as the only publication authority.
- Cache-only render uses saved Blender data and no frame-change callbacks, solver, network, or extension.
- Preview is a real shortened run with distinct resolved inputs and cache; native `--preview` is read-only.
- Keep M1 physics, replay and hardware gates separate from Blender workflow evidence. A failed M1 gate does not invalidate independent adapter measurements.

## Review Focus

- An orphan JSON mirror or corrupt point blob cannot become a displayed sample (task 1).
- A new attachment that fails validation must leave the previous cache and edited material intact (task 3).
- A path with spaces or non-ASCII characters works both before and after relocation (tasks 1 and 3).
- Editing a supported animated paddle invalidates the resume fingerprint without replacing a frame (task 2).
- Opening a saved project does not launch any executable named in its Blender data (task 3).

## Decision and file map

Use the native checked cache reader rather than reproducing FlatBuffers validation in Python. A separate bridge keeps the solver artifact and resume fingerprint unchanged. Reuse the qualified OPoints representation; the previous vertex-only mesh path failed the empty-frame control. Use a translating infinite slab represented by a closed axis-aligned paddle proxy. Reject rotation, transverse travel, non-linear motion and non-box geometry; finite paddle ends require a solver feature outside this slice. Blender Z-up maps to solver Y-up by `(x,z,-y)`; inverse is `(x,-z,y)`.

- `tools/blender/point_cache_writer/`: checked frame inspection/export and the Ogawa sample writer shared with the standalone qualification fixture.
- `native/cli/blender_resolve.cc`: canonical bounded scene authoring.
- `blender/fluidity/`: manifest, scene preflight, jobs, native cache binding, translations and operators.
- `tools/blender/`: package staging, deterministic example creation and headless receipts.
- `docs/blender/m1-b-evidence.md`: per-gate evidence and pending qualification.

---

### Task 1: Read committed frames and export native Points

**Interfaces:** `fluidity_blender_bridge inspect CACHE FRAME` returns verified JSON with execution ID, time, positions, velocities and radii in Blender meters. `export CACHE LAST OUTPUT` exports contiguous committed frames 0 through LAST into a new Ogawa file; a missing frame fails and leaves no published archive.

- [x] Add a headless test comparing a real cache to every Blender sample at integer frames. Assert counts, positions and radii; revisit frames out of order, including an independently committed empty cache; check relative bindings after copying.
- [x] Run the test without the bridge. Expected: missing bridge.
- [x] Build the bridge using the existing cache store and shared OPoints writer. Check the published clock and float narrowing. Preserve stable IDs and encode width as diameter. Use a temporary archive and promote only after a successful close.
- [x] Rerun the test in the pinned Blender build. Require corrupt content and missing commit tests to fail with the output absent.
- [x] Commit the adapter and its receipt summary.

### Task 2: Author bounded scenes and exercise real jobs

**Interfaces:** `fluidity_blender_bridge resolve SOLVER BACKEND INPUT SCENE STATE` encodes canonical FLRS and frame-zero FLST from a bounded versioned text packet. Python `scene_inputs(scene, preview=False)` returns this packet from evaluated world-space geometry. `Job.poll()` drains bounded protocol work; `Job.cancel()` sends `cancel\n` then times out safely.

- [x] Add a headless operator test covering field errors, separate Preview fingerprint, fresh Bake, cancel after a frame commit, Resume, and changed-paddle rejection. Compare SHA-256 of all preexisting committed records before and after each recovery path.
- [x] Run against the incomplete extension. Expected: missing operator.
- [x] Implement scene and job modules using native defaults and the real solver CLI. Limit to one active job per scene and one solver writer per cache. Restore frame after sampling evaluated world motion and keep all transient handles outside saved IDs.
- [x] Rerun on the pinned RTX backend. Require structured failure and unchanged committed records for incompatible Resume.
- [x] Commit the authoring and job workflow.

### Task 3: Package and exercise portable playback

**Interfaces:** `fluidity.inspect` reads verified committed points; `fluidity.relink` checks a replacement before updating the saved native binding. Package the Windows x64 bridge and solver with the Blender extension and example.

- [x] Add a fresh-process test for offline install, idempotent attach, invalid replacement, independent relink, cache-only open and two distinct renders with the extension disabled.
- [x] Run before packaging and binding are implemented. Expected: missing package or operator.
- [x] Add the native Points asset with editable size and material inputs, relative bindings, cache-only Inspect, example scene, offline build script, and English/French strings.
- [x] Repeat the fresh-process test; review the actual narrow sidebar in both languages and record what was observed.
- [x] Record exact package/build pins, receipt commands, open M1 gates, and evidence still needed. Commit; review the entire diff before opening the PR.
