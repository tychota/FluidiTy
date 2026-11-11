# Blender artist workflow implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline. User's M1-B autonomy instruction carries forward.

**Goal:** Replace the first-bake sidebar with an understandable Physics workflow, preferences, fields, catalogs and diagnostics.

**Architecture:** Keep the checked solver and native cache reader unchanged. Separate UI, preferences, interchange and translations from operators. Use Blender-native node assets and cache readers for portable playback.

**Tech Stack:** Blender 5.2.3 LTS Python/RNA, Geometry Nodes, gettext PO/MO, existing Ogawa bridge.

**Spec:** User's artist-workflow request and the M1-B contracts in `../specs/2026-11-07-fluidity-design.md`.

## Design and approaches

A cosmetic sidebar adjustment would leave discovery and defaults unresolved. Replacing the solver or adding a custom node engine would expand beyond the verified M1 adapter. Use Physics Properties with nested sections, real AddonPreferences, and ordinary Geometry Nodes sockets. This is an implementation decision under the existing instruction to work autonomously.

Scene controls: named object inputs, meter resolution, backend, explicit timeline constraints, separate preview/bake paths, cancel/resume, native playback display and field documentation. Preferences: default backend, preview length and debug visibility. Debug: structured details and a copyable support report. Translations: editable French PO plus compiled MO loaded into Blender's translation registry. Interchange: checked Alembic export, native Alembic/USD/VDB import, capability-tested USD export; do not advertise VDB generation from point-only M1 data.

## Global constraints

- Preserve the pinned solver/device and M1 scene validation, frame zero, 24 fps, meter units and cache publication authority.
- No executable launch on file load or panel draw. No automatic overwrite of export destinations.
- Display changes do not change resolved solver inputs or cache identity. Existing edited node groups remain intact.
- Geometry Nodes output integration is supported; procedural solver source geometry remains subject to the existing box restrictions.

## Review focus

- Registration under an extension-qualified module name and repeated enable/disable.
- Settings edits while a solver job is running.
- Export errors preserving frame, selection, active object and timeline range.
- French translation contexts, formatted diagnostics and catalog packaging.
- Existing cache relink and artist-edited material/node controls.

### Task 1: Physics workflow and defaults

- [x] Add and run `tools/blender/verify_artist_workflow.py`; require Physics-only panels and registration, tooltip descriptions, preferences and node field sockets.
- [x] Create `preferences.py` and `ui.py`; preserve existing operator identifiers and scene data. Add setup and navigation operators, progressive disclosure, correct enabled states and readable status.
- [x] Run the Blender verifier and existing real-job extension verifier.

### Task 2: Catalogs, fields and interchange

- [x] Extend the verifier to exercise MO loading and native export/import with user state restoration and overwrite rejection.
- [x] Add editable PO/POT catalogs, deterministic catalog compiler, MO registry loader, interchange operators, and documented node outputs for points/radius/velocity/speed.
- [x] Run focused checks, existing offline package/render verification, and format/lint checks.

### Task 3: Delivery

- [x] Review actual Physics panels in English/French, document capabilities and remaining limitations, package a new offline ZIP.
- [x] Request fresh code review, fix material findings, verify and commit; create a follow-up PR to merged #47.

## Execution record

Used the already isolated M1-B worktree on codex/m1-blender-first-bake. Existing user instruction permits autonomous execution. Native solver binaries and numerical contracts are unchanged.

Fresh review found USD temporal resampling under changed fps; fixed with timing/unit preflight and regression checks. Expanded empty-start testing exposed Blender 5.2.3 exporting separate USD prims; reject empty-sample USD derivatives before publication, with Alembic as the supported lossless path. Windows native-reader handles are removed before temporary archive cleanup. Windows rename publishes USD without replacing an existing file; other platforms use an exclusive hard link.

All Blender checks, catalog unit tests and Python lint/format passed. Real package receipts: out/m1-artist-final-check. Final offline package: out/m1-artist-release-final/fluidity-0.2.0-windows-x64.zip. SHA-256: 74ff7505b7e2f38f8aaf16cacc9150389b614f2584c93a35fe1ab8d7495658ba. Fresh review confirmed the timing/empty-frame fixes and external GN fields. See docs/blender/artist-workflow.md for measured scope.
