# Blender domain and fluid authoring plan

> Execute inline using executing-plans, test-driven-development and verification-before-completion. The user's existing autonomy instruction applies.

**Goal:** Turn the first-bake adapter into an authoring workflow with real domain boundaries, material presets, collider models and compact Blender UI.

**Design:** A UI-only change would leave the domain inert and material controls fictitious. A new arbitrary-mesh collision solver is a separate numerical project. Extend the adapter to the engine's existing open tank, halfspace, slab and free-space models, and expose material/step parameters through a checked version-2 input packet. Keep the original packet/legacy paddle path for existing scenes. New quick setup uses an open tank. Presets are editable numerical starting points, with no viscosity claims.

**Constraints:** Preserve v1 resolved bytes and old test scene behavior. Native inputs must reject nonfinite/invalid parameters before publication. Each new parameter must affect canonical resolved settings or particle state. No unsupported collection of arbitrary colliders. Keep 24 fps, units and committed cache validation. Do not touch user attachments.

## Tasks

- [x] Add Blender authoring integration tests: create domain/fluid/collider, select/apply preset, inspect version-2 packet, resolve/check and real short bake; preserve legacy scene and resume tests.
- [x] Implement checked v2 native resolution with material, initial velocity, step options and collision model; rebuild bridge, keep numerical solver unchanged.
- [x] Implement Blender quick setup/assignment, material presets, native parameter fields and scene validation; map Blender Z-up to solver Y-up.
- [x] Replace paragraph rows with aligned compact controls, concise state labels and explicit contextual help; review narrow English/French panels including domain, fluid and collider controls.
- [x] Update catalogs/docs, verify new workflows plus legacy package/render, review, package and update PR #50.

## Review focus

- Tank coordinates and particle containment, including a fill at a boundary.
- Material/velocity/solver edits invalidate Resume through the resolved fingerprint.
- Presets do not reset cache paths, geometry or unrelated display settings.
- Creating or assigning a collider cannot silently turn a closed-domain simulation into a different collision model.
- Existing legacy .blend data and native packet-v1 calls remain valid.

## Execution record

Implemented inline in the existing M1-B worktree. Version-2 authoring uses the existing native collision/material models; the numerical solver executable SHA-256 remains ee953ed5f397e0f3418b408197c19267752dd68b3f062dc2cab522fa2f28eab8.

Fresh review found a hidden inactive collider blocking Tank/Free after changing models. The resolver now consumes role pointers only for the selected model; transition tests pass while retaining assignments for switching back. No additional concrete defect was found. Existing v1 resolved/state bytes match the previously packaged bridge exactly.

Passed short real Tank/Free/Plane/Slab bakes on CPU and RTX, installed-package authoring, physical-parameter identity checks, initial-velocity mapping and material-change Resume rejection. The default Water tank completed 48 frames with 64 points above the floor. Legacy offline install/recovery/render checks also passed. Ruff, catalog tests and clang-format 19.1.5 over all tracked native sources passed. Reviewed installed English/French panels at 424-pixel Physics width; persistent paragraph rows are gone.

Offline package: out/domain-release/fluidity-0.3.0-windows-x64.zip; SHA-256 f783ff942fa5a5039a6f53142d4f515b13320226ceb71f1cb122f25982162359. Includes Water Tank.blend and original First Bake.blend. Receipts and scope are in docs/blender/artist-workflow.md. PR #50 remains the review destination.
