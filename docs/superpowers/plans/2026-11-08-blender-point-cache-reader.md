# Blender Point-Cache Reader Qualification

> Implement the tasks in order with independent reviews and retain ignored raw evidence.

**Goal.** Qualify a pinned native Blender reader for published point frames on Windows.

**Scope.** Produce a small Ogawa Alembic Points archive outside Blender, bind it through Blender's native Alembic importer, and measure evaluated points in a fresh process. This is an M0 fixture and reader qualification. The simulation publisher, extension, restart schema, and production cache representation are later integration decisions.

**Design authority.** [FluidiTy design](../specs/2026-11-07-fluidity-design.md), sections 14.1–14.2 and 15.2. Blender 5.2.3 LTS Windows x64 build e13b728c6880. The earlier Blender-exported mesh test read a later nonempty sample at the expected empty frame. Use an independent native producer and the Alembic Points schema so the archive can contain a true zero-point sample. A vertex-only mesh carrier would repeat the earlier representation without native point widths. OpenVDB addresses volumes. Select an Alembic Points reader for this fixture only if the measured gate passes.

**Language.** A committed point frame is one published point sample at a physical time. An Alembic sample is the carrier record. A Blender timeline frame selects the sample. Display radius is half the stored Alembic point width. A particle ID and diagnostic velocity belong to published data and inspection. Display requires world position and radius. Measure diagnostic velocity exposure separately and keep full IDs in the source archive.

## Task 1 — Independent Alembic point-frame producer

- Create standalone tools/blender/point_cache_writer CMake/Ninja and Conan configuration. Use Alembic 1.8.8 with exact Conan recipe and resolved transitive revisions in a separate lock. The native operator lock, presets, CMake targets and dependencies are unaffected. A small standalone C++20 executable is allowed to catch upstream Alembic exceptions at its CLI boundary.
- Expose fluidity_write_point_cache <output.abc>. Reject missing arguments and invalid output paths with a nonzero exit. Write one Ogawa Points object named FluidiTyPoints with four explicit samples at Blender timeline frames 1, 2, 3 and 4 with sampling step 1/24 seconds and first time 1/24. Expected point counts are 3, 5, 0, 4, with no placeholder point. For zero-based point index i and timeline frame f, Blender Z-up position is (f, i, -(f+i)), display radius is (i+1)/16 m, diagnostic velocity is (1, -2, 3) m/s, and ID is 100+i. Convert positions and vectors to Alembic Y-up as (x,z,-y). Store widths as diameter in the Alembic schema and preserve uint64 IDs. Use explicit typed empty arrays.
- The executable reopens its archive and independently verifies one Points schema, sample times, exact cardinalities including all empty arrays, positions, widths, velocities and IDs. It prints a clear pass marker and returns nonzero on mismatch. RED by starting with a compilable stub and proving the producer check fails. GREEN via a real pinned Conan install, Release build, archive readback, and a CLI negative check. Record the recipe and package revisions and build tools, keep the binary, archive and logs ignored, and commit the task separately.

## Task 2 — Native Blender playback qualification

- Under tools/blender add a Blender Python script with separate prepare and verify modes and a locked uv standard-library launcher. The launcher runs two fresh pinned Blender processes with factory startup, Python auto-execution disabled and nonzero Python exception exit handling. The first imports the actual writer's archive, sets scene units and 24 fps, verifies the point-cloud object and native cache modifier, stores a relative cache binding, and saves a .blend. Move a copy of the scene and archive together to a directory with spaces and non-ASCII characters. The second opens that copied .blend without an extension and evaluates the cached points. Use an isolated, newly created ignored output directory with raw logs and receipt.
- Visit timeline frames 1, 2, 3, 4, 3 and 1. Require counts 3, 5, 0, 4, 0, 3. Check all world positions and display radii using the Task 1 formula and 1e-6 tolerance. Include at least one check distinguishing times with similar point indices. Record whether expected point-domain velocity is readable and agrees with the fixture. Do not infer IDs from vertex order. Native playback needs no frame-change callbacks.
- RED through a deliberately wrong expected nonzero count or incorrect radius on the real archive, then GREEN with the actual archive and saved-scene playback. Also require a negative test with the copied archive missing to distinguish stale startup data from valid playback. A successful receipt requires the exact Blender build, native cache binding, every requested sample, correct required attributes and process success. A measured carrier failure remains a failed qualification with exact observations.
- Format and lint the scoped sources. Keep executable, .abc, .blend, receipts and raw logs ignored. Commit the task separately.

## Task 3 — Evidence and acceptance

- Write reproducible commands and measured Windows/Blender/Alembic versions, counts, attributes, timings and limits in docs/blender/README.md. Name the supported carrier and intended follow-up constraints only if the real gate passes. Record which M0 gates remain.
- Run a clean repeat from the final source with real native writer, saved scene and reopened Blender process. Run relevant locked Python and formatting checks. Finish with a fresh independent review and resolve findings in separate commits. Preserve the local branch and worktree for review.

## Review focus

- A real zero-point sample must round-trip. Frame order includes backward seeks and repeated evaluation.
- World positions exercise axis conversion. Radius values exercise per-point widths and diameter conversion.
- An actual external Alembic archive survives reopening and relocation. Missing archive cannot publish a passing receipt.
- Per-point velocity diagnostics and archive IDs are distinguished from Blender-render requirements.
- The qualified result is tied to a pinned Blender build and a specific Alembic carrier.
