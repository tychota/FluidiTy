# Committed 3D Liquid State and Checkpoint Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Run deterministic bounded uniform 3D liquid substeps on CPU and CUDA, publish only coherent frames, and resume or inspect them from a versioned checkpoint.

**Architecture:** The frame scheduler runs the same ordered substep contract against checked CPU and CUDA workspaces. A committed version, a trial version and an immutable frame-start snapshot keep physical state separate from retries and capacity growth. A cache writer fences committed epochs and publishes verified content-addressed records before a frame manifest.

**Tech Stack:** C++20, CUDA and pinned CCCL, CMake/Ninja, Conan 2, uv, GoogleTest, Lean/mathlib, FlatBuffers 25.12.19 and BLAKE3 1.8.5.

**Spec:** [FluidiTy design](../specs/2026-11-07-fluidity-design.md) §§4.2, 5–5.2, 6.1, 6.3, 11, 14.1–14.3 and 15, [F9](../../references/formal-contracts.md), and the [M1 roadmap](2026-11-08-m1-uniform-3d-liquid-roadmap.md). Start from PR #12 merge 7a205d952a4692aea9312cadc54b8f261d724cb0. Existing scalar, transfer, pressure and contact oracles remain required.

## Global constraints and design choices

- Scope is the uniform sparse single liquid, one supported translating halfspace/slab, and committed state, frame scheduling and restart. Disable wave, viscosity, gas/EOS, whitewater and adaptivity stages as explicit identities. Blender, production scene quality, full job throughput and later material models remain in their roadmap slices.
- In host code use descriptive domain names, two-space blocks, four-space continuation and 120 columns. Public headers document time level, coordinate frame, units, failure, lifetime and aliasing. Source comments explain ordering, precision and invariants. Follow AGENTS.md.
- Use independent owning versions. Persist stable particle ID/tag, world position and velocity, APIC slope rows and affine-off, mass, reference volume, J, side and PIC initialization state. Persist active block keys, generation/ages, selected cell occupancy and prior pressure, owned carrier velocity and validity, resolved source ordinals/remainders, seed/accepted-step Philox key inputs, next ID, time/frame/substep and epoch. Do not serialize padding, scratch, addresses, capacity or event handles as physical state. Record capacity plan and execution identity in separate resume metadata. Future persistent families are enumerated in F9 but are absent from the liquid wire format.
- Candidate acceptance swaps the whole version and increments the epoch, accepted-step ordinal and time exactly once. A rejected attempt changes only diagnostics. Only pressure nonconvergence and step-dependent stability/certification rejection may request smaller dt. Incompatible physical constraints, invalid data/range and device errors are fatal. From the same committed version, allow at most two retries at dt/2. On capacity rejection leave the graph and replay once from the immutable frame start after resource-checked growth. Never grow inside the physical attempt or consume IDs on preview/reject.
- Build dt from committed state, source and translating wall preview, the remaining frame interval, displacement ≤0.5h, and applicable capillary limits. Use remaining/ceil(remaining/dt_max) and cap the final accepted time exactly to the frame boundary within the verified floating-point condition. After a rejected attempt, halve its dt and recompute the remaining schedule only after acceptance. Check actual displacement, finite fields, source counts, true pressure residual and volume-scaled defect before acceptance.
- Follow §5 order. Stable sort by signed bin in (z,y,x) order, preserve incoming order for equal keys, activate all 81 original staggered samples including zero weights and all six faces of selected pressure cells. Frozen center and corner distances use one deterministic world-space field per attempt, supplied by the resolved qualification scene at time n. The plan admits an analytic signed-distance field for qualification and a fixed ordered union of particle spheres for particle-defined scenes. Use min of physical signed distances with a declared positive radius and deterministic tie order. Scene acceptance and geometry refinement criteria remain in slice 5. Check old/new wall samples at t and t+dt and store carrier validity independently of mass and side.
- Particle equations use α=0.99e and β in [0,e]. For liquid, enable β=e only with positive true-gradient divergence and distance to the free surface ≤h on the fluid side, excluding wall influence within h; otherwise use zero. Start new particles PIC. Store v_pic+αδ and advect with v_pic+βαδ; apply checked wall contact separately. Update J with exp(dt tr L), reject nonpositive/nonfinite J, preserve m and V0. Deposit signed round-to-even quanta with existing contributor guard and no saturation. Preserve pressure links and capillary lift from slice 3.
- Workspace allocation and trial failure are explicit. Query checked bytes/slots outside an attempt, enforce the device ceiling min(0.70 total, 0.80 free at job start), use fixed address ranges and count every reservation, then execute against caller-owned scratch spans. Compute a conservative checked workspace requirement for the worst legal active shape, including coarse pressure levels, solver work vectors, scan/sort, snapshots and pending uploads. Existing owning APIs remain convenience wrappers outside the scheduler. Instrument host allocator and CUDA allocation/event creation call sites to test that physical attempts use no heap/device allocation or deallocation. Preserve outputs on error. CUDA branch uses real device numerics, explicit 120-real/120-virtual, pinned deterministic sort/reduction and no CPU fallback. Capture or update graphs outside physical attempts and rebuild on storage growth.
- Cache bytes use FlatBuffers with explicit field IDs and ForceDefaults. BLAKE3-256 hashes domain-separated canonical resolved settings and exact artifact bytes. Keep hashed physical data independent of process/heartbeat and local paths. Checked readers enforce identifier, version, declared length, digest, bounded counts, state relations, enum/range constraints and safe content-relative names. Resume requires identical resolved inputs and execution identity; inspection remains portable. Same-artifact replay compares byte content and reports divergence without rewriting. Cross-backend checks use declared numerical tolerances and exact topology/order/IDs. Adopt FlatBuffers recipe 071c47a360801bf3ee2aaa81262c5040 and BLAKE3 recipe 7e63c1d9e2a1ad7401ec11d6c5e47efb, verified on Windows in the ignored dependency preflight. Qualify in the real four-preset and hosted matrices.
- Publish blob and JSON mirror temporaries on the destination filesystem, flush data, rename files, and publish the FlatBuffers frame manifest last by atomic rename. Existing content with the same digest is reused and byte-verified; mismatches, incomplete writes and canceled trials leave earlier manifests readable. Retain the last two checkpoints at ten-frame intervals plus the initial checkpoint until sufficient later history is available. A published frame record is immutable. Process liveness/heartbeat is separate from immutable content. Fence each source arena with its epoch and an event until a pinned D2H snapshot completes; crash diagnostics use the most recent valid host snapshot only.
- Use uv, locked Conan, Ninja presets, real RTX 5060 Ti CC12.0, named-pipe Compute Sanitizer and independent proof verifier. Pin actual tool/recipe/binary revisions and source hashes. Retain ignored logs and ledger. Do not treat absent or skipped checks as passing.

## Review focus

- A candidate can alter topology, source remainder and warm-start state before failure. Check the entire committed snapshot and identity after each injected stage error (Tasks 1, 5 and 10).
- A bin may be negative or lie exactly at a half-grid boundary. Assert order, signed lattice checks and zero-weight activation before P2G (Task 2).
- A frame may have accepted substeps before capacity failure. Verify one rollback and byte-for-byte replay from frame start and reject a second growth (Tasks 1, 5 and 10).
- An epoch can still be in a device arena when a later attempt wants to overwrite it. Hold a real D2H completion event, attempt reuse and show the original host bytes are preserved (Tasks 6 and 8).
- An interrupted publish may leave a temporary manifest, a valid orphan blob, or a live process marker. Resume from the last verified immutable manifest and classify orphaned baking as crashed (Tasks 7, 9 and 10).

## File ownership

| Area | Files |
| --- | --- |
| Persistent state and frame transitions | native/include/fluidity/host/liquid_state_3d.h, native/src/liquid_state_3d.cc, native/tests/test_liquid_state_3d.cc |
| Deterministic activation and workspace accounting | native/include/fluidity/host/liquid_step_workspace_3d.h, native/src/liquid_step_workspace_3d.cc, native/tests/test_liquid_step_workspace_3d.cc |
| Bounded numeric workspace APIs | Existing liquid boundary geometry, transfer and MAC pressure sources with adjacent internal workspace headers and tests |
| Ordered liquid scheduler | native/include/fluidity/host/liquid_substep_3d.h, native/src/liquid_substep_3d.cc, native/tests/test_liquid_substep_3d.cc |
| CUDA persistent execution | native/cuda/liquid_substep_3d.cu and host bridge, native/include/fluidity/host/cuda_liquid_substep_3d.h, native/tests/test_cuda_liquid_substep_3d.cc |
| Cache schema and publication | schemas/simulate/, native/include/fluidity/host/liquid_cache_3d.h, native/src/liquid_cache_3d.cc, native/tests/test_liquid_cache_3d.cc, CMake/Conan and Native CPU CI |
| Snapshot lifetime | native/include/fluidity/host/liquid_snapshot_3d.h, native/src/liquid_snapshot_3d.cc, native/cuda/liquid_snapshot_3d_host.cc, native/tests/test_cuda_liquid_snapshot_3d.cc |
| Replay protocol and formal bridge | native/cli/fluidity_simulate.cc, tools/native/tests/test_simulate_replay.py, proofs/Fluidity/LiquidTransaction.lean, tools/verify_proofs.py, proofs/README.md |

## Task 1: Version the complete liquid state and enforce transaction decisions

**Produces:** LiquidParticleState3D, LiquidBlockState3D, LiquidCellState3D, LiquidFaceState3D, LiquidSourceState3D, LiquidClock3D and LiquidState3D. LiquidStateCapacities3D counts particle, block, cell, face and source slots. State record arrays have explicit valid sizes and field-wise value equality. Particle order is the stable sort result, so ID order need not be increasing. Blocks, cells and faces have canonical unique keys. Store geometry and a versioned clock in state. Keep solver-independent capacities out of physical equality. Expose these transaction operations in liquid_state_3d.h:

    enum class LiquidAttemptDisposition3D { kAccepted, kRetryDt, kCapacity, kFatal };
    static absl::StatusOr<LiquidFrameTransaction3D> Create(
        const LiquidState3D& initial_state, LiquidStateCapacities3D capacities);
    const LiquidState3D& committed() const;
    const LiquidState3D& frame_start() const;
    absl::Status BeginFrame(double target_time);
    absl::StatusOr<LiquidState3D*> ResetTrialFromCommitted();
    absl::StatusOr<LiquidAttemptOutcome3D> FinishAttempt(
        LiquidAttemptDisposition3D decision, double attempted_duration);
    absl::Status RestartFrameWithCapacity(LiquidStateCapacities3D capacities);
    absl::Status FinishFrame();

The trial is borrowed until FinishAttempt. Its producers may resize within reserved capacities only and may never change the clock. LiquidAttemptOutcome3D carries disposition and optional retry_duration. BeginFrame freezes the coherent snapshot. FinishFrame requires committed physical time equal to the target and advances the frame ordinal once. FinishAttempt accepts only finite positive durations that advance time without overshooting, with the only frame-time snap restricted to one ULP on a final scheduled step. Reject when the counter or time cannot advance. Fatal and exhausted retry leave committed state at the last accepted substep. RestartFrameWithCapacity requires the most recent capacity outcome, a single checked growth and replacement of all versions, then restarts from frame_start with the same target and IDs.

- [ ] Write Debug CPU RED for poison in each trial field followed by rejection and equal entire committed values, two retries from original at 1/2 and 1/4 then stop, capacity after one accepted substep followed by one full frame restart, failed growth preserving the accepted frame start, and advancing time/epoch/step once. Include duplicate or zero IDs, malformed topology/validity, nonfinite or nonpositive J, impossible clocks, and unrepresentable next time.
- [ ] Implement checked construction and version swaps. Pre-size/reserve all versions outside attempts, keep frame scheduling diagnostics separate, check proposed accepted timestamp before publish, and preserve the epoch of an immutable snapshot. Frame-start capacity growth is one atomic replacement or leaves the previous complete versions available.
- [ ] Run LiquidFrameTransaction3D.* and both locked CPU presets. Commit feat: version committed liquid state and frame rollback.

## Task 2: Stable bins, complete support and fixed workspace accounting

**Consumes:** LiquidStateCapacities3D and persistent records from Task 1. **Produces:** LiquidStepWorkspace3D::Create(capacities, geometry), LiquidStepWorkspace3D::PrepareTrialTopology(particles, pressure_cells), and QueryLiquidWorkspaceBytes3D(capacities). Return active block keys in (z,y,x) order and a stable permutation by signed bin, using original ordinal as the tie key. Add checked preallocated refill to SparseMacGrid3D or a borrowed view sharing the same ownership/index logic.

- [ ] Write RED on negative keys and equal bins, zero-weight node block activation, six pressure face owners, persistent-key remap against old/new grids, generation changes and checked byte/capacity exhaustion. Require zero heap allocation during successful preparation.
- [ ] Enumerate checked global stencil indices before requiring an owner, sort keys/permutation with preallocated storage, deduplicate activation keys, clear scratch in used ranges, and remap history by explicit face or cell identity. Query every bounded region using checked arithmetic; expose used versus reserved counters and failure before publication.
- [ ] Run LiquidStepWorkspace3D.* and both locked CPU presets. Commit feat: activate complete liquid support in fixed workspace.

## Task 3: Preallocate geometry, compatible transfer and contact workspace

**Consumes:** Workspace extents from Task 2 and the accepted slice 3 APIs. **Produces:** checked workspace overloads of FreezeLiquidBoundaryGeometry3D, BuildLiquidPressureLinks3D, DepositCompatibleMacQuanta3D, NormalizeLiquidMacFaces3D and GatherLiquidBoundaryParticles3D that take a LiquidStepWorkspace3D view and return borrowed result views. Lifetime ends at the next use of that region. ResolveLiquidWallContact3D retains its value API.

- [ ] Write RED for CPU geometry→links and masked deposit→normalize→two-history gather against owning APIs, exact integer parity, and zero heap allocation at configured capacity. Check missing support and inadequate region preserve outputs.
- [ ] Share arithmetic with the owning paths. Use sorted scratch spans for samples, candidates, incident faces and donor anchors. Split live buffers by usage so pre-force velocities survive normalization and the projection and gather can read their respective versions. Preserve per-stage status, optional-contact semantics and numerical ordering.
- [ ] Run LiquidBoundaryWorkspace3D.* and both locked CPU presets. Commit feat: reuse bounded boundary and transfer workspace.

## Task 4: Preallocate pressure assembly and solver workspace

**Consumes:** LiquidStepWorkspace3D and borrowed typed pressure problem from Task 3. **Produces:** ProjectMacPressure3D(inputs, options, MacPressureWorkspace3D&) returning a checked borrowed MacPressureProjectionView3D; QueryMacPressureWorkspaceBytes3D(cell_capacity, face_capacity), with checked worst-case levels, coarse solve work, reductions and result vectors.

- [ ] Write RED for mixed anchored/gauge, incompatible sealed, multilevel, warm-start and exhausted workspace cases. Match owning projection diagnostics and values, including true original residual and physical defect, and require zero allocation on successful projection.
- [ ] Use bounded sorted topology and scratch spans for row/face indices, levels and iterative work. Legacy owning API allocates a workspace at its normal boundary and returns owning copies. Distinguish solver aborted from physically incompatible precondition. Keep all F5/F6 bridges and avoid writing trial carrier velocities until all gates pass.
- [ ] Run MacPressureWorkspace3D.* and both locked CPU presets. Commit feat: reuse bounded MAC pressure workspace.

## Task 5: Integrate the ordered CPU physical substep

**Consumes:** Tasks 1–4. **Produces:** LiquidSceneInputs3D for resolved material, geometry, source, force and wall timeline, LiquidStepOptions3D for energy/timestep/capacity settings, LiquidStepDiagnostics3D, and LiquidSubstepScheduler3D::AdvanceFrame(target_time) using explicit stages and immutable frame-start storage. Expose a test-only stage fault hook separate from physical state and a checked single-attempt entry for replay diagnostics. Put common ordering in liquid_substep_3d.cc and distinguish backend execution through checked stage operations.

- [ ] Write RED for observable stage order, analytic uniform gravity at e=0, 0.6, 1 with 0.1% acceleration tolerance, stored/advection distinction, finite J update, static capillary and translating wall use, source preview/retry stability, and a failure at each physical stage. Compare committed values before/after failure and assert no physical-attempt allocation. Include nonconvergence retry versus sealed incompatible fatal.
- [ ] Implement ordered emission, activation, deposition, normalization/old snapshot, forces, frozen interface, pressure, new snapshot, ghosts/G2P, two-velocity wall response, state validation and acceptance. Record typed stage diagnostics separately. Freeze model settings and deterministic source/child ordinals; use counter-based Philox4x32-10 with the committed seed, accepted step and source/child key. Advance time only on commit and route growth between attempts.
- [ ] Run LiquidSubstep3D.* and both locked CPU presets. Commit feat: run transactional CPU liquid substeps.

## Task 6: Integrate a persistent real CUDA substep backend

**Consumes:** Tasks 2–5 and accepted CUDA arithmetic. **Produces:** CudaLiquidSubstepBackend3D::Create(capacities, device_ordinal), measured frame execution through the same scheduler stage contract, and immutable committed-device-epoch spans for Task 8. Keep qualified standalone CUDA APIs.

- [ ] Write actual Debug CUDA RED for CudaLiquidSubstep3D.PersistentSubstepMatchesCpu and CudaLiquidSubstep3D.ReplaysAfterCapacityGrowth, checking an analytic moving-wall/capillary frame, fixed-point exact parity, sorted IDs/topology, pressure and velocity tolerances inherited from existing operator tests, retry isolation, launch receipt and no in-attempt allocation. Add exact negative receipt and sanitizer inventory checks.
- [ ] Reserve device storage, pinned stage transfers, stream, events and graph resources outside attempts. Adapt the qualified device kernels to caller-owned ranges and active counts. Stage checked topology on host and use pinned CUB stable ordering on device with fixed capacities, explicit count/neutral padding, and queried temporary space. Use fixed order/reduction tree and explicit FP flags. Reuse pointers during replay; rebuild after checked growth. Include one valid empty-count run. Instrument device queries and fail on loss of CUDA; an unusable device must never execute CPU as substitution.
- [ ] Run four locked presets, real-device tests, named-pipe memcheck and SASS/PTX audit. Preserve existing 28 GPU/26 memcheck and add two mandatory cases. Commit feat: run liquid substeps in persistent CUDA arenas.

## Task 7: Version and atomically publish frame and checkpoint records

**Consumes:** Complete persistent state from Task 1. **Produces:** schemas/simulate/liquid_state.fbs, point_frame.fbs, frame_manifest.fbs, resolved_scene.fbs and a checked-in v1 conformance baseline. LiquidCache3D::PublishFrame, ReadCommittedFrame, ReadCheckpoint and ResolvedSceneFingerprint use explicit checked C++20 view types documented in liquid_cache_3d.h.

- [ ] Write RED for all-field round trip and canonical golden bytes/digest, default materialization including explicit default equals omitted, correct sensitivity to resolved setting/input bytes, malformed/truncated and changed IDs, digest mismatch, malicious paths, duplicate frame with unequal content, and interrupted publish preserving previous frame.
- [ ] Pin Conan flatbuffers host and tool requirement and blake3 host requirement. Generate headers through pinned flatc, verify conformance in CI with a negative incompatible schema, keep generated files ignored. Use BLAKE3 known-answer tests, bounded verification before accessing generated views, and ForceDefaults when encoding resolved fields. Persist exact upstream content hashes, deterministic execution ID and frame status. Keep heartbeat metadata separate. Durably write and rename content blobs and immutable JSON mirror before manifest publication. Use same-directory temp files, native file flush and atomic move, and cross-platform tests.
- [ ] Run LiquidCache3D.*, schema conformance, both locked CPU presets and the portable hosted matrix before merge. Commit feat: publish verified liquid frames and checkpoints.

## Task 8: Fence asynchronous snapshots by committed epoch

**Consumes:** Persistent versioned device state from Task 6 and cache writer from Task 7. **Produces:** LiquidSnapshotLease3D and CudaLiquidSnapshotQueue3D with BeginCopy(epoch, committed_spans), Poll, Finish and LastValidHostSnapshot. A lease owns pinned buffers until writer completion and a source fence prevents arena reuse until D2H completion.

- [ ] Write actual Debug CUDA RED for CudaLiquidSnapshot3D.DelayedCopyFencesSourceEpoch and CudaLiquidSnapshot3D.DeviceFailureUsesLastValidHostSnapshot. Intentionally delay the D2H stream behind a gate, try reuse and prove original bytes survive. Include two in-flight epochs, canceled unpublished copy, stale completion, explicit test injection of device failure and independent source/destination lifetime checks.
- [ ] Use pinned preallocated buffers and stream/event dependencies with no global synchronization during successful steady use. Check epoch on completion, preserve last validated host state on errors, and free/drain resources on safe teardown outside a substep. The cache writer sees only complete host snapshots. Tie source lease to every referenced arena, including carrier/topology and control spans.
- [ ] Run four locked presets, real hardware and named-pipe memcheck. Add two mandatory GPU/memcheck cases with exact receipts (32 GPU/30 memcheck total). Commit feat: fence committed CUDA snapshot epochs.

## Task 9: Deliver the Simulate CLI and replay protocol

**Consumes:** Scheduler, cache and snapshots from Tasks 5–8. **Produces:** fluidity_simulate with run --stage simulate, check, schema, step, replay --frame --substep, device-info, frame range and resume/trace options. Native CLI exposes versioned JSON Lines only on stdout, diagnostics on stderr and safe stdin cancel. Inspect uses the same FlatBuffers schema as runtime.

- [ ] Write RED with real process launches for valid defaults, unsupported roles, invalid scene, coherent heartbeat at least 1 Hz, cancellation, resume and replay. Assert frame/substep and epoch identities, process status transition from baking to crashed for orphaned PID/lease, and zero unexpected output on protocol stdout.
- [ ] Implement checked manifest and schema inspection, exact scene and executable identity validation, monotonic/coherent heartbeat without claiming in-flight steps, stable error domain/name/field/cause, cache-relative paths, and replay verification without rewriting content. Persist deterministic committed frame/checkpoint identity and publish operational status separately. CLI CPU and CUDA select explicit backends.
- [ ] Run tools/native/tests/test_simulate_replay.py against CPU and real qualified CUDA configurations, CTest entry inventory and portable checks. Commit feat: inspect and replay committed liquid frames.

## Task 10: Prove F9 and validate interruption, replay and fault isolation

**Consumes:** All earlier slices and Tasks 1–9. **Produces:** proofs/Fluidity/LiquidTransaction.lean and runtime/hash fault bridge. Use typed explicit persistent-field records including later-domain inventory as abstract typed fields. Add audited theorem targets for reject identity, finite sequence of rejects, once-on-accept time, deterministic replay under identical key, and independent adaptation input preservation.

- [ ] Record typed Lean RED before implementing statements; prove by transition case analysis and reject induction with the existing independent pinned verifier. Check axiom closure and negative target inventory controls.
- [ ] Write runtime RED around staged rejection hash checks if the bridge is missing. Inject each physical stage plus capacity retry, NaN, incompatible pressure, cancellation, process kill, device failure and slow D2H. Compare every persistent field including counters/IDs/RNG, topology, prior pressure and carrier history. Force a warm-start retry and a repeated capacity failure. Compare uninterrupted and resumed bytes for identical artifacts and separate cross-backend metrics. Exercise 50 crash/cancel/device-error trials and ensure all published frames can be read. The 1,000 cumulative default-scene and full job quality/performance gates also require the real slice 5 scenes.
- [ ] Run exact archive independent proof check, all four locked Windows presets, known receipt and skip negatives, real named-pipe sanitizer, explicit architecture and image audit, schema conformance, fixed-point and physical bridges. Measure memory reservations, live bytes, replay hashes, elapsed time, iteration counts and actual faults. Keep raw evidence ignored. Commit test: verify liquid transaction and replay guarantees.

## Final gate and integration

- [ ] Mark each task complete only after fresh independent review and resolved findings. Use one final whole-branch review and scoped follow-up, with an identifier and comment audit.
- [ ] Re-run complete validation on exact clean reviewed HEAD. Record actual dependency revisions, driver/runtime/API, all four preset results, local GPU event and host durations, proof inventory and remaining scene gates. No skipped GPU checks count.
- [ ] Push the feature branch and open a PR against main with readable results. Leave the PR open for review, preserving the branch, worktree and ignored evidence.
