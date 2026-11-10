# Piston Capture Provenance Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development, test-driven-development and verification-before-completion. Use the existing piston worktree and distinct task and branch reviews. Keep receipts in the ignored SDD ledger.

**Goal.** Qualify frozen piston receipts only when the producing solver, fixture and actual execution device are bound to the exact captured inputs and outputs.

**Architecture.** Report a stable CUDA execution identity from the actual run's existing compatibility admission sample. Pin the solver and fixture bytes, bind the decoded scene's solver digest to the running solver's schema digest, and require a separately pinned manifest for offline inputs. Compare the stable device identity across the primary and refined scenes.

**Tech stack.** C++20, CUDA SM120, existing BLAKE3 implementation, Python 3.13.18, uv 0.13.7 and standard-library SHA-256.

**Spec.** Canonical design §§2.3, 6.1, 11.2 and 14.1. Scene plan 2026-11-09-real-3d-liquid-acceptance-and-throughput.md, scene authority and Task 4. Existing receipt-validation branch must be the exact base.

## Constraints and review focus

- Preserve the 1% physical and 0.25% refinement limits, the existing FLCUDA1 bytes and cache semantics, and the existing non-frozen harness interface. Keep capture verification read-only with a fresh evidence directory.
- Capture the CUDA UUID, compute capability and driver/runtime identity from the same admission query for the actual job. Driver/model equality alone cannot establish physical-device equality. Keep raw UUIDs and personal paths out of commits and receipts intended for publication.
- SHA-256 pins are supplied from a trusted build or launch channel. A digest drawn solely from the capture being verified is not an independent trust anchor. Bind exact regular files within the capture directory, avoid symlink escapes, and reject duplicates or malformed identities. This is local evidence integrity under trusted binaries and a controlled launch directory.
- Hash evidence before physical evaluation. A qualified temporal comparison requires the same solver and fixture pins, matching seed digest, distinct scene/execution IDs and matching stable CUDA execution identity for the actual runs. CPU inputs must omit CUDA identity. Do not infer any GPU result from mocks.
- Legacy captures without a trustworthy manifest or actual-run device identity remain diagnostic. Handle malformed temporal references with an ordinary validation error.
- Exclusive RTX ownership and real SM120 remain requirements for the GPU verification step. The current archival capture must finish undisturbed.

## Task 1 — Bind the terminal result to the actual CUDA admission

- [x] Read native/cli/simulate_cuda_compatibility.{h,cc}, native/cli/simulate_execution.cc, the protocol tests and the cache compatibility tests. Test that equivalent device/runtime fields produce a stable digest across two scene fingerprints while different UUID or exact runtime fields change it. Test that FLCUDA1 remains byte-identical for the same full input. Use deterministic data and a testable digest helper without adding a device launch to the unit test.
- [x] Have the CUDA compatibility code construct a small named result containing the unchanged FLCUDA1 record and a lowercase hex digest of the device/runtime fields under a different domain. Consume both from the same call in the measured CUDA run. Report the stable digest on the actual successful measured CUDA result with the actual device ordinal and existing compiled architecture. Ordinary CPU results omit those fields. Keep existing run/replay/cache admission checks intact.
- [ ] Run appropriate scoped portable checks and clang-format 19.1 on changed native files. Compile explicit SM120 when the supported toolchain is available. Once the archive releases the device, run the existing real-device compatibility/replay check on actual SM120 and record exact build and driver receipts. Keep these verification categories separate. Commit and send the exact range for independent review.

## Task 2 — Verify the producing binaries and captured bytes

- [ ] In tools/native/tests/run_liquid_acceptance.py and a focused companion test, write RED cases for a wrong solver or fixture SHA-256, a wrong schema/decoded-scene solver digest, missing or malformed manifest, untrusted manifest pin, altered scene/seed/result or compatibility bytes, captured-reader mutation, wrong or absent actual-run identity and malformed primary reference shape. Exercise same-device different-fingerprint CUDA pairs and the CPU control with process doubles.
- [ ] Reuse the existing file SHA-256, strict receipt JSON and capture compatibility validation patterns. Require solver and fixture pins for frozen cases. Call the pinned solver schema without CUDA execution and compare the build digest with both independently decoded scene and authored identity. Check the pinned binary hashes before launch and on verification completion. Publish a versioned manifest from the authorized launch path binding case, backend, ordered input and receipt hashes, schema identity, and the exact captured CUDA compatibility record when applicable. An interrupted run never emits a complete launch manifest.
- [ ] Require an independent expected manifest SHA-256 for offline verification and an independent measurements SHA-256 for an externally supplied temporal primary summary. Validate both manifests and decoded identities before scoring. Prefer shared validation between live and offline cases. Preserve failed physical results as failed typed outcomes.
- [ ] Run pinned offline/locked focused and reference/tool suites. Keep native and GPU verification gated by the corresponding exact artifact and device availability. Record RED/GREEN logs, commit and arrange independent task review. Obtain whole-branch review before a stacked PR.
