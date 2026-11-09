# Reuse Uniform Liquid Wall Samples Implementation Plan

> Use superpowers:subagent-driven-development. Require an independent reviewer, recorded RED and GREEN evidence, and a separate verified commit for each correction.

**Goal**

Remove redundant full-wall sampling in bounded uniform liquid gathers and measure how much work remains in the frozen acceptance scenes.

**Architecture**

The scheduler uses old and new prescribed wall velocities at distinct time levels. Each pair depends on the fixed signed MAC grid, the two obstacle poses, and the particle wall side. Keep the pair alive across consecutive particles on the same side. Continue to gather and sample the liquid field in the existing per-particle order.

**Tech stack**

C++20, existing CUDA geometry and transfer stages, CMake Ninja and Conan 2, uv, and the locked Windows validation harness.

**Spec**

[FluidiTy canonical design](../specs/2026-11-07-fluidity-design.md) and [M1 acceptance and throughput plan](2026-11-09-real-3d-liquid-acceptance-and-throughput.md), especially Tasks 3A, 4 and 6. This companion correction preserves the original qualification plan, settings and gates.

## Evidence and constraints

- Frozen Release CPU piston frame 1 took 64.365 s, with 63.995 s in gather/contact. RTX hydrostatic frame 1 took 56.9897 s, with 2,575 gather/contact CUDA intervals and 55.719 s across those intervals. These are actual baseline probes, separate from end-to-end physical acceptance.
- The scheduler currently samples both complete owned-face wall arrays for every particle. It also rebuilds or stages anchors and histories inside each scalar gather. Ghost continuation scans owned anchors and performs QR, and world-space scalar field evaluation also remains. The aggregate phase timers do not attribute these costs individually.
- The independent builder check found that a multi-particle CUDA gather consolidates work in one single-thread dispatch. Moving it before scalar field samples also changes which failure and diagnostics can win. This correction keeps those operation boundaries and their status ordering.
- Preserve the solver, physical limits, fingerprints, schemas, precision, accumulation order, per-particle result and accepted diagnostics. Add no attempt allocations, CUDA kernels, global fences or new configuration.
- Preserve raw older preflight and full/partial scene data. Before rebuild, verify exact executable/argv and stop only controlled superseded test jobs that block rebuild or compete for the hardware. Retain resulting partial files and explain the stop. Never overlap GPU workload with sanitizer.
- Follow AGENTS.md. Use exact Windows/MSVC/Conan/uv/CUDA versions, architecture 120-real and 120-virtual, real RTX 5060 Ti, and isolated Windows named-pipe Compute Sanitizer. No fallback, unsupported compiler override or firewall changes.

## Task 1 — Reuse wall-only samples for consecutive particles on the same side

**Files**

Modify native/src/liquid_substep_3d.cc and the sampling contract in native/include/fluidity/host/liquid_substep_backend_3d.h if necessary. Extend native/tests/test_liquid_substep_3d.cc and native/tests/test_cuda_liquid_substep_3d.cc. Reuse existing staging, wall-sampling, gather and scalar world-field APIs.

- [ ] Record expected RED before production edits using a counted backend wrapper. For N consecutive particles on one side, require exactly two wall-only sample calls, one old and one new, and N unchanged ordered gather/field pairs. For repeated and alternating signed slab sides check old/new poses, exact side and canonical particle order. Empty input performs no wall-only sample calls. Filter out full-face tag and other non-gather calls.
- [ ] Check ownership, history contents and failure ordering. No later gather or world-field operation can precede an earlier one. Inject old-wall refresh failure and new-wall refresh failure separately after an earlier successful side run. Require no gather/field call for the failing particle, unchanged committed host/device bytes, correct prior diagnostic prefix and fresh sampling on retry/replay. Preserve status ordering for retained operations. Removing duplicate calls necessarily removes faults tied solely to their invocation counts. Reinitialize the remembered side on each attempt and do not reuse across poses or signed side changes. Verify unchanged PIC fallback counts, thin-slab incident-wall behavior, retained state on replay and no attempt allocations.
- [ ] Store the current wall side locally for the gather phase. Refresh the existing old/new wall-only scratch arrays only for the first particle or a changed side. Mark the new side ready after both samples succeed. Document that wall values depend only on the supplied grid, obstacle pose and side, and are caller-owned until overwritten. The later gather and scalar field callbacks do not mutate those wall arrays. Use no new large state, no particle sorting, and no changes to the gather or liquid field implementation.
- [ ] Record measured operator call inventories and wall times before and after on tiny and frozen hydro/piston first-frame probes, where available per backend. Attribute work using actual host/GPU traces and the counted callback wrappers. Keep exact FLST/FLPT and accepted numerical receipt comparison per preset/backend and authenticate each executable. Report remaining gather/field cost separately. When call deduplication still leaves prohibitive runtime, capture an operation-specific profile before planning any additional correction.
- [ ] Run all four locked presets, actual real-device side and failure controls, and isolated Release named-pipe sanitizer. Check the explicit image target, formatting, commit scope and exact source digest. Request independent review, then resume physical acceptance at the original frozen setups. A performance result cannot satisfy a physical gate.

## Review focus

- Repeated nonconsecutive wall sides keep canonical order and require a fresh pair of wall samples. Thin slab particles can share a face while requiring opposite incident normals.
- Old/new samples belong to different poses. A failed refresh must not mark a partial pair ready.
- A new attempt, retry or growth replay cannot retain an earlier wall side or its samples.
- The borrowed wall arrays are disjoint from the gather work bank and survive unrelated scalar field sampling. Result and failure order remain per particle.
- Measure the reduction and the remaining cost independently. A count of avoided calls does not establish improved long-job throughput.
