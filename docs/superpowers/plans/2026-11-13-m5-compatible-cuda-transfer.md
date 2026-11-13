# M5 compatible CUDA transfer implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task.

**Goal** — Shorten the compatible-floor Q-SAND CUDA runs while preserving their numerical and failure semantics.

**Architecture** — Reuse the existing parallel deposit and per-particle gather kernels in the owning CUDA boundary adapters used by the dry solver. Keep the public owning results and caller-side publication rules.

**Tech stack** — C++20, CUDA SM120, CTest, clang-format 19.1.

**Spec** — [FluidiTy design](../specs/2026-11-07-fluidity-design.md), Q-SOLID and Q-SAND.

## Constraints

Preserve the integer quantization, signed impulse accounting, deterministic first error, output publication on success, and the shared liquid and dry API contracts. The acceleration work cannot supply the missing Q-SAND physical pass.

## Deposit

- [ ] Locate the existing CUDA boundary deposit fixtures and establish serial reference outputs for valid multi-particle stencils and overlapping deposits.
- [ ] In native/cuda/liquid_boundary_transfer_3d_host.cc, dispatch the owning deposit through LaunchLiquidBoundaryDeposit3D. Initialize face results and the ordered error sentinel before launch as in the arena adapter.
- [ ] Reproduce the serial quanta and stencil results exactly. Check empty input, invalid contribution, ordered failures, and bounded accumulator behavior. Run the CUDA operator tests.

## Gather

- [ ] Establish serial reference output for multi-particle compatible gathers with both regular and reconstructed support.
- [ ] Dispatch the owning gather through LaunchLiquidBoundaryGatherRun3D and inspect each completed report in particle order before building the owning result. Confirm that every public gather field is available in the parallel result layout before switching the caller.
- [ ] Compare the full outputs and rejection status against the serial fixture. Run both default M5 dry scenes and the relevant shared liquid transfer tests.

## Qualification

- [ ] Time the same single aligned compatible-floor case before and after the change. Use the original CUDA executable as the baseline where available.
- [ ] Run both backends with --q-sand-compatible-floor-long-settle all aligned 6 and compare all four angles, radii, last-second drift and work receipts.
- [ ] Record the measurements and share the transfer change with M6 review before merging any shared implementation.

## Review focus

Check ordered errors across components, launch behavior for zero particles, exact integer sums under different scheduling, complete population of owning gather fields, and rollback after a late failed particle report.
