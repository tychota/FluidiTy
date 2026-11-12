# FluidiTy M6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Qualify elastic work, cutting, continuous contact and rigid integration against the FluidiTy design.

**Architecture:** Establish the control on owned MAC supports using true basis derivatives. Extend the same operator with compatible deposition and a complete ghost gather. Persist only accepted state and use shared contact and body identities at the production boundary.

**Tech Stack:** C++20, Abseil, GoogleTest, CMake and the existing checkpoint tooling.

**Spec:** docs/superpowers/specs/2026-11-07-fluidity-design.md, Q-SOLID and sections 5, 8, 11, 14 and 15.

## Global constraints

- SI units, signed lattice coordinates and canonical traversal.
- Q-SOLID scaled relative/absolute tolerance of 1e-6.
- Distinct body IDs and cut-side affinity. Rejected attempts preserve committed state.
- Continuous motion certification is required for moving obstacles.
- Local commits from upstream 549fb3f. Publishing and merging are reserved.

## Approach

Reuse the native MAC derivative gather as the control. A separate elastic grid would require proving parity for ownership and derivatives. Extending liquid-only checkpoint layout immediately would conflict with the concurrent M5 contact work. Qualify standalone mechanisms first and promote integration with a shared schema contract.

## Review focus

- A non-affine velocity field must distinguish the derivative from the APIC fit.
- Missing owners, invalid constitutive state and rank-deficient compatible masks reject or take an explicitly specified fallback.
- A rejected cutter traversal and restart preserve side identities and allocator history.
- A thin moving obstacle can cross between endpoint samples.
- Asymmetric inertia and off-center reactions exercise torque and the world gyroscopic term.

## Tasks

1. **Uncut control.** In native/tests/test_transfers_3d.cc use the translated eight-block corner and manufactured quadratic face fields. Check native GatherMac3D against analytic gradients. Add checked stress deposition on the same owned samples, with a Neo-Hookean stress oracle and nonzero face power. Compare grid power against the negative reference-volume stress contraction with the gathered gradient. Run the targeted CTest and commit.
2. **Cut budget and side identity.** Compute separate deposited and gathered gradients and report dt times the stress contraction of their difference. Check complementary masks, rank failure, momentum, torque and actual blade work. Advance stable cut-side identity only on accepted blade traversal. Exercise rejection and restart with deterministic replay.
3. **Continuous contact.** Certify sweep for the supported obstacle geometry, including a thin moving crossing between endpoints, grazing, and exhausted certification. Apply the response to physical and advection velocity using incoming normal speed and shared M5 impulse semantics. Exercise rejected attempts.
4. **Rigid integration.** Test positive mass and inertia, force, off-center torque, world angular velocity, normalized orientation, and sweep consistent with the pose update. Verify checkpoint state/history and rejected attempt preservation. Run isolated contact and rigid fixtures before attempting floating acceptance.
5. **Promotion.** Run the floating scene only with production pressure traction, body motion feedback, exactly-once reactions, lag and added-mass diagnostics, and transactional replay. Record measured gate evidence and any missing dependency. Run the native validation and formatting gates, review the branch and retain local commits.
