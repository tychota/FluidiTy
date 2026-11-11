# M3 Adaptive Width Policy Implementation Plan

**Goal** — Qualify immediate physical refinement and accepted-time 2:1 coarsening hysteresis independently of topology publication.

- [x] Add tests for physical/camera bounds, equality at the 0.8 coarsening margin, accepted-time accumulation and veto resets.
- [x] Exercise cavity, sheet, gap, contact, masked support and incomplete halo checks. Verify invalid controls fail without producing a decision.
- [x] Run the test target with an error-only stub, then implement the checked pure decision routine.
- [x] Run native tests and formatting. Record the adapter requirements for a production sibling/halo planner and transactional state.

## Evidence

All four tests failed against the error-only stub and passed with the policy. The MSVC Debug native executable passed 534 tests and the tracked native formatting check passed with clang-format 19.1.5. The tests cover immediate refinement, an inclusive 0.8 parent-width margin, accepted time only, and resets after missing support or unresolved cavities, sheets and gaps.

The caller must produce the actual parent/support/movement halo and publish the proposed hysteresis state only with a successful topology transaction.
