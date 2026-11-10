# Finite MAC Trace Reindexing Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development, test-driven-development and verification-before-completion. Record RED and GREEN in ignored evidence. Obtain task and branch reviews.

**Goal.** Prove the finite signed one-dimensional summation identity underlying the uniform mixed MAC trace.

**Architecture.** Keep the stencil identity separate from spline polynomial differentiation. Prove reindexing over n consecutive cells and n+1 faces with explicitly zero exterior endpoint weights. The existing numerical bridge covers the 3D tensor trace.

**Tech stack.** Lean 4.34.1 and pinned Mathlib d13f23b723b8a846827a245b89c10fc7d3f11612. Existing axiom audit and checker inventory in tools/verify_proofs.py.

**Spec.** Canonical design §§4.2 and 13. Method evidence and follow-up qualification. Base reviewed method head 2c220b955e12d807dbb9856b6384f2ec43a6cadc.

## Constraints and review focus

- Use exact reals and signed spatial indices. Include the empty n=0 case. The two explicit exterior zero weights identify the required finite halo and prevent treating truncated sums as complete support.
- No derivative-of-spline, cut, topology, time integration, floating-point or full solver claim follows. The h scale and transverse product application are outside the lemma.
- Use pinned compiler, verify the new theorem's closure against the existing allowlist, and add the theorem to the independent verifier inventory. Keep any unavailable independent checker execution separate from the direct Lean result.

## Task 1 — Signed finite reindexing

- [x] Add an exact typed consumer and a finite negative control in proofs/Fluidity/UniformTransfer.lean. Use RED for the missing theorem and an endpoint case that would fail without exterior zero. Include asymmetric signed data, n=0, and both endpoint hypotheses.
- [x] Add theorem finite_face_sum_eq_weighted_forward_difference. For origin lo : ℤ, n : ℕ and velocity and weight functions ℤ → ℝ, assume w(lo−1)=0 and w(lo+n)=0. Show the sum over j from 0 through n of u(lo+j)[w(lo+j−1)−w(lo+j)] equals the sum over i from 0 through n−1 of w(lo+i)[u(lo+i+1)−u(lo+i)]. Use a direct finite reindex or the existing range-sum identities.
- [x] Print its axiom closure in the existing module audit and list it in tools/verify_proofs.py MODULE_TARGETS. Describe what it proves and its boundary in proofs/README.md.
- [x] Check the changed module with pinned Lean and the verified Mathlib revision, then run the proof verifier tests. Keep the existing independent-kernel execution distinct and run it only if the pinned checker environment and dependency receipts are available. Record exact commands, outputs and any gate remaining. Commit and arrange independent task and whole-branch reviews.
