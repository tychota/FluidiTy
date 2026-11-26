# How to run a module (coordinator rules)

For the coordinator session that runs one module from its plan. The numbers in rule 10 come from the usage analysis
of the grid module, 2026-11-26.

## Sessions

1. Start a fresh coordinator session for each module. Start another whenever the median context passes 250k.
2. A new session works from a resume card of at most 150 lines. The card holds: BASE, the open tasks, the rulings
   made so far, the questions waiting for the owner, and the worktrees in use.

## Waiting and checks

3. No paid waiting. Agents leave no background process. A notification for an agent whose hand-back was already
   handled gets no status message. The coordinator never polls.
4. After every push, read the CI result for that head once and report it. During the rewrite a red CI does not
   block work (owner, 2026-11-26), and the CI workflow itself may change.
5. A commit an implementer already passed through the commit-speed gate in a clean tree is gated again in the
   clone only when a merge, fold or cherry-pick produced it, or before a push.

## Jobs

6. At most two building jobs: one task, plus one fix or tooling agent. Reviews do not count.
7. Pipeline task N+1 while task N is in review.
8. Run tasks in parallel only when their `Touches` sets are disjoint.
9. Never merge while the main worktree holds staged work.

## Models

10. Implementers: Sonnet for every Kind. All 12 Sonnet implementers in the grid's build phase landed first try, and
    the one that met a design fault stopped BLOCKED correctly.
11. A correction to a design rule is a `definition` task, not an Opus implementer on a provisional ruling. Sonnet
    writes the reference and the test. Opus reviews it.
12. Reviews: Opus for `definition`, `numerics`, `device` and `integration` tasks. Sonnet for `mechanical` tasks and
    scene checks.
13. Final review: the most capable model.
14. Coordinator: Opus if the harness allows. A ruling that changes a rule goes to a `definition` task, not to the
    coordinator's judgment.

## Module end

15. Keep the final review.
16. Keep the as-built audit. Each ruling adds a ledger line "design section to change", so the audit reads a list
    and does not search.
17. Shrink the Minor batches with plan rule 11. Aim for fewer than 25 parked items per module.
18. Move comment truth into the task. The review checks each number and claim in the comments its diff adds
    (bytes, counts, bit positions, "every", "never", "always", "one launch").
19. Then run one comment pass at the module end, only over files changed by three or more tasks.
20. Move the question "which plausible wrong implementation do the tests accept?" into the plan (rule 6) and the
    design (section 7, formula values).
21. Drop the comment pass over `testing/` and tests.
