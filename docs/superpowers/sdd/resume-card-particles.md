# Resume card: particles, building (written 2026-11-28, end of the first build session)

For a fresh coordinator session. Read this, then `how-to-run-a-module.md` and `dispatch-template.md` beside it,
then the ledger (its "Tasks" part holds every hand-back of the first session).

## Where things are

- Code: worktree `../FluidiTy-rebuild`, branch `rebuild`. Pushed BASE `74e1861a` (CI green). Local head `578d57bb`
  (Tasks 1, 2, 4 closed; Tasks 3, 5, 8 landed, their fixes open), then Task 9 31f34710, Task 7 b8f22802; NOT pushed: gate it in `~/fluidity-gate-check`
  before a push (the cuda step needs Docker, see Environment). Task 7 b8f22802 is on top.
- Other worktrees: `../FluidiTy-fix` (branch `fix-batch-1` from `578d57bb`, fix rounds), `../FluidiTy-proof`
  (branch `task-06`, commit `c7c31dd3`, not landed). Never build or check out `main`.
- Planning: this checkout, branch `rebuild-planning`. Plan: `docs/superpowers/plans/2026-11-14-rebuild-03-particles.md`.
  Design: `docs/superpowers/specs/modules/03-particles-design.md` (1,367 lines; briefs attach sections, never all).
- Ledger, briefs, reports, reviews (not in git): `../FluidiTy-rebuild/.superpowers/sdd/2026-11-14-rebuild-03-particles/`
  (`progress.md`: Rulings 1 to 10, every task's state, parked items).
- Gate: `uv run python tools/check_commit.py --speed quick|commit|full` in the worktree.

## State of the tasks

| Task | State |
|---|---|
| 1 contact in place | closed (de2e5dcd + fix), on rebuild |
| 2 paper: particles, grouping, try | closed (7468b8ff + fix 737654c8), on rebuild |
| 3 paper: split, merge, count | on rebuild (b6d07173 + fix 93aadde9); round 2 = 6b561fd9 on fix-batch-1, re-review next |
| 4 errors, fields | closed (c826e09a) |
| 5 formulas | on rebuild (c6b3624a); fix = 352e4582 on fix-batch-1, re-review next |
| 6 Lean proof, gate step `proof` | approved on branch task-06 (c7c31dd3); owes Minor 1..5 and a full commit gate |
| 7 comparison with main | closed (b8f22802, on rebuild); two quadratic values hand-worked, see ledger |
| 8 test grids and sets | on rebuild (2df4b4cc); fix = 7f61cb95 on fix-batch-1, re-review next |
| 9 arrays, limits, estimate | on rebuild (31f34710); review CHANGES_NEEDED (I1, I2 tests; m1..m10), fix queued; owes a clean gate |
| 10 to 24 | open |

## What to do first

1. Read the hand-backs that arrive (Task 9 review, Task 7, fix batch 1). If this session ended before they
   arrived (Task 7 and fix batch 1 did arrive and are recorded), look at the branches: `git -C ../FluidiTy-rebuild log --oneline 31f34710..` and `git -C
   ../FluidiTy-fix log --oneline 578d57bb..`, and the files `task-09-review.md`, `task-07-report.md`,
   `fix-batch-1-report.md`. Task 9's design-silent choices (buffer sizes, count rule counts without a total item)
   are listed in the ledger for the review to judge.
2. Fix batch 1 is done (`fix-batch-1-report.md`): 7f61cb95 (Task 8), 352e4582 (Task 5, also rewords the glossary
   entry of kMergeGapFloor: review it), 6b561fd9 (Task 3). Re-review: Opus for 352e4582 and 6b561fd9, Sonnet for
   7f61cb95. Then cherry-pick them in that order onto `rebuild` when its worktree is clean, copy notes with
   `git notes --ref=claude-session copy -f <old> <new>`.
3. Task 6: one Sonnet fix agent in `../FluidiTy-proof` for the review's Minor 1..5 and a full commit gate on a
   quiet machine; then cherry-pick onto `rebuild`.
4. One fresh Sonnet fix agent for Task 9 (I1, I2, small m's; `task-09-review.md`); Task 10 onward in plan order (Task 10 brief not written yet).
5. Gate the head in the clone, push `rebuild`, read the CI result once and report it.

## Rulings made this session (details in the ledger)

- 8: memory per particle from design 5.11 item 8 (one liquid 407 B; snow and springy 565 B; +16 N per ingredient).
- 9: merge waiting wording only; the rule is "the chunk's first particle's count >= 16".
- 10: every particles test grid has room `kTestGridCellBlocks = 144` (cascade grid A needs 16 + 64 + 64).
- Design sections to change at the as-built audit: 6.4 (1100-1101), 5.11 item 8 (copies = 1 for count rule
  counts), 5.6 (831, 834, 855-856); plan line 271 (room 144).

## Questions waiting for the owner (ask in the next round, with a recommendation)

1. `CommitTry` when the try holds removed or waiting particles not yet grouped in: the design does not say.
   Recommend: `kGroupingStale`. Needed before Task 12 (else a definition task).
2. The verb `HasField` (a design name) was added to `tools/check_rules_allowed_names.txt` by Task 4; its commit
   message did not name it for the owner's review.

## Environment

- colima was stopped by the owner's restart, which failed: `~/.colima/default/colima.yaml` says `disk: 60` but the
  VM disk is 100 GiB (no shrinking). Start with `colima start --disk 100` (or set disk: 100). The docker context
  is now `colima`. Until Docker answers, the gate's cuda step fails with "Docker is not running". Agents are told not to touch Docker and to record cuda as not run; gate in the clone before a push.
- Machine load was very high (load average up to 290) while Mathlib, two CMake builds and the Lean build ran:
  unit tests then fail only on the debug time limit. Rerun before suspecting a change.

## Lessons from this session

- Paper text: a source line that starts with `=`, `-`, `+` or `N.` becomes a heading or list item; the gate's
  paper step does not catch it. Every paper review renders the PDF and looks.
- Two agents cannot gate in one worktree (the gate refuses a dirty tree, and builds collide): a second building
  job always gets its own worktree; land its commits by cherry-pick when the main worktree is clean.
- A fresh worktree's first gate builds every dependency (slow).

## Rules that are easy to get wrong

- The repository is public: no trailer, no tool or model name, no process word in a commit message. The session
  link goes only into local git notes through `tools/add_session_note.py`; never push `refs/notes`.
- No pull request. Read CI after each push. GPU commits end with `GPU: compiled, not run`.
- At most two building jobs (one task plus one fix or tooling agent). A fresh agent for each fix round.
- Sonnet for implementers and for reviews of mechanical tasks; Opus for reviews of definition, numerics, device
  and integration tasks. Name the model at every dispatch.
