# Resume card: particles, ready to build (written 2026-11-27, after the pre-flight)

For a fresh coordinator session. Read this, then `how-to-run-a-module.md` and `dispatch-template.md` beside it.

## Where things are

- Code: worktree `../FluidiTy-rebuild`, branch `rebuild` (pushed). BASE `74e1861a`, CI green. Never build or check
  out `main`; read it with `git show main:<path>`.
- Planning: this checkout, branch `rebuild-planning`, pushed after each planning commit. Subagents never run git here.
- Plan: `docs/superpowers/plans/2026-11-14-rebuild-03-particles.md` (24 tasks, 726 lines), its revisions file beside
  it. Design: `docs/superpowers/specs/modules/03-particles-design.md` (1,349 lines; 5.11 holds the rules every task
  follows).
- Ledger, briefs and reports (not in git): `../FluidiTy-rebuild/.superpowers/sdd/2026-11-14-rebuild-03-particles/`
  (`progress.md`: Rulings 1 to 7).
- Gate: `uv run python tools/check_commit.py --speed quick|commit|full` in the code worktree. Clone for gating
  merged commits before a push: `~/fluidity-gate-check`.

## Open tasks

All 24. None dispatched. Order to start: Task 1 (contact in place), then 2, 3 (paper), 4, then 5 and 8 may run at
once (disjoint `Touches`), then as `Needs` allows. Tasks 1, 2 and 3 all touch the method paper: one at a time.

## Building a brief

For each task copy into `task-NN-brief.md`: the task's text, the plan's Global Constraints, and every design section
its `Brief:` names, in full. Check each `Consumes, checked:` name against the code at dispatch. Then the dispatch
template with BASE, the brief path, the report path, the model id and the session link.

## Models (owner, 2026-11-27)

Sonnet for every implementer and for reviews of `mechanical` tasks and scene checks. Opus only for reviews of
`definition`, `numerics`, `device` and `integration` tasks. No fork of the coordinator's thread. Name the model at
every dispatch. A fresh agent for each fix round (findings and diff only).

## Rules that are easy to get wrong

- The repository is public: no trailer, no tool or model name, no process word in a commit message. The session
  link goes only into local git notes through `tools/add_session_note.py`; never push `refs/notes`; never write a
  session link into a file.
- No pull request. A red CI does not block work, but read it after each push.
- GPU code is compiled in the container only; every such commit ends with `GPU: compiled, not run`.
- Ask the owner when unsure, in rounds of four with a recommendation; a hard question gets a drawn page.
- Implementers cannot wait on a gate the harness moved to the background: tell them to end the turn and continue
  when woken, and allow amending their own commit until its gate passes.

## Questions waiting for the owner

None. Answers A128 to A131 are applied. Not asked, ruled by the coordinator (ledger Rulings 1 to 7).

## Known risks

- Task 6 (Lean): the proof may not close in one run, and Mathlib makes the gate's `proof` step and CI slower. If the
  implementer stops BLOCKED, ask the owner before spending more.
- Task 20: the per-cell visit's stack on the GPU is measured there, not known yet.
- A position one step below a cell side lands in the next cell: Tasks 8, 10 and 11 carry such positions.
- The design is above the 900 lines the guide asks for; briefs attach sections, never the whole file.
- Other designs still describe the removed names (`changes-to-apply.md`, "From the particles pre-flight").
