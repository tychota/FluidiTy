# Resume card: start of the particles module (written 2026-11-27)

For a fresh coordinator session. Read this, then `how-to-run-a-module.md` and `dispatch-template.md` beside it.

## Where things are

- Code: worktree `../FluidiTy-rebuild`, branch `rebuild` (orphan; pushed to `origin/rebuild`). Never build or check
  out `main`; read it with `git show main:<path>`.
- Planning: this checkout, branch `rebuild-planning`, pushed after each planning commit. Subagents never run git here.
- Ledgers and reports (not in git): `../FluidiTy-rebuild/.superpowers/sdd/<plan>/`. The grid's is
  `2026-11-14-rebuild-02-grid/progress.md` (Rulings 1 to 75).
- Gate: `uv run python tools/check_commit.py --speed quick|commit|full` in the code worktree. A clone for gating
  merged commits before a push: `~/fluidity-gate-check` (its origin is the local worktree).
- Second worktree `../FluidiTy-rebuild-tools` (branches `rebuild-tools`, `rebuild-task28`): merged, may be removed.
  About 46 backup branches `rebuild-*`: delete when the owner agrees the history is final.

## State of the modules

- Foundation (base), contact, grid: built. Grid tasks 1 to 28 done, final review done, later commits reviewed.
- GPU: every GPU test is compiled in the Linux container and never run. The owner runs them later
  (`docs/gpu-runs.md` in the code worktree). CI has a Windows CUDA compile job; check whether it is green yet.
- Owner's answers: `specs/modules/owner-answers.md` (A1 to A127). No question is open for the grid.

## Rules that are easy to get wrong

- The GitHub repository is public. Commit messages carry no trailer, no tool or model name, no process words
  (`tools/check_message.py` checks). The session link goes only into local git notes through
  `tools/add_session_note.py`; never push `refs/notes`; never write a session link into a file.
- A red CI does not block work during the rewrite, but read its result after every push.
- No pull requests unless asked. The naming audit of the old code is never committed.
- Ask the owner when unsure, in rounds of four questions with a recommendation; explain a hard one with a drawn
  page first (the new-cell question needed three tries in words and one drawing).

## Before planning particles (pre-flight)

1. Trim `specs/modules/03-particles-design.md` (about 1,400 lines, with history inside): delete replaced text; add a
   property and an `Evidence:` line to the sections that define a set (split and merge completeness 5.4 to 5.6,
   grouping 5.8, loaded groups 5.10).
2. Check the names its section 4 consumes against the built grid headers. Changed since it was written:
   `ComputeNewCellSizeLevel` and `FindNearestCommittedCell` are gone (`FindNewCellNeighbors` replaces them, used by
   the step, not by particles); `BuildRangeWithBorder` returns `std::expected`; an empty range is `kBadInput` except
   over a grid with no block.
3. Decide who owns the four merge and split settings (`merge_width_share`, `merge_wait_duration`,
   `min_merge_wait_steps`, `max_cell_splits_per_resize`): owner said with cell resize (A124).
4. Write the plan by `specs/modules/00-how-to-write-a-module-plan.md`: at most 900 lines, a `Brief:` block per task,
   a `definition` task before each set-defining rule, `Needs` and `Touches`, test sizes as numbers.
5. One pre-flight scan with three questions only: which set rules lack a definition task, which design fields no
   task produces, which consumed names are missing.

## Known risks for particles (from the grid)

- A position one step below a cell side lands in the next cell (`FindCellAt` at the exact double); test positions
  on a side and one step below it.
- "Border blocks are not copied back" rests on the grid's border rule; values are bound again at every load and
  never read through another piece (A124).
- Positions far from the origin lose precision (4e-12 on a fitted value at 1,300 m with 3 m cells): accepted.
- Implementers cannot wait on a gate the harness moved to the background: tell them to end the turn and continue
  when woken, and allow amending their own commit until its gate passes.
