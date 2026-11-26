# Dispatch template for implementers

The fixed text every implementer dispatch carries. The coordinator fills the placeholders and adds the brief path.

```text
Worktree: <worktree>
BASE: <BASE>
Branch: <branch>
Brief: <brief path>
Report: <report path>
Model id: <model id>
Session link: <session link>

Work only in the worktree above, on the branch above.
Never run `git stash`, `git reset`, `git rebase`, or `git checkout` of another branch. Never run git in the
planning checkout. Never push.
Start no background process and no wait loop. Run the gate in the foreground and wait for it.
Start no subagent.

Gate speeds: `quick` while you work, `commit` once before the commit, `full` only if the brief says so.
Tests go where the brief says, at the sizes the brief gives. Add no time caps. If the sizes cannot fit, stop and
report it.

Commit rule:
- The subject and every body line are at most 72 columns.
- No trailer, no tool name, no model name, no process words.
- After each commit run:
  uv run python tools/add_session_note.py --session <session link> --models <model id>

Words to keep out of names, comments and commit messages: refuse, band, tank, tile, slab, rows, family, margin,
nudge, sample, gap, kernel, qualify, validate, provider, defect, and the suffix `View`. Use "level" only as
"size level".

Report contract. End with one of:
- DONE
- DONE_WITH_CONCERNS
- BLOCKED, with a counterexample (grid, place, expected and actual set)
Then give the commit hash, a test summary, and your concerns.
```
