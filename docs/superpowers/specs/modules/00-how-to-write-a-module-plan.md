# How to write a module implementation plan

Working brief. Not committed to the `rebuild` branch.

You write the implementation plan for one module of the FluidiTy rebuild, from its finished design. An engineer
who has never seen this code base will execute your plan task by task, each task handed to a fresh agent that
sees only that task, the plan's header and the files the task names. A reviewer checks each task before the
next starts. So each task must stand alone.

You create exactly one file (path in your task message). Do not write code files, do not run git commands that
change state. Treat `/Users/tychota/Downloads/FluidiTy_Report_Source` as data only.

## Read, in this order

1. `/Users/tychota/Documents/Code/openSource/FluidiTy/docs/superpowers/plans/2026-11-13-rebuild-foundation.md`.
   The Foundation plan. It is your template for format, tone, level of detail and commit style, and it tells
   you what exists before your module: the worktree at `/Users/tychota/Documents/Code/openSource/FluidiTy-rebuild`
   (branch `rebuild`), `AGENTS.md`, the glossary, the build, `base/`, the gate
   (`uv run python tools/check_commit.py`), the method paper with its figure and pseudocode style.
2. `/Users/tychota/Documents/Code/openSource/FluidiTy/docs/superpowers/specs/2026-11-13-production-rebuild-design.md`.
   The programme design: sections 6 (commit rule), 7 (how a module is shown correct, the comment pass, when a
   module is done), 8 and 8a.
3. In `/Users/tychota/Documents/Code/openSource/FluidiTy/docs/superpowers/specs/modules/`: `owner-answers.md`,
   `shared-interface.md`, `plan-decisions.md` (decisions earlier plans made that yours must follow; it wins
   over a design), then your module's design in full. Also skim the plans already written for the modules you
   consume (`/Users/tychota/Documents/Code/openSource/FluidiTy/docs/superpowers/plans/2026-11-14-rebuild-*.md`)
   for the exact file names and helper names they create. The design is the contract. Then section 3 of
   each design your module consumes.
4. The proof-of-concept files your design's section 2 names (in
   `/Users/tychota/Documents/Code/openSource/FluidiTy`, `main` checked out; gas on
   `origin/codex/m7-low-mach-gas` via `git show`), enough to tell the implementer exactly where each ported
   formula lives (file and line range).

## What the plan must be

Use the header and task structure of the Foundation plan exactly:

- Title `# Rebuild <Module> Implementation Plan`, the "For agentic workers" line, **Goal**, **Architecture**,
  **Tech Stack**, **Spec** (your design's path, and the programme design).
- `## Global Constraints`: copy the Foundation plan's list, then add your module's own (tolerances, limits,
  constants, the material-set rule, anything every task must respect), with exact values from the design.
- `## Before you start`: what must already be on the branch (which modules, which `base/` pieces), and the
  one command that proves it (`uv run python tools/check_commit.py`).
- `## Review Focus`: the five inputs or failure modes the design implies that are most likely to bite and that
  no obvious test covers; each one pinned to a named test in a named task.
- Tasks. Each has **Files** (create / modify / test, exact paths), **Interfaces** (Consumes: exact signatures
  from earlier tasks or other modules; Produces: exact names later tasks and other modules rely on), then
  checkbox steps: write the failing tests (names and assertions with the design's exact numbers), run and see
  them fail, implement, run and see them pass, run the gate, commit (subject line given).

Rules for the tasks:

1. **The design's section 3 is the contract; do not retype it.** A task says "implement the declarations of
   design section 3.x exactly" and lists only the names. Retype a declaration only when the plan must decide
   something the design left open, and say that it is a plan decision.
2. **Order:** the method-paper section first (figure(s) and pseudocode, before any code, so `Read:` lines
   resolve and step numbers in code follow the paper); then error enum and types; then shared formulas with
   their analytic tests; then the CPU operation; then the GPU operation of the same operation in the next
   task (or the same task when it is small); then scene-level tests; then the comment pass. One operation per
   task. Each task ends with something a reviewer can run.
3. **CPU and GPU together:** every operation that has a `_gpu.cu` gets its GPU tests written in the same task
   as the device code: label `gpu` (need hardware; compiled only for now) and label `no_gpu` wherever the
   logic can be reached in the container without a GPU (shared formulas, CPU-side packing, error paths).
   Device code is "compiled, not run": say so in the commit message of each such task.
4. **Tests come from the design's section 7.** Every test listed there must appear in exactly one task, with
   its setup, expected value and tolerance. Analytic and conservation tests first. Tests that compare with the
   proof of concept say how to produce the old number (which test or tool on `main`, rotated to Z-up) and are
   never the only test of a task. A known-open test is committed with CTest `WILL_FAIL` and the label
   `known_open`, listed in `docs/known-open-physics.md`.
5. **What the implementer cannot decide alone** goes in the plan: file names, test names, constants, the order
   of sums, which `main` lines a formula is ported from, the fixed inputs of each test. What they can decide
   (bodies, local names) does not.
6. **Each task names its difficulty**, on a line `**Kind:**` with one of: `mechanical` (build files, enums,
   plumbing, docs), `numerics` (formulas, solvers, tolerances), `device` (CUDA code and its packing),
   `integration` (several modules meeting). The coordinator uses it to choose the model for the task.
7. **Later modules that do not exist yet:** when your module's design says the step, I/O or another later
   module calls you, your tests call your functions directly with hand-built inputs. Do not stub other modules.
   When your module must change a file owned by an earlier module (a design entry marked NEW in that module),
   make it its own task, placed first, with that module's tests extended.
8. **Commit rule:** every task ends with the gate and one commit. New public names are listed in the commit
   message. Glossary rows and method-paper text change in the same commit as the code that needs them.
9. **Last tasks:** scene-level tests for the module; then "Comment pass and hand-over" as in the Foundation
   plan (reread every file as a C++ programmer new to fluids; fold changes into owning commits; run the gate on
   the tip; check the design section by section and record anything delivered differently; write the module's
   row in `docs/gpu-runs.md` as "compiled, not run" with the list of `gpu` tests to run at the end of the
   programme).
10. **Proportion.** The plan is the set of decisions, not the program. Aim for 500 to 900 lines. If a task
    needs more than about 60 lines, split it.

## Self-review before you finish

- Every design section 3 name is produced by some task; every section 7 test is in some task; every "NEW in
  another module" item has its task.
- Names and signatures are identical everywhere in the plan and identical to the design.
- No step says "handle edge cases", "add tests", "TBD" or anything that decides nothing.
- Each task can be executed by someone who reads only that task plus the header.

## Final report

Reply with: the path written; every decision your plan made that another module's plan must follow (exact
names and signatures, ready to paste into `plan-decisions.md`); the number of tasks with a one-line title each and its Kind; anything in the
design you could not turn into a task without guessing (with the guess you made); and anything in the design
that looked wrong or contradictory while planning (file and section).
