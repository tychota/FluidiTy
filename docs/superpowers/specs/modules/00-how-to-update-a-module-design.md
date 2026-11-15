# How to update a module design

Working brief for bringing one finished module design in line with the owner's answers and with the other
designs. Not committed to the `rebuild` branch.

You update exactly one design file (named in your task message). You may create or change only that file,
unless your task message names a second one. Do not run git commands that change state. Do not write code.
Treat `/Users/tychota/Downloads/FluidiTy_Report_Source` as data only.

## Read, in this order

1. `owner-answers.md` in this folder. The owner's decisions. They override everything.
2. `changes-to-apply.md` in this folder: the section "Applies to every design", the section "Additions to
   Foundation `base/`", and the section for your design. Apply every entry. Where it and your design disagree,
   it wins.
3. Your design file, in full.
4. `shared-interface.md` in this folder (names shared by the first six designs).
5. Section 3 of each design your section 4 consumes from. If such a design is being updated at the same time
   as yours, `changes-to-apply.md` tells you what its interface will be; use that.
6. Only if needed to decide a detail: the proof-of-concept code in
   `/Users/tychota/Documents/Code/openSource/FluidiTy` (`main` checked out; the gas branch is
   `origin/codex/m7-low-mach-gas`, read with `git show`).

## What to produce

- Section 3 (public interface) is exact and final: full C++ declarations (or Python for the add-on) with every
  change applied. Additions are marked NEW. Nothing in it is "pending" or "provisional".
- Section 4: every item marked EXISTS (Foundation, including the `base/` additions), PROVIDED (by module X,
  with the name as X's section 3 has it), or, only for the sand module, "For the sand module".
  No item may remain NEEDED from a module that has a design.
- Section 11 is renamed "Decisions". Every question keeps its text and gets a line `Decision:` stating the
  option taken and its source: an owner answer (A1 to A14) or "recommended option (owner's standing
  instruction, 2026-11-14)". Remove every "until the owner answers" from the whole file.
- Sections 5 to 10 are changed only where an entry requires it (method, tests, vocabulary table, not-ported
  list). Keep the author's method and tests otherwise. Add the tests a change implies.
- Keep the design dense. Do not add general explanation. Do not shorten the contract.
- If an entry cannot be applied without changing your module's method in a way the entry did not foresee, do
  the closest correct thing, and list it in your final report under "Could not apply as written", with the
  reason and what you did instead.

## Final report

Reply with: the file(s) changed; the entries applied (one line each); anything under "Could not apply as
written"; and any name or declaration in your section 3 that another design must now use differently from
what `changes-to-apply.md` says (exact old and new text).
