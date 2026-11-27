# Questions waiting for the owner (collected 2026-11-26, after the grid module)

Not committed to the `rebuild` branch. Ordered by how much later work depends on the answer. Each has the option
taken meanwhile, so work does not wait; (rec) marks the recommended option. Grid design section 11 "Open" and
section 12 hold the details.

## Before the step and cell resize plans

1. **New cells and the size-step rule.** `ComputeNewCellSizeLevel` gives a new cell the size level of the nearest
   committed cell plus one, but its search looks at 27 positions one biggest cell width apart, so the cell it finds
   may not touch the new cell and a touching smaller cell can be missed: the new list can break the size-step rule and
   the try fails. No function closes the rule over a list of new cells (grid Rulings 46, 48).
   Options: (a) the grid's search looks at nearer distances first (positions at each size level's width), so a
   touching small cell is found; (b) the step owns a closure over its new list, with a GPU form for its marking pass
   (rec: (b), since no search of fixed size can find every touching cell, and the step already owns the list);
   (c) both. Taken meanwhile: the step's design states it as an open point; nothing is built.
   Answered 2026-11-27: fill from the top; the grid has FindNewCellNeighbors, the step owns the loop (A118).
2. **Driver-stack entries: sum or largest.** The grid's border search keeps 3,600 B of GPU stack per thread, about
   199 MB on a 36-multiprocessor card, and contact has its own entry. The driver keeps one stack sized for the largest
   need of any function launched, so the sum of the modules' `kDriverStack` entries overcounts.
   Options: (a) the run plan takes the largest `kDriverStack` entry (rec: it is what the driver takes); (b) keep the
   sum, an upper bound. Taken meanwhile: the sum.
   Answered 2026-11-27: the run plan takes the largest entry, not the sum (A115).
3. **Status or precondition: three pairs drawn differently.** The rule is: an input that is data fails with a
   status; the processor of an object is a debug assertion; `ReadGrid` is the stated exception. Three pairs still
   differ: a range outside the grid is `kBadInput` in `BuildRanges`, `BuildGridPiece` and the buffer form of
   `FindBorderBlocks`, a debug assertion in `BuildRangeWithBorder`; a wrong processor is `kBadInput` in every build only
   in `ReadGrid`; an empty range is `kBadInput` in `BuildGridPiece` but accepted by the buffer form of
   `FindBorderBlocks` for a grid of 0 blocks. Later modules copy whichever they read.
   Options: (a) `BuildRangeWithBorder` returns `std::expected` with `kBadInput`; `ReadGrid` stays the stated
   exception; an empty range is bad input except for a grid with no block (rec); (b) leave as documented in the
   headers; (c) turn `ReadGrid`'s check into an assertion too. Taken meanwhile: (b).
   Answered 2026-11-27: one rule, kBadInput for a bad or empty range, except over a grid with no block (A116).
4. **The four merge settings of `CellSizeSettings`.** Cell resize reads `merge_width_share`, `merge_wait_duration`,
   `min_merge_wait_steps` and `max_cell_splits_per_resize`; the grid design declared them, the grid built four other
   fields only.
   Options: (a) add them when cell resize is planned, in that plan (rec: only cell resize reads them); (b) add them
   to the grid now. Taken meanwhile: (a).
   Answered 2026-11-27: added with cell resize (A124).
5. **A load counter in a grid piece's version.** Values are checked against a grid by version number and block
   count only. A piece has the whole grid's version, so two pieces of one whole grid with equal block counts pass
   each other's check; values bound to one piece could be read through another without an error.
   Options: (a) keep the rule "bind again at every load, never read through another piece", stated in the headers
   (rec for the first version); (b) add a load counter to a piece's version, so the check catches it.
   Taken meanwhile: (a).
   Answered 2026-11-27: no load counter; values are bound again at every load (A124).

## Before the transfer plan

6. **Face search: size levels `l - 1` to `l + 1` only.** The face search, and so the wide border and transfer's
   fills, finds face pieces of the start's own size level and one either side, strictly within the radius; pieces
   two or more size levels away are not found. A search of every size level has no fixed block bound, so it could not
   run on the GPU in a fixed array nor give batches a finite border (grid Ruling 56).
   Options: (a) confirm (rec); (b) every size level, which changes `FaceSearch`, the wide border and transfer 5.4.
   Taken meanwhile: (a).
   Answered 2026-11-27: confirmed, the face search covers the start's size level and one either side (A124).
7. **Positions far from the origin.** A face offset is a difference of absolute positions and carries about `eps`
   times the distance from the origin: 4e-12 on a fitted value at 1,300 m with 3 m cells. The grid documents the
   limit, and its tests carry a term for it (grid Ruling 72).
   Options: (a) accept the documented limit; scenes keep the frame origin near the liquid (rec: the error is far
   below the solver's tolerances); (b) compute offsets relative to the cell, a change in the weights and every caller.
   Taken meanwhile: (a).
   Answered 2026-11-27: accepted, the precision limit far from the origin stays (A124).

## Before the first GPU run

8. **A compile-only Windows CUDA job.** No grid `.cu` file or shared header has gone through nvcc with MSVC as the
   host compiler; CI compiles CUDA on Linux only. The first GPU run on Windows is likely to stop at the build.
   Options: (a) add a compile-only Windows CUDA job to CI before the run; it needs no GPU (rec); (b) find out at the
   run. Taken meanwhile: (b).
   Answered 2026-11-27: a compile-only Windows CUDA job is added to CI (A117).
9. **One atomic add per active cell in the GPU piece build.** The counts of a GPU piece take one integer `atomicAdd`
   per active cell (grid Ruling 71).
   Options: (a) keep it until a GPU run shows a cost (rec); (b) count per thread block first. Taken meanwhile: (a).
   Answered 2026-11-27: kept until a GPU run shows a cost (A124).

## Rulings taken without the owner, to confirm

10. **Four grid rulings.** (a) The form of `FindBorderBlocks` that wrote into a caller's span is retired; the buffer
    form serves every caller that must not allocate (Ruling 66). (b) The cover step of the GPU build is two launches
    instead of one, so no thread reads a byte another thread writes (Ruling 38; one launch more per build). (c) A
    cell whose box reaches outside the refinement volume takes the smaller of its code and 0, since the factor outside
    is 1 (Ruling 44). (d) On the sloped-surface scene a range of 8 blocks loads up to 105 of 218 blocks with its near
    border (97 border blocks, about 12 times the range): told for the batch design.
    Options: confirm all (rec); or name the one to reopen. Taken meanwhile: as ruled.
    Answered 2026-11-27: all four rulings confirmed, the sloped-scene border size included (A124).

## Tests and the gate

11. **Debug functional tests out of the commit-speed gate.** At commit speed the debug build runs unit tests only
    and the release build unit and functional tests; full speed runs everything in both (Ruling 68). Cost if wrong: an
    assertion that fails only in debug in a functional test is seen at full speed or in CI, not at each commit.
    Options: (a) keep (rec: commit speed fell from 4 to 6 min to under 2 min); (b) run debug functional tests at
    every commit. Taken meanwhile: (a).
    Answered 2026-11-27: kept, debug functional tests run at full speed only (A124).
12. **Grid piece tests: full counts or reduced.** The border piece tests run on 5 to 16 of the 200 random grids
    (plus 16 framed grids); 30 or 50 grids take minutes in debug at full speed. `stacked_borders_in_grid_pieces` takes
    108 to 114 s in release at 100 grids, over the 60 s budget of Ruling 69.
    Options: (a) keep the reduced counts, and plan piece tests at real range sizes in the batches module (rec);
    (b) restore 30 or 50 grids; (c) 30 in debug, 100 in release. Taken meanwhile: (a).
    Answered 2026-11-27: reduced counts kept, stacked-border program cut, real sizes in the batches module (A119).
13. **clang-tidy over the GPU-only branches of host files.** The gate's tidy step reads the debug compile database,
    so the `FLUIDITY_HAS_GPU` branches of host `.cc` files are never checked.
    Options: (a) a second tidy run in the container over the CUDA compile database for those files, adding about
    their tidy time to the cuda step (rec); (b) accept it and say so in `AGENTS.md`. Taken meanwhile: (b).
    Answered 2026-11-27: a second clang-tidy run in the CUDA container (A120).
14. **Fix commits: folded or kept.** Fixes after a task's commit can be folded into the commit that owns the code
    (every later hash changes, conflicts are frequent, every rewritten commit is gated again) or kept as described
    commits after it (each passes the gate). A tools commit also holds two small topics that change one function.
    Options: (a) keep fix commits as described commits, and leave the tools commit as it is (rec); (b) fold, and split
    the tools commit. Taken meanwhile: (a) for the latest commits.
    Answered 2026-11-27: fix commits stay as described commits (A124).

## Words and tools

15. **The word "chunk" and the other one-sense words.** The glossary bans "chunk" for a piece and keeps it for the
    stable order; the grid uses it in `SetPlaceChunk`, `ComputeMarkingChunk` and `ComputeMarkingChunkCount`. About
    5,200 one-sense notes remain in the tree, mostly ordinary senses the glossary's qualifiers do not exclude
    (the physical `floor`, `sorted`, `record`, `wall`, `reach`).
    Options: (a) rename the three grid names (for example `SetPlaceLaunchPart`) and rewrite the qualifiers so the
    allowed senses stop noting (rec); (b) allow "chunk" for a launch's share; (c) leave the notes. Taken meanwhile:
    (c).
    Answered 2026-11-27: renamed to shard; chunk kept for the stable order (A121).
16. **Verb rule for Python names.** C++ names start with an allowed verb; the tools' Python names do not follow it.
    Options: (a) apply the verb rule to Python functions too (rec: one rule); (b) Python keeps its own conventions.
    Taken meanwhile: (b).
    Answered 2026-11-27: Python tools keep Python's own naming (A122).
17. **Old file names in comments.** The rules check bans glossary words (stencil, basis, mac, adaptive, node) in
    comments, so the comparison tests with `main` describe the old files instead of naming them; a backtick span
    starting with `main:` is now allowed.
    Options: (a) the `main:` span is enough (rec); (b) an allowed-names entry for each old file. Taken meanwhile: (a).
    Answered 2026-11-27: the main: span is enough (A124).
18. **A page-overflow check for the method paper.** Text below the page's text area slipped through twice; four
    older side or top overruns are known (pages 9, 31, 36, 57).
    Options: (a) a PDF step that reads glyph positions (needs a PDF text tool in the pinned environment); (b) a check
    inside Typst that fails the compile (rec: no new tool); (c) check the pages by eye. Taken meanwhile: (c).
    Answered 2026-11-27: a check inside Typst that fails the compile; the four known pages are fixed (A123).
19. **Two new detectors.** One-word lines inside multi-line comments, and notes on Python tool code.
    Options: (a) add both as notes (rec), once "one-word line" is defined for lists; (b) drop them. Taken meanwhile:
    neither added.
    Answered 2026-11-27: neither detector is added (A125).
20. **The editor settings.** The `.vscode` file nesting and icon keys were written but never opened in an editor.
    Options: (a) check them once in VS Code (rec); (b) remove them. Taken meanwhile: kept, unchecked.
    Answered 2026-11-27: checked by the owner, fine (A126).
21. **The gate on Windows.** The tidy cache needs `-M` on Windows, and with the cache on every file fails there.
    Options: (a) make the gate work on Windows now; (b) only when the owner runs the gate on Windows (rec). Taken
    meanwhile: (b).
    Answered 2026-11-27: the gate on Windows waits until the owner needs it (A124).
