# To reconcile after the back-propagation agents finish (2026-11-14)

Not committed to `rebuild`. Collected from the agents' reports. One cleanup pass applies these.

## Plans that contradict plan-decisions.md (plan-decisions wins; fix the plan text)
1. Gas plan Task 18: `ApplyGasStepStart` runs in stage 10 whenever stage 10 runs (viscous scenes too, with
   `kReadOneStepLate`); only a gas scene without stage 10 calls it at stage 13.
2. Step plan Task 14: under A17 no needed block ids come back to the CPU; the test
   `ChooseActiveCells, OnGpu_EqualsCpu` must compare through `ReadGrid` or the status record.
3. Step plan Task 33 / pressure interface: `ResetSolveCopyReport` takes `SolveBatches&` but
   `FindSolveBatches` returns `const SolveBatches*`. Give pressure a non-const accessor or a reset on the buffers.
4. Step plan Task 28: blocking faces are built in every scene with liquid and a blocking set, but the viscous
   joint solve returns `kBadInput` for equations with material faces. A springy solid beside a viscous liquid
   therefore fails (also the five-material scene of `ContinuedFiveMaterialRun_SameBytes`). Needs a decision:
   exclude the pair as a known limit (`kExperimental`/unavailable finding), or fall back to the one-step-late
   push for viscous liquids, or carry material faces in the joint solve. OWNER QUESTION.

## Designs not yet updated
- 12-io-and-cli-design.md: both batched modes, `max_solve_loaded_blocks`, `allowed_cpu_bytes`, planned cost
  output (after the plan-gap agent finishes).
- 07-materials-design.md:415 (`ApplyPressureToMaterials` faces form); 15-sand-design.md:156 (push one step late
  replaced by A18); 05-pressure-design.md ~:1033 heated sealed gas test row (if the pressure agent missed it).
- A23 rename everywhere: `MemoryRow` -> `MemoryEstimateEntry`, `Build<Module>MemoryRows` ->
  `Build<Module>MemoryEstimate`, `*_rows` -> `*_estimate`.

## From the pressure/viscosity back-propagation
5. Pressure plan Tasks 24, 44(b), 48 still say "one thread per blob sums": replace by the per-group block rule
   (block partials then a tree sum) that the plan's own Global Constraints and plan-decisions.md state.
6. plan-decisions.md "Viscosity" still says samples ordered by ascending `CellId`; its A15 section says block
   first. Mark the old line replaced.
7. Pressure plan uses the name `EquationArrays` twice (Task 13 test-scene helper; Task 23 the opaque
   `PressureEquations::arrays` type). Rename the test helper (for example `EquationSceneArrays`).
8. Pressure plan Task 44 `SealedGasBlob` comments differ slightly from plan-decisions.md: make them identical.
9. 12-io-and-cli-design.md:133 `"liquids_that_do_not_mix_touch"` (reason removed).
10. Check after all agents finish: 08/09/10 designs no longer mention the stale items the pressure agent listed
    (they were being rewritten at the same time).

## From the grid/batches back-propagation
11. Grid: the default `max_cell_blocks = 1u << 22` now costs about 8.4 GB of GPU memory per grid storage
    (1,996 B per block under A17) and the step keeps two: over a 16 GB card before any field. The grid storage
    must be sized by the planner from the scene (and the build scratch shared between the two storages, or the
    scratch sized for one build at a time), not from the default. Needs a plan decision; OWNER should know.
12. Batched GPU run: stage 2 runs on the CPU but A17 builds the grid on the GPU from a list there; how the
    CPU-made cell list reaches the GPU builder, and how `step::ApplyResizedGrid` receives the uploaded list, are
    not declared. (The plan-gap agent's inventory may settle it; check.)
13. `ChooseCellSize` callback needs CPU tables a GPU grid lacks (proposal: the `ReadGrid` copy cell resize keeps).
14. Stale lines to re-check after all agents finish: 10-step-design.md:343, 624, 1038, 1055, 1312;
    05-pressure-design.md:115; 06-viscosity-design.md:334; 09-gas-design.md:853, 907; 08-solids-design.md:319,
    909, 944; 12-io-and-cli-design.md:405, 567, 783, 1289.

## From the Blender / render-output / shared-interface back-propagation
15. plan-decisions.md lacks "Foundation as built" entries for: slice forms and deduction from the buffer,
    `math::TreeSumOf`, `math::LargerMagnitude`, same-bits-unless-NaN, Mat3 by value, the `<name>_gpu.h` rule,
    verbs `Copy` / `Release`, `OrderScratch::byte_count()`, `RunOnError` living in `run_on.h`. Add them (the
    built headers are the source; shared-interface.md section (a) already copies them).
16. shared-interface.md has no rows yet for: transfer `TransferScratch`, particles `ApplyParticleLimits` /
    `ComputeAmounts`, solids `BuildSolidsBuffers` / `BuildSolidCoverageValues` / `MotionMap`, step
    `RandomWords` / `StageOutcome` / `run_whole_grid` / `FindLoadedTry` / `WriteStepArrayBlocks`, I/O
    `FrameWriter` and `MaterialPreset`. Add from plan-decisions.md.
17. After Foundation's fold: re-copy section (a) of shared-interface.md from the final headers
    (`MemoryEstimateEntry`, `check_can_run_on_gpu.h`, any signature the comment pass touched).

## After A24/A25 were applied (next planning pass, together with the Foundation final-fix adoptions)
18. Batches plan Task 29 still says `ChooseCellSize` gets null tables in a batched GPU run; reconcile item 13
    says the CPU copy's real tables.
19. "Buffers in batched runs": the step's CPU grid-copy pair is counted with `BuildGridMemoryEstimate(kCpu, 2)`;
    with the shared scratch it should be `(kCpu, 2, kOneForAllBuffers)`.
20. Grid plan Task 6: say which header declares `ScratchSharing`.
21. Names: `io::LiquidViscousEffect` vs `step::LiquidViscousEffect` (same name, different fields: rename the io
    one, for example `LiquidViscousEffectLine`); gas test-only `GasLoopScene::sealed_gas_read_one_step_late`.
22. Adopt Foundation's final-review additions in the plans (scratch forms of the sums and of the grouping,
    `fluidity_add_test_library`, test-support helpers, kernel-wrapper rule): list comes from final-fix-report.md.
    Done 2026-11-14: plan-decisions.md "Plans adopt Foundation as built (2026-11-14)".

## Before the viscosity module is built
23. The A24/A26/A27 method (material band and free solids as unknowns of the joint solve, stick at the true
    distance, saddle-point block form) was designed by one planning agent and checked only by greps. Have an
    independent reviewer check the derivations and the test numbers (Couette slab time constant 0.25781 s and
    per-step factor 0.96266, film bound 1.511e-3 m/s, Faxen factor, half-shear rotation) before module 5 starts.

## After the plans adopted Foundation as built (small follow-ups; do with the first module that touches them)
24. Designs: pressure and gas section 3 gain the `RunOn` parameter of `ComputeBufferBytes` /
    `BuildPressureMemoryEstimate` / `BuildGasMemoryEstimate`; `fluidity_apply_compile_flags([RTTI ON|OFF])`.
25. Viscosity and solids error enums have no out-of-memory value (a failed allocation is `kRunOnCallFailed`):
    align with `kOutOfMemory` only-from-allocations, or state the exception.
26. Render output Task 17: per-field-part stable order has no bound, so no memory estimate entry: bound it.
27. `BuildTransferScratch` may need `material_count` (its bytes depend on it).
28. Grid's GPU `OrderScratch` (6 B per item) is counted by no estimate entry.
29. Test-library names: `_test_helpers` (plans, glossary ruling) vs `fluidity_base_test_support` (Foundation):
    pick one suffix and apply it (Foundation's is built: prefer `_test_support` everywhere, or rename base's).
30. Known-open tests: `DISABLED_` prefix + a second registration with `WILL_FAIL` and label `known_open`
    (owner to confirm the form).

## Controller rulings after the A28–A33 pass (2026-11-15; apply when each module starts)
31. Every module error enum has `kOutOfMemory`, returned only by its building functions, APPENDED at the end of
    the enum (never inserted, so earlier values keep their numbers): pressure, gas, transfer too; move the value
    the pass inserted in viscosity and solids to the end.
32. A30 applies to module functions: a span (CPU) form called on a GPU grid, or a CPU list with GPU buffers, is a
    precondition with a debug assertion, not `kBadInput` (`FindBorderBlocks`, `BuildGridPiece`, the grid checks).
33. Accepted as written: allocating `BuildGrid(inputs, settings)` removed in favour of test-only `BuildTestGrid`;
    span forms of `ComputeGridCounts` / `CheckSizeStepRule` kept; `ComputeAmounts`, `ComputeWorkRates`,
    `CheckMaterialState` stay one allocating form each; `kBadStepInputs`, `step_inputs`; `max_field_parts = 1u << 26`
    with one frame `OrderBuffers`. Sand Task 7's per-call GPU copy: settle in the sand module.
34. Known-open test form is confirmed by the owner (A38).
