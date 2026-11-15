# Questions waiting for the owner (collected 2026-11-14, while Foundation was built)

Not committed to the `rebuild` branch. Each has the option taken meanwhile, so work does not wait.

## Naming
1. Verbs `Copy` ("copies values between two arrays on the same processor") and `Release` ("gives back memory or
   buffers that a Build call took") were added to the allowed verbs for `CopyBuffer` and `ReleaseBufferBytes`.
   Taken meanwhile: both allowed.
2. Tree vocabulary in the fixed-order sums: working words "layer", "pair sum", "padding"; the tree is explained
   once with "node" and "child". Taken meanwhile: as written.
3. Two public names beyond the shared interface: `math::TreeSumOf` and `math::LargerMagnitude`; and
   `BufferError::kRunOnMismatch`, `OrderScratch::byte_count()`. Taken meanwhile: kept.

## Physics and method
4. Snow's own compressibility inside the liquid pressure solve (today only its inertia is). Needs the materials
   module to drop the volumetric part of its stress. Taken meanwhile: not built.
5. Sealed gas beside a viscous liquid still lags one step and keeps its step limit (the viscous solver does not
   carry the gas term). Taken meanwhile: left lagged.
6. Wind acts on gas only. Wind on liquids is a solver change. Taken meanwhile: gas only, with a note in the add-on.
7. Does a viscous liquid always stick to solids; one unbounded plate; the split/merge rule for particles
   (from the first question list, still on the recommended options).
8. Should velocity or surface curvature enter the cell-size rule (the owner called it "a really good question").
   Taken meanwhile: the design's rule.

## Batches and memory
9. Two new constants of the batched solves: a quarter of the GPU budget per solve piece
   (`solve_piece_share = 0.25`), and 4 values per cell of direct-solve room in the batched layout (32 in one piece).
10. In the solve-in-pieces mode the geometry snapshot stays on the GPU; streaming it needs three solids functions
    to run batch by batch. Taken meanwhile: stays on the GPU.
11. The I/O and CLI plan does not yet print the planned budgets and copies per step. Taken meanwhile: to amend.
12. Stage 1 runs on the CPU in batched runs while some gas marker buffers live on the GPU; one was moved to CPU
    memory, the rest is open.
13. `BuildGroupsByDestination` allocates per call; the solvers call it every step. Taken meanwhile: add a scratch
    form in the first solver plan that needs it.

## Blender and outputs
14. A new Wind field has flow 1.0 by default, so the "flow not honoured" note shows on every default field.
    Drop the note, or keep it?
15. Refuse Blender 5.4 and later, or only print a note? Taken meanwhile: note only (no maximum version).
16. OpenVDB from Conan Center is 12.1.1, older than Blender 5.3's 13.0; the Blender read tests decide.
17. OpenUSD from the start (confirm), and the Conan remote on this Mac must be switched to the current one.
18. The GPU build links the CUDA runtime as a shared library, so the add-on must ship it (or link statically).
    Blender may also turn on flush-to-zero on its threads, which would make the CPU half of the GPU check differ.

## Tooling and paper
19. Paper fonts (Century Supra / Concourse / Triplicate with fallbacks): keep?
20. Test names containing "And" are forbidden by naming rule 13; a few plan test names have it.
21. Lean and the independent kernel checker (A21) are not installed yet: planned for the first module with a proof.
22. clang-format / clang-tidy latest versions (A19): measured in the comment pass; folded if the reformat is small.

## Added 2026-11-14 (later)
23. Viscous liquid against snow or a springy solid: the liquid SLIDES along the material (the solids design's rule
    for separate sets) and the viscous stress from the material's own velocity change in the step is left out.
    The shared-velocity papers stick instead, which needs the material's velocity as a viscous unknown.
24. Free solids (rigid bodies moved by the liquid) beside a VISCOUS liquid are still refused: the solid's velocity
    would have to become a viscous unknown. A24 lifted snow, springy solids and sealed gas only.
25. `kSmallestVisibleViscousEffectPerStep = 2.5e-4` (below it: under 0.1 % change per step).
26. Grid: no default room; planner sizes it; scratch shared (2,840 B per block for two grids). Confirm.
27. Run mode stays in the run fingerprint; should a resume reuse the recorded mode or refuse if memory that day
    picks another?
28. Size choice on a GPU grid: keep the callback (two small copies per grid version) or a rule that runs on both
    processors.
29. Warning margins: memory 10 % over budget, copies 25 % over.
30. One early Windows build (ten minutes, CPU only) or one early run of the GPU script, instead of discovering
    Windows compile errors at the very end.
31. `ApplyLimits` in a GPU run rebuilds the committed grid but the committed cell list is not kept on the GPU.

## Viscous solver with sticking (A26/A27): modelling choices made in the plan, to confirm
32. Only a band of the material (faces within 1.5 cell widths of a liquid cell) is solved with the liquid; the rest
    of the material keeps its velocity for that step.
33. Where two materials share a face piece, the one with the larger face mass is what the liquid sticks to.
34. A guided solid (some motions forbidden) holds those components at their value before the solve.
35. Viscous drag on driven solids (plates with prescribed motion) is not reported in any total.
36. The absolute settling speed of a cube in syrup is tested as a bound ([0.7, 1.2] x the corrected Stokes value),
    not as a match; linearity in viscosity and density excess is tested to 2-3 %.

## Added after the plans adopted Foundation as built
37. Eleven new "same bits on CPU and GPU unless NaN" promises in module plans (contact formulas without turning,
    transfer cell sums, particle amounts, two materials functions, pressure distance to surface, three gas
    functions, velocity after gravity), each because only basic operations in a fixed order are used. Confirm,
    together with the base rule (D7).
38. Known-open tests are written with GoogleTest's `DISABLED_` prefix and registered a second time as expected
    failures under the label `known_open`.
39. Names from Foundation's last rounds: `RunOnCallStatus`, `ApplyGpuFunction`, `CheckGpuWork`, `SumScratch`,
    `OrderScratch`, `MemoryEstimateEntry`, "thread block", verbs `Copy` and `Release`.
