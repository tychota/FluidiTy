# Where particles live, and how big they are (decision note, 2026-11-15)

Question for the owner: in the first version, which liquid cells hold particles, and how big is each particle?
This note gives the counts and memory for four scenes, what the papers say about accuracy, the cost per step,
the work each option needs, and a recommendation. Read-only input: the module designs as they were on
2026-11-15 (a pass is rewriting them for A64; nothing here edits them).

Words used: **w_min** is the smallest cell width. A **level k** cell has width `2^k w_min`. A **cell block** is
4 x 4 x 4 cells of one size (64 cells). **Particle size** means the particle's `size_level`: its mass and rest
volume are those of one eighth of a cell at that level.

## 0. Short answer

- The first version should be **option B**: particles everywhere, eight per cell, so particle size follows the
  cell. A64 should be read as B.
- The designs already do B **after** the grid has coarsened, but they **start** as A on an all-smallest grid, and
  the memory plan is sized on that start. So today the plan needs A's memory (plus a uniform grid) even though the
  run later uses B's. That must change for B to save anything (section 1.4).
- At w_min = 1 cm a 16 GB GPU holds the glass and the bathtub under every option. The pool and the open sea do
  not fit under any option, not even in batches. What limits them is the surface pinned to the smallest cells
  (A61), not the deep particles. C would save about a third of the memory, and halving the surface reach of the
  size rule saves about the same with no new algorithm.

## 1. Counts and memory

### 1.1 The three options, defined for this solver

The grid has cells in powers-of-two sizes, at most one level between neighbours, and the surface, air pockets,
gas and solids in the smallest cells (A61; cell resize design 5.3 items 1 and 7). Size rule (cell resize 3.2,
5.3):

```text
wanted = min( w_min * max(1, depth / (8 w_min)),  w_min * max(1, distance_to_solid / (2 w_min)) ),  capped by w_max
cell width = the largest 2^k w_min that is not above wanted
```

So level 0 fills depth 0 to 16 w_min below the surface (wanted reaches 2 w_min only at depth 16 w_min). Level k
then fills depth `8·2^k w_min` to `16·2^k w_min`: a layer 8 cells thick at its own size. Near a floor or wall,
level k fills distance `2·2^k` to `4·2^k` w_min: a layer 2 cells thick.

| Option | Which cells hold particles | Particles per cell of level k | Particle size |
|---|---|---|---|
| A, one size | every liquid cell | `8 · 8^k` (spacing w_min/2 everywhere) | always the smallest |
| B, size follows cell | every liquid cell | 8 | the cell's level. Split 1 to 8 when the cell splits, 8 to 1 merge when cells merge |
| C, narrow band | cells within the band of the surface, cells touching solids, air, gas or another material | 8 in the band, 0 deeper | the cell's level (B inside the band) |
| B + C | B everywhere except grid-only cells at level >= 2 | 8, or 0 in the deepest cells | as B |

C as the designs had it (cell resize 3.4, 5.5, 5.6): particles down to 6 cell widths below the surface, a full
cell deeper than 7 widths becomes grid-only, no change between the two depths, a 2-layer velocity blend at the
edge, 8 particles per refilled cell. Ferstl et al. 2016 (NB-FLIP) keep particles to R = 3h or 4h and switch
sharply at r = R − h (section 3, section 4 "We used narrow band widths of R = 3h and R = 4h (with r = R − 1)").
For memory I use the design's band: **7 w_min** below the surface (the upper end, because the plan must be an
upper bound, A43), plus a **2 w_min** shell at the floor and walls (cells touching a solid keep particles).

### 1.2 Which option the designs imply today

**B in steady state, A at the start, and a leak back toward A.**

- B: particles design 5.4 to 5.6 (a particle has a `size_level`. `SplitParticle` gives 8 children at offset
  `sqrt(3)/4 H` with mass/8. `MergeParticles` joins 2 to 8 particles of one bigger cell. `AdjustParticleCount`
  splits a particle whose cell is smaller than its size and merges 8 particles of a size one below the cell's).
  The refill gives 8 particles per cell at the cell's level (cell resize 3.4 `particles_per_refilled_cell = 8`,
  5.6 step 3). The batch plan counts 512 particles per block "8 per cell" whatever the block's size (cell resize
  5.9 step 2). Sources use `particle_spacing^3` "at the source's size level" (I/O 5.1).
- A at the start: start regions are sampled at `particle_spacing = smallest_cell_width / 2 (8 per smallest
  cell)` (I/O 5.1). With no committed grid, new cells take `new_cell_size_level` (step 5.3 step 2; cell resize
  5.3 `ChooseCellSize` returns the smallest level when nothing is found). `BuildRunPlan` sizes every limit on the
  start particles and the 2.5-width cover of them **at the smallest cell**, times 1.25 (step 5.9). So the run
  starts as "A on a uniform grid". It reaches B only after the merges: one level per resize, after a 0.25 s and
  8-step wait for each level (cell resize 5.3 item 5). The memory is planned for the start.
- Leak: a particle that drifts across an existing size change is never split or merged and "keeps its
  `size_level` for good" (particles 5.4, last paragraph; cell resize 5.4). Small particles carried into coarse
  cells pile up beyond 8 per cell, which is A locally and grows past the plan. Large particles carried up into the
  surface band stay large at the surface, which is B's known surface problem (section 2.3).

### 1.3 Assumptions for the numbers

| Value | Assumed | Source |
|---|---|---|
| Particles per smallest cell | 8 | I/O 5.1; cell resize 5.7 "64 cells, 512 liquid particles" |
| Bytes per particle, one-piece GPU run | **675 B** (particles, contact, materials, transfer working rows) | cell resize 5.7: "particles and their working buffers about 345 kB" per 512 particles. The particles table alone is 412 B (particles 6.4); with transfer's 168 + 200 B it would be 780 B, which is +16 % (would not change a verdict) |
| Bytes per cell block, one-piece GPU run | **163 kB** = whole-grid arrays 55 kB + transfer and solids 105 kB + two grid tables 2.84 kB | cell resize 5.7; grid 3.6 ("2,840 B per block"). No viscosity (190 kB instead of 55 kB with viscosity) |
| Check | a full smallest block = 512 × 675 + 163 kB = 509 kB. 13.05 GB / 509 kB = 25,600 blocks | cell resize 5.9: "about 22,000 full blocks ... 11 GB allowed" (500 kB per block). Consistent |
| Block size | the same 64-cell cost at every level | grid 3.6 (per-block rows do not depend on the level) |
| Blocks per cell, adaptive | cells / 64 × **1.5** (partly used blocks at level changes and walls), plus one block layer of air over the surface, never more than the uniform count | my estimate, ±30 % |
| Largest cell | w_max = 64 w_min (6 levels) | The floor rule already caps widths at about depth/10 (where `(H−z)/8 = z/2`), so for these scenes the cap only acts in the sea. Raising the cap to 2^10 changed no count by more than 1 % (checked) |
| GPU budget | 13.05 GB (A58: 90 % of 14.5 GB free) | task |
| CPU budget, batched run | 0.7 × 64 GB = **44.8 GB** | step 3 `memory_share_of_total = 0.7` |
| Batched run, CPU | 780 B per particle (plan-decisions "about 400 kB per block of particles"), plus about 150 kB per block of particle-stage face arrays and solver rows (my assumption) | cell resize 5.7 |
| Batched run, GPU | 18 kB per block stays whole (solves in pieces, 15 kB plus grid), and the loaded batch must hold `T + 8 >= 9` layers of blocks at w_min, at about 540 kB per loaded block (640 particles + 105 kB + grid) | cell resize 5.9 steps 2 to 4 |
| Scenes | glass 0.3 × 0.3, liquid 0.2 deep, walls and floor; bathtub 2 × 1 × 0.5; pool 10 × 5 × 1.5; sea 50 × 50 × 10 with open sides (no wall refinement) and a solid floor | task |

Method: cells = ∫ dV / W(x)³ over the liquid. The distance to a solid is `min(height above floor, plan distance
to the nearest wall)`, and the plan area at wall distance s has density `2(X − 2s) + 2(Y − 2s)`. Integrated
numerically, cut at every level boundary. Particles: A = 8 V / w_min³, B = 8 × adaptive cells,
C = 8 / w_min³ × (surface area × 7 w_min + solid area × 2 w_min).

Hand check, pool at 1 cm, level 0: under the surface 0.16 m plus 0.04 m at the floor = 0.20 m × 50 m² = 10 m³,
which is 1.0e7 cells. Walls: 30 m perimeter × 0.04 m × 1.3 m = 1.56 m³, which is 1.56e6 cells. Total 1.16e7, which
matches the integral. Sea at 1 cm, level 0: 2,500 m² × 0.20 m = 500 m³, which is 5.0e8 cells (matches).
A = 8 × 25,000 / 1e-6 = 2.0e11. Bathtub B memory: 4.16e6 × 675 B = 2.81 GB; 13,414 blocks × 163 kB = 2.19 GB;
total 5.0 GB.

### 1.4 Cells and blocks

| Scene, w_min | Cells per level (0, 1, 2, ...) | Adaptive cells | Uniform cells | Adaptive blocks | Uniform blocks |
|---|---|---|---|---|---|
| glass 1 cm | 1.8e4 | 1.8e4 | 1.8e4 | 337 | 337 |
| bathtub 1 cm | 4.70e5, 4.69e4, 2.4e3 | 5.19e5 | 1.0e6 | 1.34e4 | 1.69e4 |
| pool 1 cm | 1.16e7, 1.38e6, 3.2e5, 6.2e4 | 1.33e7 | 7.5e7 | 3.43e5 | 1.20e6 |
| sea 1 cm | 5.0e8, 6.3e7, 1.6e7, 3.9e6, 9.8e5, 2.4e5, 3.4e4 | 5.83e8 | 2.5e10 | 1.52e7 | 3.92e8 |
| glass 0.5 cm | 9.0e4, 6.8e3 | 9.67e4 | 1.44e5 | 2.47e3 | 2.47e3 |
| bathtub 0.5 cm | 1.98e6, 2.2e5, 4.7e4, 2.4e3 | 2.25e6 | 8.0e6 | 5.78e4 | 1.30e5 |
| pool 0.5 cm | 4.67e7, 5.7e6, 1.4e6, 3.2e5, 6.2e4 | 5.42e7 | 6.0e8 | 1.39e6 | 9.50e6 |
| sea 0.5 cm | 2.0e9, 2.5e8, 6.3e7, 1.6e7, 3.9e6, 9.8e5, 5.2e5 | 2.33e9 | 2.0e11 | 6.09e7 | 3.13e9 |

At 1 cm the glass has no coarse cells at all: the floor and walls pull the cells back to w_min before the
surface rule lets them grow. In every scene, **level 0 holds 86 % to 91 % of the cells**: the 16-cell surface
band plus the floor shell.

### 1.5 Particles and memory (one-piece GPU run)

"Today" = what the current plan is sized on: A particles on uniform smallest cells (section 1.2).

| Scene, w_min | Particles A | B | C | GB today | GB A | GB B | GB C |
|---|---|---|---|---|---|---|---|
| glass 1 cm | 1.44e5 | 1.44e5 | 1.03e5 | 0.15 | 0.15 | 0.15 | 0.13 |
| bathtub 1 cm | 8.0e6 | 4.16e6 | 1.92e6 | 8.2 | 7.6 | 5.0 | 3.5 |
| pool 1 cm | 6.0e8 | 1.07e8 | 4.32e7 | 601 | 461 | 128 | 85 |
| sea 1 cm | 2.0e11 | 4.67e9 | 1.80e9 | 199,000 | 137,000 | 5,630 | 3,700 |
| glass 0.5 cm | 1.15e6 | 7.73e5 | 4.13e5 | 1.18 | 1.18 | 0.93 | 0.68 |
| bathtub 0.5 cm | 6.4e7 | 1.80e7 | 7.68e6 | 64 | 53 | 21.6 | 14.6 |
| pool 0.5 cm | 4.8e9 | 4.33e8 | 1.73e8 | 4,790 | 3,470 | 520 | 344 |
| sea 0.5 cm | 1.6e12 | 1.87e10 | 7.2e9 | 1.6e6 | 1.1e6 | 22,500 | 14,800 |

Split for B at 1 cm, particles + grid: bathtub 2.8 + 2.2 GB; pool 72 + 56 GB. In the band, 8 particles cost
5.4 kB per cell and the grid costs 2.5 kB per cell. Under C the grid becomes the larger part (pool: 29 GB of
particles against 56 GB of grid).

B + C (grid-only only in cells of level 2 and above): those cells hold **0 % to 3.6 %** of B's particles (glass
0 %, bathtub 0.5 %, pool 2.9 %, sea 3.6 % at 1 cm). The combination saves nothing worth a second algorithm.

### 1.6 Verdict on a 16 GB GPU (13.05 GB budget)

"Batches" requires both the CPU rows within 44.8 GB **and** the design's slab rule (cell resize 5.9 step 4: a
batch is a slab of z-ordered blocks; the GPU must hold 9 layers of blocks at w_min, each layer
`(X / 4 w_min) × (Y / 4 w_min)` blocks). For one-piece runs the slab rule is not needed.

| Scene | 1 cm: today | A | B | C | 0.5 cm: today | A | B | C |
|---|---|---|---|---|---|---|---|---|
| glass | fits | fits | fits | fits | fits | fits | fits | fits |
| bathtub | fits | fits | fits | fits | no | no | no | no (14.6 GB) |
| pool | no | no | no | no | no | no | no | no |
| sea | no | no | no | no | no | no | no | no |

Batches never rescue a scene here. The bathtub at 0.5 cm needs 22.7 GB of CPU memory under B (fine), but one
layer is 5,000 blocks, and 9 layers × 5,000 × 540 kB = 24 GB of loaded batch against about 9.6 GB of room. The
pool at 1 cm has 31,250 blocks per layer: even one layer (17 GB) does not fit. By this rule, a batch run on this
card needs a plan area under about 3 m² at 1 cm (0.8 m² at 0.5 cm). Batches as designed help deep, narrow
scenes, not wide ones. The design states this: "a wide, flat scene whose single layer of blocks does not fit
with its border cannot run" (cell resize 5.9).

Smallest w_min that fits one piece (step of my search in brackets):

| Scene | A | B | C |
|---|---|---|---|
| glass | 0.5 cm | 0.5 cm | 0.5 cm |
| bathtub | 0.9 cm (0.8 in batches) | 0.7 cm | 0.6 cm |
| pool | 4 cm | 3.5 cm | 3 cm |
| sea | 25 cm | 25 cm | 20 cm |

At the size where each scene fits, it is only 30 to 70 cells deep, so A and B differ by less than a factor of 2
(pool at 3.5 cm: A 14.0e6 particles, 13.6 GB; B 7.9e6, 9.5 GB; C 3.5e6, 6.6 GB). **A 16 GB card holds about 1
million cells and 8 to 12 million particles**, and in a pinned-surface solver the surface area over w_min²
decides almost all of it.

### 1.7 A cheaper lever than C: the surface reach

With `surface_reach_in_cell_widths = 4` instead of 8, the fine band is 8 cells deep instead of 16. Recomputed:

| Scene, 1 cm | B, reach 8 | B, reach 4 | C, reach 8 |
|---|---|---|---|
| bathtub | 5.0 GB | 3.6 GB | 3.5 GB |
| pool | 128 GB | 86.5 GB | 85 GB |
| bathtub 0.5 cm | 21.6 GB | 15.3 GB | 14.6 GB |

Halving the reach saves about what C saves, and it also cuts the grid. Its cost is accuracy below the surface
(the fine band is shallower than the orbital motion of waves longer than about 16 w_min). The choice of 8 was
ported, not measured. It is a setting, so it can be tuned by test, not by building a new algorithm.

## 2. Accuracy and behaviour

### 2.1 Energy loss in the deep region

- **C, grid-only cells:** Ando and Batty 2020 report that their adaptive run damps motion "noticeably faster"
  than a uniform grid, from "a lack of resolution and excessive diffusion on coarse cells": the step comes from
  the finest cells, so semi-Lagrangian advection diffuses coarse cells over many small steps (section 9.2.2, read
  in literature-cell-size.md 3). Bridson's notes give linear semi-Lagrangian a viscosity-like term with
  coefficient about `u Δx` (sections 1.5, 3.3; literature-practice.md 1). This is exactly the design's
  `ComputeCarriedVelocity`, first order (cell resize 5.5: "it loses a little energy"). NB-FLIP found large-scale
  motion in "very good agreement" with full FLIP using semi-Lagrangian advection in the interior (Ferstl 2016,
  section 4, read 2026-11-15). But their interior was a uniform grid at the band's cell size, not coarse cells.
- **A and B, particles in coarse cells:** no semi-Lagrangian step. The transfer still filters velocity at the
  **cell's** width. Reasoning from literature-practice.md 1: `ν_num ≈ (1 − α) h² / (4 Δt)` for the PIC share, so
  a cell 2^k wider damps 4^k more per step. With the fitted (affine) change carried and `own_motion_kept = 0.99`
  (transfer 5.7, decision 7), α is close to 1 and the filter is small. This is reasoning, not a measurement. Both
  A and B damp the same here, because the filter is on the grid, not on the particles.
- **Large particles in large cells (B):** eight particles per cell carry velocity detail at the cell's
  resolution, the same ratio as at the surface. The grid cannot represent detail finer than its cell. The extra
  particles of A carry sub-cell detail only as "own motion" (the FLIP share) that the coarse grid never sees and
  pressure never corrects: noise that rides along, not resolved flow. **B loses nothing relative to the coarse
  cell itself.** What deep coarse cells lose is the cell size, and A does not recover it. (Reasoning. Ando et al.
  2013 make the same choice: particle radius follows the sizing function, Eq. 15.)
- Size-dependent damping: Ando 2013 set a per-particle PIC/FLIP blend `ν Δt / r²` (Eq. 14, clamped at 0.1) so
  that damping does not jump with particle size. It is worth adding to B if the standing-wave test shows a
  difference between levels.

### 2.2 Stability at the edges

- **C, band edge:** Ferstl 2016: blending by particle density "leads to inherently unstable simulations" and
  gains momentum (section 3.1, Eq. 2). A smooth transition "does not increase the quality", and the sharp switch
  (Eq. 3) is stable. The designs' blend is by layer distance over 2 layers (cell resize 5.5 step 3), not by
  density. That is safer than Eq. 2 but is not what NB-FLIP validated. The band must be thick enough that
  level-set advection error does not open gaps within one step, which NB-FLIP admits "is admittedly difficult to
  quantify" (section 4, Limitations). The designs also report momentum not carried when particles go into the
  record (cell resize 3.4 `momentum_not_carried`). Momentum is not conserved there.
- **B, size changes:** split conserves mass, volume, momentum, angular momentum and kinetic energy exactly
  (particles 5.4, offset `sqrt(3)/4 H`). Merge conserves all but energy, and the gap test guarantees no energy
  gain (particles 5.5, Lean proof `Merge.lean`). Mass and volume are exact by construction, unlike C's records.
  Resampling bumps: Adams 2007 (popping near the surface) and Ando 2013 (bumps from splits near the surface) are
  surface effects. Here size changes never happen in the surface band (cell resize 5.3 item 7). Size ratio:
  this design changes size by a factor 8 in mass per level, one level per resize, and only one level between
  neighbours. Adams uses split and merge thresholds 2 and 3 on the feature size (section 4.1). Winchenbach 2017
  (as quoted by Xu 2023) leaves the mass ratio 0.9 to 1.1 alone and splits above 2, a much finer change. Gao 2018
  warns that particle size not matching cell level "can easily lead to numerical fracture" (AGIMP).
- **Large particle carried up faster than sizes follow:** in the current design it is never split (section 1.2,
  the leak), so it reaches the surface large. Particles move at most half a cell per step and resize runs after
  every committed step, so cells could follow. What blocks the split is the rule that checks the parent at its old
  size on the old grid. Fix: a "drifted particle" rule, splitting at resize any particle coarser than its cell
  with only the children's check, and merging any group finer than its cell with only the merged check and the
  gap test. Momentum stays exact. The split's energy is exact only for a complete neighbourhood, so report the
  difference.

### 2.3 Surface quality

All three options put the same small particles in the surface band, because the band is pinned to w_min and all
of it lies within the particle region under C. So B's literature surface problems (popping, bumps, blobbiness
from resampling at the surface) **are moot here, provided the drift fix of 2.2 is made**. Without it, a large
particle that rises into the band shows up in the surface (render radius `cbrt(3 rest_volume volume_ratio / 4π)`,
surface design 5).

### 2.4 Interaction with the rest of the solver

| Concern | A | B | C |
|---|---|---|---|
| Batched runs: particles in CPU memory, copies per try (~275 kB per full block) | worst: `particles_per_block = 512` in the batch plan is wrong for coarse blocks (it would be 512·8^k) | matches the plan | fewest. The plan's 512 per block overcounts deep blocks |
| Grouping (two stable sorts) per step | ∝ particles: 1.9× B (bathtub 1 cm), 5.6× (pool) | baseline | 0.4 to 0.5× B |
| Solids and gas coupling | same in all: cells touching solids and gas stay smallest and keep particles. The designs forbid grid-only cells next to gas or solids | | |
| Sticking viscous solve | same in all: the viscous band is 1.5 cells at the surface (A42), inside the particle region | | |
| Repeatability | no split or merge | split and merge in fixed id and cell order. Byte-repeatable (particles 6.3) | region transfer in fixed order. Repeatable, but momentum is not conserved |
| Mass | exact | exact | exact within `kAmountTolerance` (records plus a limiter on split) |

## 3. Cost per step

Particle work (transfer two ways, grouping, moving) grows with particles. The pressure solve grows with cells and
is the same in A, B and C on the same adaptive grid. Rough figure: NB-FLIP measured more than 60 % of the step in
particle work at 8 per cell with a PCG solve (section 2). So one cell's solve costs about 5 particles' work.
Cost index = particles + 5 × cells, relative to B:

| Scene, 1 cm | today (A, uniform) | A | B | C |
|---|---|---|---|---|
| glass | 1.0 | 1.0 | 1.0 | 0.82 |
| bathtub | 1.9 | 1.6 | 1.0 | 0.67 |
| pool | (does not fit) 7.3 | 3.8 | 1.0 | 0.63 |
| sea | (does not fit) 41 | 27 | 1.0 | 0.62 |

The constant 5 is borrowed from a different solver (PCG, trilinear weights). This design's multigrid and
quadratic weights could move it either way. Read the table as an order of magnitude.

## 4. Implementation and risk

| | A | B | C |
|---|---|---|---|
| Exists in the plans | seeding (I/O 5.1). Nothing else, but the batch plan and the refill assume 8 per cell | split, merge, `AdjustParticleCount`, both-grid checks, tests and Lean proof (particles 5.4 to 5.6, 7, 10); the remap (cell resize 5.4) | the designs had it all: `ApplyGridOnlyRegionToGrid`, `ComputeCarriedVelocity`, `MoveGridOnlyLiquid`, region transfer and border refill, ledger, tests (cell resize 5.5, 5.6, 7). A64 is removing it |
| Missing | the leak is reversed: coarse cells must accept 8^k per cell (transfer, plan) | (1) start on the adaptive grid and seed each start cell at its own size; size the plan on that; (2) the drifted-particle rule (2.2) | the NB-FLIP sharp switch instead of the 2-layer blend; band 3 to 4 cells instead of 6 to 7 (or keep 6 to 7 as a margin); the two A64 gating tests (standing wave amplitude loss; band-edge stability); a planner that counts particles per block by region |
| Risk | memory and cost grow 8^k in deep cells; wrong batch plan | low: conservative by construction, already designed and tested on paper | medium: energy loss on coarse cells (Ando and Batty 2020), edge stability, momentum not carried, two holders of one liquid |
| Built first, extended later | dead end | C can be added later as a region on top of B with no rework: B's split, merge and size levels stay, and C only adds a holder for deep cells | needs B (or A) inside the band anyway |

Start-state change for B: build the start grid with the size rule (depth from the start regions' own shapes, a
distance computed on the CPU before the run), then sample each start cell on a lattice of its own width / 2.
`rest_volume = (cell width / 2)³`. The plan's limits come from that grid. This is a change to I/O 5.1 and step 5.9
only. The particle code is unchanged.

## 5. Recommendation

1. First version: **B**, eight particles per cell at every size, split and merge at size changes as the designs
   already do.
2. Read A64 as **B**. Two changes in the plans: build and seed the start state adaptively and size the plan on it
   (I/O 5.1, step 5.9); add the drifted-particle split/merge rule (particles 5.6, cell resize 5.4).
3. Without change 1 the plan needs A-plus-uniform memory: the bathtub at 0.5 cm needs 64 GB instead of 22 GB.
4. Do not build C in the first version. It saves about one third (18 % to 34 %) in every scene, never the factor
   needed for the pool or the sea, and halving `surface_reach_in_cell_widths` saves as much.
5. The owner's choice does not make mid-size scenes impractical. The pinned surface (A61) and the slab batch rule
   do: the pool fits only at about 3.5 cm, the sea at about 25 cm, under any option.
6. Order after v1, by memory gained: coarse surface cells (A61's later task) and batch pieces cut by columns or
   tiles instead of z-slabs; then C.
7. C becomes worth it when the deep region holds a large share of particles, which happens only if the surface
   band shrinks (coarse surface or reach 3 to 4) and the scene is deeper than about 50 w_min.
8. Build C on top of B with NB-FLIP's sharp switch and its two A64 tests, after the independent review.
9. What would change my mind: the standing-wave test showing B's coarse levels damping much more than the fine
   level (then add Ando's size-dependent blend first); a measured per-particle cost far above 675 B; or the
   rewrite pass choosing a batch layout that holds wide scenes, which would make CPU-side particle count (and so
   C) matter.
10. Uncertainty: block overhead ±30 %, per-particle bytes ±16 %, cost constant ±2×. None of them flips a verdict
    in the 1 cm table except the bathtub at 0.5 cm under C (14.6 GB against 13.05).

## 6. Questions for the owner

1. **Read A64 as B (particle size follows the cell) or A (one size everywhere)?** Recommended: **B**.
2. **Should the run start on the adaptive grid, with each start cell seeded at its own size, and the memory plan
   sized on that start (instead of all-smallest cells with 8 particles each)?** Recommended: **yes**.
3. **Should a particle whose size no longer matches its cell (it drifted across a size change) be split or merged
   at the next resize, with momentum exact and the energy difference reported?** Recommended: **yes**.
4. **After v1, which comes first: (a) coarse surface cells and batches cut by columns, (b) the narrow band C,
   (c) tune the surface reach from 8 down to 4 by test?** Recommended: **(c) now as a test, then (a), then (b)**.

## Sources

Read for this note: Ferstl, Ando, Wojtan, Westermann, Thuerey 2016, Narrow Band FLIP
(https://pub.ista.ac.at/group_wojtan/projects/2016_Ferstl_NBFLIP/nbflip.pdf): sections 2 to 4, Table 1 (pour:
9.36 M → 0.56 M particles, memory 518 → 303 MB; letters 16.6 M → 0.88 M, 857 → 456 MB; teaser 24 M → 1 M
particles, memory halved). Through literature-cell-size.md and literature-practice.md (checked there, not
reopened): Ando and Batty 2020; Ando, Thürey, Wojtan 2013; Adams et al. 2007; Winchenbach 2017 via Xu 2023;
Nielsen and Bridson 2016; Gao 2018; Bridson notes; Hammerquist and Nairn 2017. Seen only in a search abstract:
Sato et al. 2018, Extended Narrow Band FLIP (switching to the grid in arbitrary places,
https://research-explorer.ista.ac.at/record/135). Counting scripts in the session scratchpad, not in the
repository.
