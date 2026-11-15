# What the papers say about choosing cell sizes (checked 2026-11-15)

A reader agent downloaded each paper below as full text from the URL given, extracted the text, and searched it.
Section numbers and constants come from that text. Items it could not open are listed as such. Anything marked
"from memory, not verified" was not read in a source during this check. This file is input to the owner's
question "what should decide how small the cells are?" (cell resize design 5.3).

## 1. Which criteria, and the exact formulas

| Paper | What it does | Where | URL |
|---|---|---|---|
| Ando, Thürey, Wojtan 2013, tetrahedral FLIP | `S(x) = max(d(x), V(x, min(κ_liquid(x), κ_solid(x), e(x))))`. `S` is a wanted particle radius, which sets the mesh size. `d = |φ|` is the depth (absolute level set). `V` returns its argument inside the camera's view and `r_max` outside. `κ_liquid = 0.8 / (extrapolated curvature of the liquid surface)`. `κ_solid = 1.6 W_smooth(d_solid, r_max) / (extrapolated solid curvature)`, where `W_smooth(x, h) = (1 - |x|²/h²)³`. `e = 30 / ‖strain-rate tensor‖_F`. They used "light mesh smoothing to increase the reliability of any curvature computations" on the surface mesh. "These values were not meticulously tuned and are certainly not optimal" | section 4 "Sizing Functions", Eq. 15; section 5; section 7 Discussion | https://pub.ista.ac.at/~wojtan/projects/2013_Ando_HALSoTM/download/tetflip_fixed.pdf |
| Ando, Batty 2020, octree liquid | `S(p) = γ_φ |∇·∇(φ + φ_solid)| + γ_u sqrt(Σ_i (∂u_i/∂x_i)²)`, units 1/m, with `γ_φ = 4` (no unit) and `γ_u = 3 s/m`. The exception is the 30 m seaplane scene, which uses `γ_u = 0.032` "because the Laplacian term … is independent of spatial scale, but the velocity is dependent on the expected maximal velocity". A cell subdivides only if `|φ| < Δx` and `S(p) > 1/Δx`, so refinement is surface-only. `S` is evaluated only in cells next to the surface, then spread inwards and outwards with a volume-weighted max rule, at most 5 updates per cell. The paper prints only the diagonal derivatives (checked on the typeset page) but says the term "technically also responds to rigid rotations", which those derivatives alone do not | section 6.2, Eq. 37; section 6.3 step 4; section 6.6, Eq. 39 | https://cs.uwaterloo.ca/~c2batty/papers/Ando2020/Ando2020.pdf |
| Aanjaneya et al. 2017, power diagrams + SPGrid | No surface criterion. The octree has 2 or 3 levels, fine near the terrain, the water sources, or "a bounding box of the object", updated every frame. The surface is tracked separately on a uniform 2048³ narrow-band level set | Fig. 1, Fig. 8, section 7 | https://cs.uwaterloo.ca/~c2batty/papers/Aanjaneya2017/high_resolution_liquids.pdf |
| Setaluri et al. 2014, SPGrid | Data structure paper (smoke). Grid adaptation is shown for one dynamic example; the text read does not state the refinement criterion | section 7, Table 3 | https://pages.cs.wisc.edu/~sifakis/papers/SPGrid.pdf |
| Nielsen, Bridson 2016, Bifrost adaptive FLIP (talk) | "Similar to Ando et al. [2013] our criterion is based on distance metrics such as distance to the surface or selected solid boundaries". One depth-first pass decides per voxel whether to keep, coarsen, or refine. Particles are reseeded, never split or merged, and their density "rapidly decays away from refined regions". "The flow needs to be adequately deep for it to be worth it" | sections 4, 6 | https://history.siggraph.org/wp-content/uploads/2022/09/2016-Talks-Nielsen_Spatially-Adaptive-FLIP-Fluid-Simulations-in-Bifrost.pdf |
| Nielsen et al. 2018, Bifrost collocated smoke (talk) | User boxes define the adaptivity in the rocket-trail example. Its multigrid "restricts the tree to be graded" | Fig. 1, section 2 | https://history.siggraph.org/wp-content/uploads/2022/09/2018-Talks-Nielsen_A-Collocated-Spatially-Adaptive-Approach-to-Smoke-Simulation-in-Bifrost.pdf |
| Nielsen et al. 2020, Bifrost auto-adaptivity (talk) | With Ando-style sizing functions, "the scale-factors involved need to be exposed and hand-tuned on a per-simulation basis", which makes them hard to use in production. They switched to a budget: `B = (f(1 - p) + p) B_sparse` with `p = 0.3`. Each candidate (one voxel to refine, or one 4³ tile to coarsen) gets a quality score from "disturbance (the norm of the velocity gradient)", "fog-detail (the absolute value of the second derivatives)", or "interior". The k-th largest score is the cutoff, with "at most … one level at a time" per step | sections 1 to 3 | https://history.siggraph.org/wp-content/uploads/2022/08/2020-Talks-Nielsen_Auto-Adaptivity-An-Optimization-Based-Approach-to-Spatial-Adaptive-For-Smoke-simulation.pdf |
| Autodesk Bifrost docs, Adaptivity attributes | Liquid spatial adaptivity "uses lower-resolution voxels in the fluid's interior" and keeps full resolution "in fast-moving regions, along collision boundaries, and along the free surface". "Delete Exceeding Particles": turn it off "if you see volume loss". No formulas or defaults are given | Spatial section | https://help.autodesk.com/cloudhelp/2022/ENU/Maya-Bifrost/files/GUID-C17149C6-5A7D-46EC-860E-9F6F47C38217.htm |
| SideFX Houdini FLIP Solver docs | "Solve Pressure with Adaptivity" uses coarser pressure "far from the surface". It helps most "at high resolution and in deep tanks" and "doesn't preserve fine detail in coarse regions". The narrow band keeps particles within a voxel bandwidth, and "very fast moving simulations might require a larger bandwidth" | Solver tab; Narrow Band | https://www.sidefx.com/docs/houdini/nodes/dop/flipsolver.html |
| Ferstl et al. 2016, Narrow Band FLIP | Particles are kept only for `φ ≥ -R` with `R = 3h` or `4h`. Particle velocity is used only for `φ ≥ -r` with `r = R - h`, and grid velocity deeper, with a sharp switch (Eq. 3). Particles are added or removed only where `-R ≤ φ ≤ -h`, so the surface does not move | sections 3.1 to 3.3, Eq. 2 to 4, Fig. 6 | https://pub.ista.ac.at/group_wojtan/projects/2016_Ferstl_NBFLIP/nbflip.pdf |
| Gao et al. 2017, adaptive GIMP (read in Gao's 2018 thesis, chapter 3, which covers it) | Finer cells and particles "closer to the free surface or collision boundary", coarser "deep inside". Elastoplastic resampling uses an approximate Manhattan distance `d` to the free surface with three thresholds: no split or merge for `d < d_small`, split for `d_small < d < d_medium`, merge for `d > d_large`. No resampling where a cell has too many or too few particles. Refinement uses "simplified heuristics, such as the proximity to collision objects and/or the free surface"; strain-based criteria were not investigated | sections 3.7.1, 3.7.2, 3.9 | https://asset.library.wisc.edu/1711.dl/G24M44FI7YSL28I/R/file-abf33.pdf |
| Adams, Pauly, Keiser, Guibas 2007, adaptive SPH | Extended local feature size `elfs(x) = min_y {‖x - y‖ + lfs(y)}`, which is 1-Lipschitz, so sizes vary smoothly. Split if `elfs < α r`, merge if `elfs > β r`, with `α = 2` and `β = 3`. View term: split if `elfs + d(view frustum) < α r`. Physics criteria were suggested but not used: "there is often a high correlation between physical and geometric complexity" | sections 4.1, 4.3, Eq. 3 | https://infoscience.epfl.ch/record/149309/files/adams_2007_ASP.pdf?version=1 |
| Winchenbach et al. 2017, continuous adaptive SPH (read only as described by Xu et al. 2023) | Target mass `m_opt = m_base((min(|φ|, |φ_max|) / |φ_max|)(1 - α) + α)`, linear in depth. Per-particle classes by `m / m_opt`: below 0.5, 0.5 to 0.9, 0.9 to 1.1 (left alone), 1.1 to 2, above 2 (split) | Xu et al. section 3.2, Eq. 8 | https://eprints.bournemouth.ac.uk/37975/1/Spatial%20adaptivity%20with%20boundary%20refinement%20for%20smoothed%20particle%20hydrodynamics.pdf |
| Xu et al. 2023, boundary refinement for SPH | Surface-only refinement wastes work on "the vast and calm surface" and misses "energetic flow beneath the surface" (a propeller). They refine by distance to chosen solid objects instead | sections 1, 5.1 | same URL as above |
| Gerris (Popinet), source `src/adaptive.c` | Vorticity cost `|ω| Δ / max|u|`, where `max|u|` is the largest speed in the domain. Gradient cost `sqrt(Σ (λ_c ∂v/∂x_c)²) · dimension`. Thickness cost `1 / thickness` with `cmax = 1/3`, using VOF height functions (thickness apparently in cell widths, my reading). A cell refines if any criterion's cost exceeds its `cmax`, and may coarsen only if every criterion's cost is at or below `cmax / cfactor` | lines 350-354, 545-557, 766-817, 1210-1236 | https://github.com/DaanVanVugt/Gerris/blob/master/src/adaptive.c (mirror; gfs.sourceforge.net wiki is gone) |
| Basilisk (Popinet), `adapt_wavelet` | Error `e = |s - prolongation(parent)|` per field. Refine if `e > max` (and level below maxlevel), coarsen if `e ≤ max/1.5`. 2:1 balance is enforced: refining a cell first refines its coarser neighbours, and a cell whose children have neighbours cannot coarsen | `src/grid/tree-common.h` | http://basilisk.fr/src/grid/tree-common.h |

## 2. Keeping cell sizes from flickering

| Source | Mechanism | Where |
|---|---|---|
| Ando, Batty 2020 | Temporal adaptivity: the sizing field from the previous step is carried with the flow (semi-Lagrangian), then `S = max(R(Δt) S*_advected, S_new)` with `R = T1^(Δt/T2)`, `T1 = 0.9`, `T2 = 0.01 s`. This "avoids flicker" and keeps salient detail. Grading: "adaptivity smoothing" dilates the active cells 3 times per level, coarse to fine. More than one level between neighbours "would increase the algorithmic difficulty and potentially introduce unexpected visual artifacts (e.g., grid-aligned reflections due to steep resolution changes)" | sections 6.1, 6.3 |
| Ando et al. 2013 | Remeshes every 10 steps. Suggests "taking more temporal information into account" as future work | sections 4, 7 |
| Adams et al. 2007 | Split and merge thresholds 2 and 3 ("to prevent oscillatory down- and up-sampling, α should be smaller than β"). Resampling every 5th step | sections 4.1, 7 |
| Gerris | Coarsen only when cost ≤ `cmax / cfactor`. The default `cfactor` is 4, error-based uses 2, thickness uses 1.1 | `adaptive.c` lines 301, 685, 818 |
| Basilisk | Dead zone between `max/1.5` and `max` | `tree-common.h` |
| Winchenbach (via Xu) | Mass ratio 0.9 to 1.1 is left alone, and only above 2 is split | Xu section 3.2 |
| Xu et al. 2023 | "Wake preservation": after a particle passes within `h` of a chosen solid, its distance is not allowed to grow for `τ_max`, which delays merging | section 5.1, Algorithm 2 |
| Nielsen et al. 2020 | One level change per step | section 3 |

## 3. Artefacts the authors report

- Ando et al. 2013: "using too sharp of a grading … can place coarse and fine simulation elements too close
  together"; "small particles land in very coarse cells … can get stuck in mid air", with "an overly strong weight
  for such particles during the velocity mapping, resulting in momentum artifacts". Their fix for the numerical
  viscosity that varies with cell size is a per-particle PIC/FLIP blend `ν Δt / r²` (Eq. 14), clamped at 0.1.
  Split and merge near the surface create bumps unless a new particle within 1.25 radii of the surface is moved
  to touch the surface. Particles near the surface (within 6 radii) may only be moved along the surface.
- Ando, Batty 2020: a naive first-order `p = 0` at surface T-junctions "clearly manifests objectionable artifacts"
  (Fig. 13). Their symmetric form (Eq. 25) does not, even though 11 % to 33 % of T-junctions are degenerate and need
  clamping (Table 2). Compared with a uniform grid, the adaptive run damps motion "noticeably faster". The causes are
  "a lack of resolution and excessive diffusion on coarse cells": the time step comes from the finest cells, so
  semi-Lagrangian advection diffuses coarse cells over many small steps (section 9.2.2). Advection does not conserve
  mass or momentum (drifting centre of mass), so they use global volume correction (sections 8, 9.2.4). With a
  surface that is chaotic everywhere the sizing "would simply set all cells to the highest resolution" (9.2.5).
  The surface-tension bunny ran slower than the uniform grid (section 8.3).
- Aanjaneya et al. 2017: sub-grid droplets and air pockets seen by the fine level set but not by the octree cause
  "non-physical motions". The 2-level grading restriction is needed for their stencil table (section 8).
- Ferstl et al. 2016: blending particle and grid velocities by particle density (Eq. 2) "leads to inherently
  unstable simulations" and over-estimates motion. A smooth transition "does not increase the quality". Shrinking the
  level set by more than `h` "can lead to spurious holes". Volume conservation is "equivalent to regular FLIP".
- Gao 2018 thesis (AGIMP): a mismatch between particle size and cell level "can easily lead to numerical
  fracture". In energetic motion, sparse interior particles are "driven to the surface, which diminishes surface
  details". Refining the whole surface gave only a "modest speedup". Splitting copies velocity and the deformation
  gradient and shares mass equally, so mass and momentum are kept.
- Adams et al. 2007: resampling near the surface causes "popping or increased blobbyness" with standard particle
  surfaces, which they fix with a carried particle-to-surface distance.
- Bifrost docs: deleting particles in coarse voxels can cause "volume loss".

## 4. Reported cost of re-meshing

| Source | Cost |
|---|---|
| Ando et al. 2013 | Remesh every 10 steps. Mesh generation is "still a bottleneck … mostly serial" (section 7, Fig. 9) |
| Ando, Batty 2020 | Per step: sizing evaluation 1.9 to 4.2 s and octree construction 0.6 to 1.3 s, out of 9.3 to 22 s (Table 2), so about 20 % to 27 % of the step |
| Aanjaneya et al. 2017 | Grid adaptation 5 s of a 58 s step (Table 2) |
| Setaluri et al. 2014 | Grid adaptation 111 s of a 756 s step; it "initializes a new pyramid every time" (Table 3, section 8) |
| Adams et al. 2007 | Feature size plus resampling 0.13 to 7.1 s per frame against 1.5 to 81 s per frame for the whole run (Table 1) |
| Nielsen, Bridson 2016 | Adaptive 12.4 s per frame against 37.6 s uniform Bifrost (Table 1) |

Pattern: every production and graphics solver read here refines by distance: to the free surface, to chosen
solids, or to user regions or camera. Distance is the cheap, robust part of the rule. Detail criteria come second:
surface curvature (Ando 2013, Ando 2020), velocity gradient or strain (Ando 2013, Ando 2020, Nielsen 2020),
vorticity normalised by the largest speed (Gerris), and film thickness (Gerris). They need constants that the
authors admit are untuned and scale-dependent, and Bifrost replaced them with a cell budget. Stability against
flicker always comes from three things: a gap between the split and merge thresholds (factor 1.1 to 4, typically
1.5), a time memory (temporal max with decay, a wait, or an interval between remeshes), and grading of at most one
level between neighbours. Deep liquid is coarse in all of them. What is lost there is energy (Ando 2020) and
subsurface detail (Xu 2023), not mass.

## 5. For this simulator

The current rule (cell resize design 5.3) uses depth to the surface with an 8-cell reach, distance to solids
with a 2-cell reach, surface and pockets at the smallest size, split at once, and merge after `2w ≤ 0.8 × wanted`
holds for 0.25 s. That rule is the literature's distance criterion, with hysteresis stronger than Ando's and
comparable to Adams'. The owner's extra criteria change it as follows.

**(a) Combination.** Use Ando 2013's shape, a minimum of wanted widths with a lower bound from depth, with every
term written as a width in metres, evaluated per cell from committed fields:

```
w_band   = w_min * max(1, depth / (8 * w_min))                         (current design)
w_solid  = w_min * max(1, d_solid / (2 * w_min))                       (current design)
w_shear  = c_u * U_ref / max(‖∇u‖_F, tiny),   c_u = 0.25 to start      (new, liquid cells only)
w_curv   = c_k / max(|∇²φ|, tiny),           c_k = 0.25 to start      (new, only if (b) is adopted)
w_user   = region width or +inf                                        (optional)
wanted   = clamp(min(w_band, w_solid, w_shear, w_curv, w_user), w_min, w_max)
```

`c_k = 0.25` is `1/γ_φ` of Ando 2020: a sphere of radius `R` has `|∇²φ| = 2/R`, so it gets `R/8` cells, about 16
across its diameter. `U_ref` must make the shear term scale-free, which neither Ando 2020 (3 against 0.032) nor
Nielsen 2020 managed. Use `U_ref = max(largest committed liquid speed, sqrt(|g| * w_max))`. That is one reduction,
exact in any order, and the step already computes the largest speed. In words, `w_shear` says that the velocity
change across one cell must stay below `c_u` times the typical speed, as Gerris' `|ω| Δ / max|u|` does.
`c_u = 0.25` is a starting guess, not from a paper. Compare the cell width with the precomputed `w_min · 2^k`,
never with `log2`, so CPU and GPU take the same branch. Use the strict rounding contract of the shared formula
headers (no FMA contraction in this function).

**(b) Curvature.** Curvature matters only if a surface cell may be wider than `w_min`. The current design pins the
surface, pockets, and solids to the smallest cells, and design 5.3 item 7 uses that to keep every size change in
bulk liquid. With the surface pinned, `w_curv` has no effect. To act on curvature you must allow coarse surface
cells, and Ando 2020 shows what that costs: a free-surface pressure condition that knows about T-junctions (their
Eq. 25), and particle resampling at the surface that does not make bumps (Ando 2013's tangent-only correction and
re-placement; Adams' popping). Recommendation: v1 keeps the surface pinned. Curvature is a v2 option, "surface
cells may be up to k = 1 or 2 levels coarser where `w_curv` and `w_shear` allow". For v2:
- Compute the curvature from a smoothed distance, not from `distance_to_surface`, which is a union of particle
  balls and reads curvature at particle scale.
- Use `|∇²φ|` with a 7-point stencil on cells within 2 cells of the surface, clamped to `1 / w_min`.
- Spread outward by a max over the 26 neighbours, 2 passes. That is order-independent, unlike Ando's averaged
  spreading.
Curvature only needs to be bounded to a factor of 2, because it only picks a level. It does not need to be exact.

**(c) Velocity measure.** Use the Frobenius norm of the velocity gradient, the full 3×3 matrix from the staggered
faces, cell-centred. Its symmetric part is Ando 2013's strain rate, and it also catches vorticity. Do not use
Ando 2020's diagonal-only sum, which misses pure shear and rotation. Speed alone is a poor criterion: a uniform
translation is represented exactly on any cell size, and speed is already used for the step duration. Bifrost
keeps fast regions fine, but gives no formula. A linear shear field is also exact on coarse staggered cells. If
the shear term refines too much, the next step is Basilisk's wavelet error on velocity (the difference from the
parent's prolongation), which ignores linear fields.

**(d) Hysteresis, as a rule.**
- Refine at once when `w_cell > wanted`.
- Merge 8 siblings only when `2 w ≤ 0.8 × wanted` holds for them and every touching cell, without a break, for
  `merge_wait_duration = 0.25 s` of committed time and at least 8 committed steps.
- Never merge a cell touching air, a pocket, gas, or a solid, or within `8 w_min` of the surface.
- Change at most one level per cell per resize.
- Have the shear and curvature terms read their maximum over the last `merge_wait_duration`. A per-cell held value
  is enough: the merge wait already gives Ando 2020's temporal memory without advecting a sizing field. If the
  wait proves too short for fast features, carry a decaying fineness on particles: `f ← max(f_now, 0.9^(Δt/0.01 s) f)`.

**(e) Deep refinement.** Refining only near the surface is the cheapest choice and the literature default: Ando
2020 subdivides only where `|φ| < Δx`, Ando 2013 bounds the size below by depth, and Houdini and Bifrost coarsen by
depth. Two costs are reported. Energy damps faster in coarse regions (Ando 2020 §9.2.2). Energetic flow under a
calm surface is missed (Xu 2023). The shear term in (a) is the answer for the second. For the first, the damping
Ando reports comes from semi-Lagrangian advection on coarse cells, which is exactly the situation of this design's
grid-only deep region. Ando 2013 avoided it because velocity lived on particles.

**(f) Tests.**
1. Conservation across a resize: the liquid mass, the momentum, and the record-plus-particle totals are equal
   to the byte before and after a split and a merge (AGIMP and Ando 2013 conserve by construction; check ours).
2. A calm flat pool reaches the biggest allowed cells below `8 w_min` within `merge_wait_duration` plus the merge
   cascade, and then stays put for 10 s with no split or merge (flicker check).
3. A droplet pinch-off and a thin sheet stay at `w_min` through the neck: no merge within `8 w_min` of either face.
4. A uniformly translating block of liquid (no gravity) gets no split from `w_shear`. A rigidly rotating one does.
5. The same scene in one piece and in batches with different cuts gives equal size levels to the byte, because
   every term is a local stencil plus exact min or max reductions.
6. CPU and GPU give equal size levels on the shear test scene.
7. A sharp-grading check: a splash landing in a coarse region shows no stuck particles and no momentum jump
   (Ando 2013's reported artefact).

## 6. Not opened

- Losasso, Gibou, Fedkiw 2004 (ACM returned 403, the physbam and typeset.io copies are gone or blocked). What is
  known comes only from search snippets of the abstract and text: water refines "near the interface", and smoke near
  objects, by optical depth, and by "vorticity concentration". Their thresholds were not seen.
- Popinet 2003 (Gerris JCP) and Popinet 2009 (surface tension JCP): HAL blocked the automated fetch and
  gfs.sourceforge.net no longer serves the PDFs. The Gerris and Basilisk source code was read instead (section 1).
  Whether Popinet 2009 refines by `κΔ` was not seen. Gerris examples with a curvature-times-cell-size function are
  from memory, not verified.
- Winchenbach, Hochstetter, Kolb 2017: not opened. Its sizing formula is quoted from Xu et al. 2023, and the thesis
  PDF (126 MB) was not downloaded.
- Gao et al. 2017 AGIMP as published in TOG: the thesis chapter was read instead.
- Sato et al. 2018 (extended narrow-band FLIP), Hong et al. 2009, Söderström et al. 2010 (grid-aligned
  reflections at steep resolution changes, cited by Ando 2020): not opened.
