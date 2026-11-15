# What the papers say about coupling (checked 2026-11-14)

A reader agent opened each paper below as full text from the URL given and reported where each claim comes
from. Items it could not open are listed as such. This file is the basis of owner decision A18.

## 1. One grid velocity, or one per material?

| Paper | What it does | Where | URL |
|---|---|---|---|
| Stomakhin et al. 2014, Augmented MPM | One shared staggered-grid velocity for all materials; "difficult to represent sharp interfaces between materials", "we focused on sticky boundaries… free-slip… future work" | Discussion and Limitations; Fig. 9 | https://alexey.stomakhin.com/research/siggraph2014_melt.pdf |
| Tampubolon et al. 2017, sand and water | Two grids, one per species, joined by a drag term solved semi-implicitly (explicit would need steps of 1e-6 to 1e-5 s) | abstract, section 4.3, Fig. 7 | https://math.ucdavis.edu/~jteran/papers/PGKFTJM17.pdf |
| Gao et al. 2018, fluid and sediment | Two grids; mixture pressure projection for the fluid, explicit sediment sub-steps handing drag back | section 3, Fig. 5 | https://mingg13.github.io/papers/2018_Fluid_Sediment.pdf |
| Hu et al. 2018, MLS-MPM with CPIC | One grid; a coloured distance field decides which grid nodes a particle may exchange with near cuts and rigid bodies; says a single velocity field makes separating materials hard; no implicit strong coupling | sections 5.3.4 to 5.4, Related Work, Limitations | https://yuanming.taichi.graphics/publication/2018-mlsmpm/mls-mpm-cpic.pdf |
| Han et al. 2019, hybrid MPM | "to prevent numerical cohesion… we adopt two separate background MPM grids"; contact by impulses at the end of the step | Coupling with traditional MPM; section 4 | https://math.ucdavis.edu/~jteran/papers/HGGWJT19.pdf |
| Fang et al. 2020, IQ-MPM | Separate solid and fluid grids; only the normal velocity is made continuous, inside one coupled pressure solve; shared-grid coupling "is inherently restricted to sticky and no-slip interactions" | sections 3, 4.1, 6.1 | https://www.math.ucla.edu/~cffjiang/research/fang2020iqmpm/fang2020iqmpm.pdf |
| Bardenhagen et al. 2001 | Each body has its own velocity field; friction when approaching under compression, free separation otherwise; cost linear in the number of bodies | Eqs. 6 to 9, p. 512 | https://collections.lib.utah.edu/dl_files/27/85/278547f0c8de2b3e321b940147ea2c64d0879492.pdf |
| Boyd and Bridson 2012, MultiFLIP | Separate liquid and air velocities on the same faces, one pressure projection of the combined flow | section 3 | https://www.cs.ubc.ca/~rbridson/docs/boyd-tog2011-multiflip.pdf |

Pattern: one shared field gives sticking and mixing for free; separate fields allow sliding and separation, cost
about twice the grid data, and must be joined inside the pressure solve when the materials interact strongly.

For this simulator: ink in water, one field (a concentration carried by the water); oil on water, one field
with a variable-density pressure solve; water over snow and water against a springy solid, separate fields
joined in the pressure solve; smoke above water, separate fields (the water sees the gas as a pressure
boundary, the gas sees the water surface as a moving solid).

## 2. Gas pressure from the previous step, or solved with the liquid?

- Aanjaneya 2013 (thesis, same title as the JCP paper): with the bubble pressure handed to the water solve as a
  fixed value and updated afterwards, a bubble that should rest "quickly goes unstable at lower grid
  resolutions"; measured convergence orders below one (Table 2.1); concludes a monolithic formulation is
  needed. https://stacks.stanford.edu/file/druid:pk959jq5605/thesis-augmented.pdf
- Patkar et al. 2013: sub-grid bubbles coupled monolithically "for stability reasons".
  https://pages.cs.wisc.edu/~aanjneya/papers/HLEFBGD/subgrid_bubbles.pdf
- Robinson-Mosher et al. 2011: partitioned coupling is most fragile when neither side dominates.
  https://www.cs.ucr.edu/~craigs/papers/2010-spd-sfi/paper.pdf
- Banks, Henshaw and Schwendeman 2014: the traditional partitioned scheme is stable only when the added mass
  is below the structure's mass per area, and then only under a step limit; otherwise "unconditionally
  unstable". https://arxiv.org/pdf/1308.5913

Lagging is harmless for gas open to the atmosphere and for a low-Mach gas above a free surface (the liquid
dominates). It is not harmless for enclosed pockets and bubbles: each enclosed pocket goes into the liquid
pressure solve as one extra unknown with its compliance `V / (gamma * p * dt^2)`.

Not opened: Aanjaneya et al. JCP 2013; Wretborn et al. 2025; Causin et al. 2005.

## 3. A deformable solid's response: inside the pressure solve, or one step later?

- Batty et al. 2007: the solid's inertia is inside the pressure solve (section 2, Eq. 2); handles a body at 0.1
  times the fluid density. https://www.cs.ubc.ca/labs/imager/tr/2007/Batty_VariationalFluids/variationalFluids.pdf
- Robinson-Mosher et al. 2008 and 2011: fully implicit, stable for large density ratios; weak schemes have
  "stability and accuracy issues". https://www.cs.ucr.edu/~shinar/papers/2008_solid_fluid.pdf
- Zarifi and Batty 2017: weak coupling needs "restrictively small time steps"; puts the implicit solid in the
  coupled projection. https://cs.uwaterloo.ca/~c2batty/papers/Zarifi2017/Zarifi_2017.pdf
- IQ-MPM (Fang et al. 2020): the solid's shape-changing elasticity first, then one coupled pressure solve
  holding the solid's inertia and half of its volumetric response (sections 4.1, 6.1).

For this simulator: the solid's inertia, and its volumetric give, belong inside the liquid pressure solve. Snow
is lighter than the water it displaces, which is the regime where a lagged force is unconditionally unstable
by Banks' condition. The IQ-MPM split is the practical form.

Not opened: Bardenhagen et al. 2000; Takahashi and Batty 2020.
