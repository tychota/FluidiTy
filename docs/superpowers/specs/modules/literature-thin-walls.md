# What the papers say about a thin wall that holds liquid (checked 2026-11-15)

Basis of the open choice in `01b-mesh-solids-design.md` section 5.12 (owner answers A85, A89, A90). The question:
a solid wall thinner than a grid cell (a glass, a hull, a sheet, a shell with no thickness) must hold liquid in a
method whose incompressibility comes from a pressure solve on the grid. What do published methods do, and what
do they recommend for contained liquid?

How it was checked. Hu et al. 2018 was read in full text by the author of this file. Three reader agents opened
the other sources as full text (downloaded and converted to text where the file was large) and reported the
sentence each claim rests on. **R** = read in the source, with the section. **M** = from memory, an abstract or a
search snippet only. Sources no one could open are listed as such. Quotes are the readers'.

## 1. Recent work compatible with MLS-MPM (2018 onward; the owner's priority)

| Source | Time stepping | What keeps material from crossing a wall thinner than a cell | Fits our transfers and pressure solve? | Limits it states | Checked |
|---|---|---|---|---|---|
| Hu et al. 2018, MLS-MPM with CPIC, doi:10.1145/3197517.3201293 | Explicit. Water is weakly compressible ("Weakly compressible water is done as in [Tampubolon et al. 2017]", Table 3). No pressure solve | A coloured distance field on grid nodes: unsigned distance, and per nearby surface an affinity (near, with a valid projection onto a triangle) and a tag (which side), splatted from sample points on the triangles to the 3 x 3 x 3 nodes around each (5.2). Particles rebuild distance, normal and colour from the nodes in their kernel; a particle's colour "will persist after being obtained, until all nodes in p's kernel lose affinities" (5.3.3). Compatible: "for all surfaces shared by the particle and the grid node, all tags are the same" (5.3.4). Particle-to-grid goes to compatible nodes only; grid-to-particle uses a ghost velocity (the particle's own, projected on the boundary) for incompatible nodes (5.4, 5.5). A particle that still penetrates gets "a weak penalty force" (5.3.3) | Transfers: yes, it is our 5.11. Pressure: it has none, so it says nothing about a cell or a face the wall cuts | "it only resolves features at a scale of grid dx. Thus we cannot handle sub-grid level boundary configurations such as sharp corners and narrow gaps as done by Azevedo et al. [2016] with cut-cells. Our compatibility condition ... is a binary decision and essentially grid-aligned" (8). Implicit strong coupling "not formulated" | R (whole paper) |
| Genesis engine, MPM solver (docs and code) | Explicit MLS-MPM | CPIC as Hu 2018: option `enable_CPIC` "to support coupling with thin objects"; "a cell separated from a particle by a rigid geom gets nothing from it" | As CPIC | None beyond CPIC's | R (docs, code comment) |
| Fang, Qu et al. 2020, IQ-MPM, doi:10.1145/3386569.3392438 | Monolithic pressure solve for incompressible liquid with MPM solids | The interface is found from particles ("locate solid particles that are within 0.5 dx to fluid particles", 5.4); a position correction "prevents penetration from drifting" | No: the solid must be a volume sampled by particles at grid scale. Scope: "incompressible fluids with volumetric elastic solids" (8) | Stretched solids open gaps: "fluid MPM particles will creep into these gaps" (8). Shells not treated | R |
| Fei et al. 2018, liquid and fabric, TOG 37(4):51 | APIC liquid with a pressure solve; cloth a porous mixture joined by drag | Nothing geometric: impermeable cloth is the limit of "infinitely large drag force" (5.1); fast liquid "can penetrate through the cloth from one side to another" (5.2) | Not a seal | "relies on a relatively fine grid resolution ... fabric thickness is not even a full order of magnitude smaller than a grid cell" (6) | R |
| Guo et al. 2018, MPM thin shells, doi:10.1145/3197517.3201346 | MPM, shell against shell through the grid; no liquid | Grid-scale contact | No | "visible separation if dx is too large ... self-penetration if the resolution is too low" (12) | R |
| Han et al. 2019, hybrid MPM contact, doi:10.1145/3340258 | MPM with collision particles on mesh boundaries | Grid impulses | No | "when the grid resolution is too high ... the method does not prevent collision" | R (through Han's 2020 thesis, ch. 6, not the paper) |
| Li, Fang, Li, Jiang 2022, BFEMP, arXiv:2108.03349 | Fully implicit MPM with finite elements | A barrier between MPM particles and triangles, with a swept test in the line search | Particles only. Names our problem as future work: particles "on different sides of the FEM mesh should not communicate with each other even when the FEM mesh is much thinner than the MPM kernel" | That | R |
| Li et al. 2024, "A dynamic duo of finite elements and material points", SIGGRAPH 2024 | Explicit MPM (weakly compressible fluid) with implicit finite elements | Particles as small spheres, barrier, swept test per particle; cloth and shells hold sand and water | No pressure solve; the grid's mixing across the wall is not addressed | "completely eliminating the penetration issue in an efficient way is yet to be achieved" (6) | R |
| Ando and Batty 2020, octree liquid, doi:10.1145/3386569.3392460 | Pressure solve with face area fractions on an octree: our architecture | Nothing for thin walls. The sizing function has a solid curvature term (Eq. 39) | Same architecture, same gap | "Very thin solids may cause 'cross-talk' artifacts, which could potentially be addressed by adapting a one-sided interpolation strategy (e.g., [Azevedo et al. 2016; Guendelman et al. 2005])" (5) | R |
| Takahashi and Batty 2020, Monolith | Variational pressure, viscosity and contact solve with volume fractions | Thin or sub-cell solids not discussed | - | - | R (search of the text) |
| Lesser, Stomakhin et al. 2022, Loki, doi:10.1145/3528223.3530058 | FLIP with a pressure solve | "The Lagrangian Object acts as a collider in the pressure projection" (6.4); wet cloth as Fei 2018 | No leak-proof thin-wall mechanism described | - | R |
| Sun et al. 2025-2026, "Penetration-free solid-fluid interaction on shells and rods", arXiv:2505.12539 | Staggered-grid pressure solve (twice a step), a level-set liquid, an optimisation with a barrier | A barrier between each solid vertex and the liquid level set, with a swept test. Says thin objects are "somewhat 'invisible' to the Eulerian grid" (1); result "the shell remains leak-proof" (5) | Partly: no particles ("Coupling particle-based fluids (SPH, MPM, etc) with solids is not considered", 6) | Solid edges must be shorter than a cell; hydrophobic contact only; dissipative | R (venue not confirmed) |
| Tao et al. 2022, VEMPIC, TOG 41(4):115 | Pressure solve with virtual elements on exact cut cells (Mandoline), PIC transfers | Polyhedral cells that follow the triangles, two sets of unknowns on each cut face; particles ray-cast and reflected. On level-set cut cells: "precludes domains with thin boundaries because fluid on both sides of the boundary will frequently belong to the same cell" (2.2) | No: not a staggered face-fraction solve; 50^3 takes 228 s a step (Table 3) | Smoke only, static solids; free surface is future work (8). "we manually pruned the few smoke particles that leaked through" (7) | R |
| Zhang and Ren 2026, fast VEM fluid, arXiv:2607.17725 | Same family, with multigrid; "up to 100x" faster; an "ultra-thin sheet boundary with a relative thickness of 10^-5" | Cut cells | No | "currently limited to static boundaries. Applying our framework to deforming boundaries or liquid simulations requires frequent grid updates" (7) | R (abstract, Fig. 1, section 7) |
| Panuelos, Grinspun, Levin 2025, arXiv:2507.18460; 2026, arXiv:2602.03988 | Explicit compressible flow on a Voronoi partition clipped by the solid; the 2026 paper couples "arbitrarily thin and codimensional solid geometry" | Cells never span the solid: "no fluid leaks through the solid, even when arbitrarily thin" | No pressure projection; Voronoi construction is "85% of our simulation runtime" | No free surface found in the text | R |
| Lyu et al. 2021, lattice Boltzmann coupling, doi:10.1145/3478513.3480493 | Lattice Boltzmann, no pressure solve | "Double-sided bounce-back ... naturally prevents flow penetration through solid boundaries, even for thin solids" (3.2) | No | Staircase forces on large cells; no free surface shown | R |
| Not found | - | Any extension of CPIC to an incompressible liquid with a pressure solve; any "leak-proof thin wall" paper for FLIP, APIC or MPM liquids with a grid projection. Two searches each | - | - | - |

**Plain verdict.** Recent work compatible with MLS-MPM gives **no answer for a pressure solve**. It either
stays explicit (CPIC, Genesis, the barrier methods) or has a pressure solve that sees only solids of grid size
or more (IQ-MPM, Fei 2018, Ando and Batty 2020, Monolith, Loki). The pressure solves that do see a wall thinner
than a cell are cut-cell solvers shown on smoke with solids at rest, each naming free-surface liquid as future
work. The answer for this solver has to be assembled from parts, and no paper shows the assembly.

## 2. Background: thin walls in a pressure solve (before 2018)

| Source | What it does | Guarantee or claim | Limits | Checked |
|---|---|---|---|---|
| Guendelman, Selle, Losasso, Fedkiw 2005, "Coupling water and smoke to thin deformable and rigid shells", doi:10.1145/1073204.1073299 | **One pressure per cell**, at its centre (4.3). "Neumann boundary conditions ... are enforced at a cell face if the ray between two adjacent cell centers ... intersects an object" (Fig. 5); that face takes the solid's effective velocity over the step. Every interpolation and difference drops points not visible from the reference point (3, 4.1). Liquid particles "need to collide with solid objects to prevent water from leaking" (4.4.2) | "our method prevents the leaking of material across the triangulated surface" (8). Shown with water: a thin cup filled and poured, cloth carrying water, a shell boat | "thin films of water can quickly compress and lose mass against thin solid objects if one is not careful in how the boundary conditions are handled" (4.3); with the effective velocity, "little to no mass loss" (5.1). No number. Says nothing of a cell the shell splits: the part beyond the centre has no unknown (the reader's inference). 5 to 20 minutes a frame in 2005 | R |
| Robinson-Mosher et al. 2008, two-way coupling to solids and shells, doi:10.1145/1360612.1360645 | Same one pressure per cell; "we prevent leaking by forcing the fluid to move with the solid effective velocity"; two-way | "we support fully one-sided discretizations without leaking"; a thin shell boat floating | Stick only at the wall; 2009 follow-up restores slip, still "a voxelized boundary approximation" (as quoted by Azevedo) | R |
| Batty, Bertails, Bridson 2007, doi:10.1145/1276377.1276502 | Face fractions inside a variational pressure solve | Sub-cell gaps between thick solids flow correctly | "rigid bodies are thick enough to have an interior sampled on the grid. For thin rigid bodies, shells in particular, this is violated and the above approach does not work as described" (3.3); ghost pressures named as future work (6) | R |
| Ng, Min, Gibou 2009, doi:10.1016/j.jcp.2009.08.032 | Face fractions from a level set, second-order pressure | - | "limits it to volumetric objects and fluid regions larger than a grid cell width ... each cell contains only a single active region" (Azevedo's words) | M, and R through Azevedo |
| Zarifi and Batty 2017, doi:10.1145/3099564.3099572 | One pressure per cell, face fractions, free-surface liquid, two-way | Works "if the object's thickness is greater than a grid cell width or two" | "prevention of fluid flow through the solid is much more difficult when it has a very small or zero thickness (e.g., cloth)" (7); suggests Azevedo 2016 | R |
| Azevedo, Batty, Oliveira 2016, doi:10.1145/2897824.2925919 | Cells clipped by the triangles into sub-cells; "We assign one pressure to each sub-cell"; a velocity per sub-face; symmetric positive-definite; PIC and FLIP with visibility in the transfers, particle collision as a fallback. 2D and 3D | Correct topology for thin walls and narrow gaps. Of Guendelman's form: "the gap between the two solid boundaries has been entirely collapsed away" | Smoke only, one-way: "should be extensible to free surface flow and two-way coupling" (8). "using FLIP in cut cells leads to instability, particularly for small cells ... we revert to a pure PIC approach in cut cells" (6). Meshing dominates the cost | R |
| Edwards and Bridson 2014; Weber et al. 2015 | Cut cells with several liquid parts per cell; multigrid for Ng's scheme | - | Neither treats thin solids (Azevedo's words) | R through Azevedo |
| Bridson's book, 2nd ed. | Recommends fractions over voxel solids; points to Guendelman for solids thinner than a cell | - | - | M, low confidence |

## 3. Small cells near solids, and production tools

- Refining toward solids is published for **flow detail**, never as the way to seal a wall thinner than a cell.
  Losasso et al. 2004: "we refine near objects since their interactions with smoke will introduce small scales
  features"; for water, a band about the surface (R). Ando et al. 2013: the sizing function has a solid
  *curvature* term, no thickness term (R). Aanjaneya et al. 2017: "Three refinement levels are used, based on
  proximity to the terrain" (R). Ando and Batty 2020: quoted in section 1. No paper read states "refine until the
  wall spans N cells".
- Guendelman 2005 used an octree (160 x 192 x 160 effective for the cup) for the water's surface, with the wall
  closed by rays, not by cells (R).
- Tools. Houdini: "If a collision/container wall is too thin, the particles can collide with the wall but the
  pressure solve step may not see it"; "Increasing the thickness of the collision geometry will fix this"; or
  "decrease the particle scale to the point where ... the collision grid will be fine enough to resolve the thin
  container wall" (R, fluid tips). FLIP Fluids for Blender: "the thinnest or smallest details of the obstacle
  should be at least 1 voxel thick ... but a thickness of at least 2 voxels is best"; thicker walls are "the
  preferred solution as it does not add extra time to simulate"; "Use a thick object for simulation and the
  original thin object for rendering" (R, as relayed by the fetch tool). Bifrost: "Thickness should typically be
  1.0 or more" (R). Mantaflow: a surface thickness setting and a planar flag, no rule (R). Phoenix, RealFlow,
  EmberGen, LiquiGen, Zibra: nothing reliable found.
- So production practice is to **thicken the collider and render the thin one**; none of the tools read stops
  particles at the true surface while the solve uses a thick one, which is the combination that traps a layer.

## 4. What small cells cost (computed here, not published)

Cells within two cells of each face of a wall of thickness `t` and area `A`, about `A (t + 4 w) / w^3`; steps per
simulated second at 2 m/s with a travel limit of half a cell, `2 / (0.5 w)`.

| Wall | Cell width | Smallest cells near the wall | Steps per second |
|---|---|---|---|
| Glass, 2 mm, 0.03 m2 | 1 mm (two cells across) | 1.8e5 | 4,000 |
| Hull, 5 cm, 60 m2 | 2.5 cm | 5.8e5 | 160 |
| Hull, 1 cm, 60 m2 | 5 mm | 1.4e7 | 800 |
| Sheet, 1 mm, 4 m2 | 0.5 mm | 9.6e7 | 8,000 |

Graded neighbours add half as much again to as much again. `tradeoff-particles.md` 1.6 puts a 16 GB card at
about one million cells and 8 to 12 million particles. So two cells across the wall is affordable for the glass
and the thick hull in memory, costs the glass ten times the steps of a 1 cm grid, and is out of reach for a
1 cm hull and any large thin sheet.

## 5. What it means for this simulator

1. **Sides for the transfers are settled practice** (CPIC, read in full): our 5.11 is that method with the side
   taken from the triangles instead of a field on the grid, and a swept stop instead of a penalty. CPIC's own
   limit, features below a cell (corners, narrow gaps), carries over.
2. **For the pressure solve, two published forms exist.** One pressure per cell with whole faces closed where
   the line between two cell centres crosses the wall (Guendelman 2005, Robinson-Mosher 2008): the only form
   **shown with free-surface liquid in thin containers and with a floating shell**, at the price of a boundary
   that follows cell faces and of gaps narrower than a cell being lost. Or one pressure per side of a cut cell
   (Azevedo 2016 and its successors): exact topology, shown only on smoke with solids at rest, unstable with
   FLIP in small cells, costly.
3. **Face fractions alone cannot do it**; their own authors say so (Batty 2007, Zarifi and Batty 2017).
4. **Small cells near the wall are an accuracy knob, not a seal**: no paper uses them to hold liquid, the tools
   name them as the costly alternative, and section 4 shows where they stop.
5. **Growing the collider is tool practice, not a published method**, and the tools pair it with rendering the
   thin wall, not with stopping particles at it.
6. Where evidence is thin: nobody has published closed faces by rays together with MLS-MPM or APIC transfers, face
   fractions elsewhere, and cells of several sizes; Guendelman's mass-loss remark is qualitative; no cut-cell
   liquid with a free surface exists to compare against.
