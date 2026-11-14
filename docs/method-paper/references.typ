// The reference list of the method paper.
//
// Every entry, DOI and URL is copied character for character from the old paper's list
// (`git show main:docs/papers/fluidity-method.typ`, section "References"), with its number. Code comments copy
// their DOIs and URLs from here, never from memory. An entry is added at the end with the next number.

[1] Jiang et al. (2015). #link("https://media.disneyanimation.com/uploads/production/publication_asset/105/asset/apic-tec.pdf")[The Affine Particle-in-Cell Method]. DOI 10.1145/2766996.

[2] Ding, Shinar, Schroeder (2020). Affine particle in cell method for MAC grids and fluid simulation. J. Comput. Phys. 408, 109311. #link("https://escholarship.org/uc/item/0vc0g86n")[Publication record], DOI 10.1016/j.jcp.2020.109311; #link("https://www.cs.ucr.edu/~craigs/papers/2019-mac-apic/paper.pdf")[author preprint].

[3] Hu et al. (2018). A Moving Least Squares Material Point Method with Displacement Discontinuity and Two-Way Rigid Body Coupling. DOI 10.1145/3197517.3201293.

[4] Fei et al. (2021). #link("https://raymondyfei.github.io/asflip/main_reduced.pdf")[Revisiting Integration in the Material Point Method], §5.3. DOI 10.1145/3476576.3476674.

[5] Wang et al. (2020). A Massively Parallel and Scalable Multi-GPU Material Point Method (Claymore). DOI 10.1145/3386569.3392442.

[6] Wang, Sun, Zhu (2026). #link("https://arxiv.org/pdf/2604.18886")[Matrix-Free Multigrid with Algebraically Consistent Coarsening on Adaptive Octrees].

[7] Klár et al. (2016). Drucker–Prager Elastoplasticity for Sand Animation. DOI 10.1145/2897824.2925906.

[8] Batty and Bridson (2008). #link("https://www.cs.ubc.ca/labs/imager/tr/2008/Batty_ViscousFluids/")[Accurate Viscous Free Surfaces for Buckling, Coiling, and Rotating Liquids].

[9] Larionov, Batty, Bridson (2017). #link("https://cs.uwaterloo.ca/~c2batty/papers/Larionov2017/Larionov2017.pdf")[Variational Stokes: A Unified Pressure-Viscosity Solver].

[10] Hyde et al. (2020). An Implicit Updated Lagrangian Formulation for Liquids with Large Surface Energy. DOI 10.1145/3414685.3417845.

[11] Sancho et al. (2024). The Impulse Particle-in-Cell Method. DOI 10.1111/cgf.15022.

[12] Ihmsen et al. (2012). Unified Spray, Foam and Bubbles for Particle-Based Fluids. DOI 10.1007/s00371-012-0697-9.

[13] Bhattacharya, Gao, Bargteil (2011). A Level-set Method for Skinning Animated Particle Data. DOI 10.1145/2019406.2019409.

[14] Zhao, Shinar, Schroeder (2025). #link("https://www.cs.ucr.edu/~craigs/papers/2025-surface-reconstruction/paper.pdf")[Fast reconstruction of implicit surfaces using convolutional neural networks]. DOI 10.1145/3747856. Output reconstruction candidate; in-simulation curvature suitability is not established.

[15] Fei, Huang, Gao (2021). Principles towards Real-Time Simulation of Material Point Method on Modern GPUs. arXiv:2111.00699.

[16] NVIDIA. #link("https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html")[CUDA Programming Guide: CUDA Graphs].

[17] Ferstl et al. (2016). Narrow Band FLIP for Liquid Simulations. DOI 10.1111/cgf.12825.

[18] Stomakhin et al. (2014). Augmented MPM for phase-change and varied materials (MPM on a staggered grid). DOI 10.1145/2601097.2601176.

[19] Stomakhin et al. (2013). A Material Point Method for Snow Simulation. DOI 10.1145/2461912.2461948.

[20] Jacobsen, Fuhrman, Fredsøe (2012). A wave generation toolbox for the open-source CFD library OpenFOAM (relaxation zones). Int. J. Numer. Meth. Fluids.

[21] Batty, Bertails, Bridson (2007). A Fast Variational Framework for Accurate Solid-Fluid Coupling. ACM TOG.

[22] Ng, Min, Gibou (2009). An efficient fluid–solid coupling algorithm for single-phase flows. J. Comput. Phys.

[23] Gibou et al. (2002). A second-order-accurate symmetric discretization of the Poisson equation on irregular domains. DOI 10.1006/jcph.2001.6977.

[24] Kang, Fedkiw, Liu (2000). A Boundary Condition Capturing Method for Multiphase Incompressible Flow. J. Sci. Comput.

[25] McAdams, Sifakis, Teran (2010). A Parallel Multigrid Poisson Solver for Fluids Simulation on Large Grids. DOI 10.2312/SCA/SCA10/065-073.

[26] Kwatra et al. (2009). A method for avoiding the acoustic time step restriction in compressible flow. DOI 10.1016/j.jcp.2009.02.027.

[27] Wojtan et al. (2009). Deforming Meshes that Split and Merge. DOI 10.1145/1576246.1531382.

[28] Wojtan et al. (2010). Physics-Inspired Topology Changes for Thin Fluid Features. DOI 10.1145/1833349.1778787.

[29] Zhou et al. (2024). Eulerian–Lagrangian Fluid Simulation on Particle Flow Maps (PFM). DOI 10.1145/3658180. Related: LFM (Sun 2025, 10.1145/3731180), Cirrus (Wang 2025, 10.1145/3731190), Compressible Flow Maps (Chen 2025, 10.1145/3731192).

[30] Yu and Turk (2013). #link("https://faculty.cc.gatech.edu/~turk/my_papers/sph_surfaces.pdf")[Reconstructing Surfaces of Particle-Based Fluids Using Anisotropic Kernels]. ACM Trans. Graph. 32(1).

[31] Huang et al. (2023). #link("https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Neural_Kernel_Surface_Reconstruction_CVPR_2023_paper.pdf")[Neural Kernel Surface Reconstruction]. CVPR. #link("https://github.com/nv-tlabs/NKSR/blob/public/LICENSE.txt")[Implementation license].

[32] Shen et al. (2023). Flexible Isosurface Extraction for Gradient-Based Mesh Optimization. #link("https://github.com/nv-tlabs/FlexiCubes")[FlexiCubes project and code].

[33] Brochu and Bridson (2009). #link("https://www.cs.ubc.ca/labs/imager/tr/2009/eltopo/eltopo.html")[Robust Topological Operations for Dynamic Explicit Surfaces]. El Topo author project.

[34] #link("https://lean-lang.org/doc/reference/latest/Elaboration-and-Compilation/")[Lean reference: elaboration and kernel checking]; #link("https://leanprover-community.github.io/mathlib4_docs/Mathlib/LinearAlgebra/Matrix/PosDef.html")[mathlib matrix positivity]. The proof project pins Lean 4.34.1 and mathlib; the independent experimental checker and exporter are pinned in the proof tooling.

[35] Chen, Zhou and Zhu (2025). #link("https://cdwj.github.io/projects/neural-pls-project-page/static/pdfs/neural-pls.pdf")[A Neural Particle Level Set Method for Dynamic Interface Tracking]. DOI 10.1145/3730399.

[36] He et al. (2026). #link("https://proceedings.mlr.press/v306/he26w.html")[Hermite-NGP]. #link("https://github.com/jinjinhe2001/hermite-ngp")[Official implementation].

[37] #link("https://markjgillespie.com/Research/SubgridMarchingTets/")[Subgrid Marching Tetrahedra] (2026). DOI 10.1145/3811358. #link("https://github.com/hbaktash/subgrid-marching")[Official implementation].
