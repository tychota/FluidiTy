# Neural surface reconstruction: verified evidence and first comparison

Verified 2026-11-07. Research only; no production implementation or adoption decision.

## Recommendation

Start with a fixed particle cache and one geometric reconstruction reference. Reproduce the published Zhao CNN on the same cache using all particles, then compare an explicitly band-limited reconstruction against that full-data result. Keep neural architecture, training and simulation-band changes out of this first comparison. A reconstruction dependency bound is a constraint to measure against available cached data; it does not justify automatically widening the simulation particle band.

## Primary artifact identity and availability

The exact [UCR 2025 paper](https://www.cs.ucr.edu/~craigs/papers/2025-surface-reconstruction/paper.pdf) is **Chen Zhao, Tamar Shinar and Craig Schroeder, “Fast reconstruction of implicit surfaces using convolutional neural networks,” PACMCGIT 8(4), Article 51, August 2025**, DOI [10.1145/3747856](https://doi.org/10.1145/3747856). The running page headers unexpectedly say “Trovato et al.”; the first-page author list and formal citation are authoritative. The prior algorithm is Zhao et al. 2024.

The exact UCR PDF was downloaded and inspected. Its SHA256 was `d314b6455cea5b680dcdf78737d5a021e2657e029ce1b5aac8f935e0355a81a2`.

The [published source archive](https://www.cs.ucr.edu/~craigs/papers/2025-surface-reconstruction/surface_reconstruction_fast.zip) was downloaded and inspected (202,053,228 bytes; SHA256 `bc4304606c6b378bfd06dec928025e38d8cf6fd767f70a43d8bdc0f56c3e16ca`). Paths below are relative to the archive root `surface_reconstruction_fast/`. Includes C++/CUDA/Python sources, `models/test_nnf.pth`, `models/test_nnf.onnx`, and ten dambreak particle frames. No CUDA build or inference was run in this verification.

License observations: the paper states CC BY 4.0. The inspected source ZIP contains no standalone LICENSE or PHYSBAM_COPYRIGHT file; several bundled files refer to a PhysBAM license. `README.txt` requires a separate PhysBAM-0/PhysBAM-1 download. A separate explicit license for the model weights was not found in this archive. Software and weight license status has not been established by this check.

## Inspected files

- `README.txt`: dependencies, pretrained-model availability, and later Houdini polygon conversion.
- `cnnnet.py`: architecture and parameter count.
- `models/test_nnf.onnx`: encoded input/output shape.
- `eval_model.sh`: shipped stride, radius, grid spacing and batch settings.
- `evaluate.cpp`: kernel accumulation, density interpolation, staggered grids and pruning.
- `construct_features.cpp` and `parallel_slicing.cu`: patch indexing, batching and output scaling.

No archive, pretrained weights or third-party source files are copied into this repository.

## Architecture and artifact checks

`cnnnet.py` defines four unpadded 2×2×2 convolutions with channels 1→64→128→256→512, ReLU activations, and a size-2 max pool after the second convolution. Input size evolves 8→7→6→3→2→1. The MLP is 512→256→128→1 with Tanh after the first two linear layers. Including biases, this is **1,542,081 parameters**, computed directly from those layer dimensions.

The shipped ONNX protobuf explicitly encodes input `[1024,1,8,8,8]` and output `[1024,1]`; no dynamic batch dimension is encoded there. `parallel_slicing.cu` launches 1024 blocks with 8×8×8 threads each. `eval_model.sh` sets batch size 1024 and pads the last batch in `construct_features.cpp`. Consequently the published runtime path needs coordinated batching changes if arbitrary batch sizes are introduced. This is an algorithm reference whose native dependencies and batching still require integration work. The README's native path requires PhysBAM, LibTorch/CUDA, TensorRT and optional OpenVDB; it describes CUDA 12.8/driver 570 as tested.

## What the 9 h_N claim actually means

The bound can be derived from the inspected source for the **shipped evaluation configuration**, rather than assuming that each use of “dx” means the same thing.

Define `d_phi = CLI -dx`, and `h_N = stride * d_phi`. `eval_model.sh` uses stride=2 and `m0_radius=2.5`, so `d_phi=h_N/2`. The output level-set grid and intermediate density/feature accumulation grid have spacing `d_phi`; each of the eight staggered inference subgrids has spacing `h_N`. Numerical values in the sample script are `d_phi=.005`, `h_N=.010`; these do not define FluidiTy's simulation resolution.

For an output query q, source indexing gives eight feature sample positions per axis:

`q + (j - 3.5) h_N`, for `j=0,...,7`.

This follows jointly from `lb_m0 = lb-(fw-.5)*d_phi*stride`, `fw=4`, the interleaved subgrid gather in `Construct_Features`, and the array lower-index offset in `parallel_slicing`. Thus the patch maximum offset is **3.5 h_N**, not 4 h_N and not 3.5 times the fine output spacing.

The feature at each patch position accumulates particles using a tensor-product quartic kernel with axis support radius **2.5 h_N**. Its per-particle normalization is the density interpolated from neighboring nodes of the fine grid. Those nodes lie less than or equal to **d_phi** away in each axis. Each density node itself depends on particles within another **2.5 h_N**. The exact dependency chain is:

`query → feature sample → contributing particle → density interpolation node → density-contributing particle`.

A conservative axis-aligned bound is therefore

`R_dep,∞ ≤ 3.5 h_N + 2.5 h_N + d_phi + 2.5 h_N = 9 h_N`.

This is a source-derived conservative **L∞ neighborhood around each evaluated grid query**, not a paper theorem, Euclidean radius, guarantee that a chosen particle band works, or minimum necessary support. At zero-valued kernel endpoints some dependencies vanish. A corresponding Euclidean enclosing ball could require √3 times this radius. Different stride, grid layout, interpolation, features or kernel parameters require rederivation; in particular the code's ten-fine-node accumulation stencil is specialized to the supplied stride-2 configuration.

The bound concerns evaluating the scalar reconstruction at grid nodes. Candidate-query selection, complete extraction cells and any additional postprocessing need their own halos. If inference nodes are selected using a surface proxy with uncertainty epsilon, support accounting must include that geometric uncertainty. A particle band measured using Euclidean distance to a surface is also not automatically equivalent to an L∞ query halo.

The density denominator is the reason retaining only particles that directly contribute to a patch is insufficient. If material particles are removed or replaced with coarse representatives, reproducing the normalization and features is an experimental question. Synthetic caps, hand-filled interior values or direct-density substitutions change the input distribution unless verified against the full cache.

The implementation also prunes nodes using occupied-cell counts: `acc_radius=4`, inner empty-region test and outer threshold test `threshold=410`, and assigns skipped nodes ±2 h_N. Those decisions must remain matched in full/band comparisons. They are not a general proof that missing bulk can be ignored.

## Timing scope

Section 5.2 and Figure 8 confirm one 2M-particle frame reconstructed in **2.21 s**, versus **72.3 s** for the predecessor, using a 32-core AMD Ryzen Threadripper PRO 5975WX, 512 GB RAM and NVIDIA RTX 4090. Of that total, **1.01 s** is GPU model inference, **0.90 s** is input/output and loading the model, and **0.30 s** remains in the CPU portion. The paper's **1.31 s** omits those disk/model-loading costs. These numbers are reference-platform observations, not FluidiTy targets or browser/native equivalence claims.

The executable writes level-set arrays/VDB, and the README subsequently instructs users to convert VDB to polygons in Houdini. The quoted timing therefore must not be presented as verified end-to-end delivery of an explicit polygon mesh, including all extraction/export work. The paper separately reports roughly 5× batch-mode speedup; do not substitute the single-frame 33× figure for the batch comparison.

## Other proposed methods

[Hermite-NGP](https://proceedings.mlr.press/v306/he26w.html) is a real ICML 2026 publication; [arXiv:2605.24774](https://arxiv.org/abs/2605.24774) is its matching preprint. It augments hash-grid values with derivatives for analytic PDE operators. Its [official implementation](https://github.com/jinjinhe2001/hermite-ngp) has an MIT license, CUDA/PyTorch code, and PDE/SDF benchmarks. This supports a possible later representation experiment; it does not establish a trained particle-to-surface replacement or an end-to-end dynamic-fluid meshing result comparable to Zhao.

[A Neural Particle Level Set Method for Dynamic Interface Tracking](https://cdwj.github.io/projects/neural-pls-project-page/static/pdfs/neural-pls.pdf) is directly relevant to dynamic interfaces: oriented particles both track the interface and seed neural training, coupled to multiresolution hash features and a coarse sparse distance grid. It changes the tracking/training representation. The inspected [author project page](https://cdwj.github.io/projects/neural-pls-project-page/) links the paper and video but provides no code link; code and software licensing were therefore not established here. Its author PDF contains placeholder publication DOI metadata; the [Crossref DOI record](https://api.crossref.org/works/10.1145/3730399) independently confirms the title, Chen/Zhou/Zhu authorship and publication date 2025-05-09 for [10.1145/3730399](https://doi.org/10.1145/3730399).

[Subgrid Marching Tetrahedra](https://markjgillespie.com/Research/SubgridMarchingTets/) is a real SIGGRAPH/TOG 2026 paper, DOI 10.1145/3811358, with [MIT-licensed C++/Python code](https://github.com/hbaktash/subgrid-marching). It extracts a mesh from possibly multiple intersections along tetrahedron edges, admitting subcell sheets. It does not itself recover trustworthy particle geometry or infer missing intersections from a coarse scalar grid. It belongs after a representation and reliable edge-intersection strategy have been selected, rather than replacing the particle-to-field stage.

## First evidence to collect

Use identical saved particle snapshots for a geometric reference, full-data published CNN, and band-limited published CNN. Include smooth bulk surfaces, thin sheets, droplets, wall contact and rapid topology changes. Report field/zero-set error against full-data CNN separately from quality against geometry, temporal artifacts, lost components, volume bias, latency per stage, peak memory and actual retained support. Distinguish deterministic missing-support error from reconstruction model error. Only then decide whether a wider reconstruction cache, interior summaries, retraining or a different representation is warranted.
