# What papers and vendor practice say about four owner decisions (checked 2026-11-15)

A reader agent checked owner answers A44 (when to advise viscosity 0), A43 and A45 (GPU memory budget and
warnings) and A46 (guided rigid bodies in the viscous solve, and the force report). For each claim it gives the URL
it opened. PDFs were downloaded and converted to text. "Opened" means the agent read the passage. "From memory"
means the claim was not checked in this pass. Each topic ends with the owner's answer, what the sources support,
a recommended decision, and what remains a judgment call. Some things in this file are the agent's own working
and not a finding of any source; they are marked **(reasoning)**.

Context in the designs: liquid particles use quadratic B-spline weights (A7). The particle velocity is a PIC/FLIP
blend: `own_motion_kept` is the FLIP share, called α below. The fitted (affine, APIC) change is not carried
(`carries_fitted_change = false`, transfer design 5.6). `BuildSimulation` allocates every planned buffer up front
and promises `allocated_bytes == estimate.on_processor` (step design 6.3). A free solid's motion
`xi = (V, Omega)` is in world axes, about `origin_at_start`. `FreeMotionLimits` holds three world-axis translation
flags and one `turns` flag.

## 1. When is physical viscosity negligible?

| Source | What it says | Where | URL |
|---|---|---|---|
| Bridson and Müller-Fischer 2007, SIGGRAPH course notes | Linear semi-Lagrangian advection is forward Euler of the modified equation `q_t + u q_x = u Δx q_xx`: "a viscosity-like term with coefficient uΔx". "If we're trying to simulate a viscous fluid, which has plenty of natural dissipation already, then the extra numerical dissipation will hardly be noticed". "Most numerical methods ... introduce errors which can be physically reinterpreted as viscosity", so viscosity is usually dropped. | sections 1.5 and 3.3, Eqs. 3.17 to 3.22 | https://www.cs.ubc.ca/~rbridson/fluidsimulation/fluids_notes.pdf |
| Hammerquist and Nairn 2017, XPIC | A PIC update convolves particle velocities with a filter P every step. For linear shape functions in 1D with uniform particles, `P_k = sinc^4(k Δx)` (sinc(x) = sin(πx)/(πx), k in cycles per cell, Nyquist at 0.5). "For PIC, all the frequencies, except the constant velocity, are damped. This frequency response explains the large diffusion caused by PIC." FLIP does not filter. In a disk-collision test, PIC lost about 15 % of total energy, FLIP on the grid 0.8 %, and XPIC(2) much less than PIC. | Eqs. 29 to 32, Fig. 1, Fig. 5 | https://www.cof.orst.edu/cof/wse/faculty/Nairn/papers/XPICPaper.pdf |
| Jiang et al. 2015, APIC | PIC is "stable, but highly dissipative"; "the dissipation in the original PIC results from" information lost in the transfers; the frequent transfers give "a viscous appearance"; standard PIC loses angular momentum, APIC conserves it. Energy is shown in plots only: no formula and no viscosity value. | abstract, section 1, Figs. 7 and 13 | https://www.math.ucdavis.edu/~jteran/papers/JSSTS15.pdf |
| Fu et al. 2017, PolyPIC | More polynomial modes improve kinetic-energy conservation in the transfers and reduce numerical damping (Fig. 10, energy against time). The authors also say numerical dissipation is often desirable and stabilises large steps. No formula. | sections 1, 7 and 8, Fig. 10 | https://math.ucdavis.edu/~jteran/papers/FGGJT17.pdf |
| Zhang, Bridson and Greif 2015, IVOCK | Splitting advection from projection loses vorticity. "Although FLIP is almost non-diffusive for advection, the velocity-pressure solver nevertheless may dissipate rotational motion." No formula. | section 2, Fig. 2 | https://www.cs.ubc.ca/~greif/Publications/zhangbg2015.pdf |
| Larionov, Batty and Bridson 2017 | Splitting causes "significant loss of kinetic energy and vorticity" at high Reynolds number. The paper targets "strongly viscous liquids, corresponding to moderate to low Reynolds numbers", where splitting viscosity from pressure loses the coupled behaviour (coiling). | section 1 | https://cs.uwaterloo.ca/~c2batty/papers/Larionov2017/Larionov2017.pdf |
| Batty and Bridson 2008 | An implicit viscosity step removes the time step limit. A block with viscosity varying along it: the inviscid end splashes, and waves "damp out as they pass towards the viscous end". The paper gives no criterion for when viscosity is negligible. | sections 3 and 6 | https://www.cs.ubc.ca/labs/imager/tr/2008/Batty_ViscousFluids/viscosity.pdf |

Not opened: Goldade et al. 2019, adaptive variational viscosity (the URLs tried returned HTML). Bridson's book,
2nd edition, is not openly available; its content here is from memory: it says the same as the notes, that
numerical dissipation is usually larger than water's viscosity. No source in this pass gives a measured value of
numerical viscosity for APIC or PolyPIC as a function of cell size, step and velocity.

**Derived numbers (reasoning, from the XPIC filter).** A quadratic B-spline has Fourier transform `sinc^3`, so one
grid-to-particle-to-grid round trip filters by `sinc^6(k h) ≈ 1 - (κ h)^2 / 4`, with κ = 2πk the angular
wavenumber. Diffusion damps by `1 - ν κ² Δt` per step. Matching the two gives
`ν_PIC ≈ h² / (4 Δt)` (linear weights: `h² / (6 Δt)`). In 3D the filter is a product per axis, so the same ν
holds. A PIC/FLIP blend keeps share α of the unfiltered particle velocity, so

```text
ν_num ≈ (1 - α) h² / (4 Δt)        ratio  r = ν / ν_num = 4 N / (1 - α),   N = ν Δt / h²  (A25's number)
```

The leading term is a Laplacian, like viscosity, so r does not depend on scale for resolved features. This
assumes uniform particles of equal mass, as XPIC does. Using the largest step gives the smallest ν_num, because
ν_num grows as the step shrinks. The ratio r is therefore a lower bound, the cautious direction for a claim that
physical viscosity is below numerical dissipation. When α = 1 the model gives ν_num = 0. Pure FLIP still
dissipates through the splitting (IVOCK), but there is no formula for it, so r cannot be claimed. Water at
0.01 m cells, 0.01 s steps and α = 0.95 gives `N = 1e-4` and `r = 0.008`: the transfer damps more than 100 times
more than water's viscosity. Semi-Lagrangian advection gives `ν ≈ u h` (Bridson). It is not used for the liquid
here.

**Scene numbers (standard fluid mechanics, from memory).** Reynolds number `Re = U L / ν`. Viscous diffusion
length over the shot `δ = sqrt(ν T)`. A feature of wavelength λ decays as `exp(-ν (2π/λ)² T)`. A44's rule (1 %
over 1 s at the smallest cell, using the discrete per-step factor `1/(1 + 4N)`) is
`4 ν · 1 s / h² < 0.01`, which is the same as `δ(1 s) < 0.05 h`.

**Owner's answer (A44).** Advise viscosity 0 when it would change the flow by less than about 1 % over one second
at the smallest cell. Keep printing the per-liquid number.

**What the sources support.** It is established that a hybrid method's own dissipation acts like viscosity, and
that a physical viscosity below it is not visible (Bridson; XPIC; APIC). It is also established that PIC damps all
nonzero wavenumbers through a filter that can be computed (XPIC). Turning that filter into `ν_PIC ≈ h²/(4Δt)` is
reasoning. A44 judges at the two-cell mode, but the transfer removes that mode anyway (XPIC: the null space is
removed). It also uses a fixed second rather than the shot length. Under A44, water at 1 cm cells counts as
visible (4 %), which contradicts r = 0.008.

**Recommended decision (refines A44).** The planner computes, per viscous liquid, at the smallest cell h, the
largest step Δt, the liquid's own α and the run's simulated duration T:

1. `N = ν Δt / h²`, printed as now.
2. `r = 4 N / (1 - α)`, or "none" when α = 1.
3. `D = ν T / h²` and `δ = sqrt(ν T)` in cells.
4. `Re = U L / ν`, printed for information, with L the domain's smallest extent and `U = sqrt(2 g L)`.

Advise 0 when `r < kNegligibleViscosityRatio = 0.1` **or** when the viscous change of the smallest faithfully
resolved feature over the run, a wavelength of `kSmallestVisibleWavelengthInCells = 4` cells, is below
`kNegligibleViscousChange = 0.01`. The second test reads `1 - exp(-π² D / 4) < 0.01`, that is `D < 4.07e-3`.

- Why 0.1: the total damping then changes by less than 10 %. This is a judgment.
- Why 4 cells: PIC removes the 2-cell mode, and XPIC's Fig. 1 shows strong attenuation up to about 4 cells.
- Why T and not 1 s: the user sees the whole shot.

Printed line, for example: `water: viscosity 0.001 Pa s; at the smallest cell the particle transfer damps
velocity differences about 125 times more than this viscosity (ratio 0.008); over 10 s it spreads momentum over
0.32 cells; Re 4.4e5. It will not be visible: set its viscosity to 0. Note: at viscosity 0 the liquid slips along
solids instead of sticking (A53).` The last sentence is required. At viscosity 0 the run takes the pressure path,
so boundary stick (A26, A53) changes even when bulk viscosity is negligible.

"Physical viscosity below the method's own numerical dissipation" is an honest statement **as an order of
magnitude, for the bulk**, with two caveats. PIC damping also damps rigid rotation, which true viscosity does not
(APIC). It also has no boundary stress.

**What stays a judgment call.** The thresholds 0.1 and 1 %. The 4-cell wavelength. Using L and U from the domain
rather than asking the user. Grid-only regions (cell resize) have no transfer filter, so r does not apply there and
only D applies.

## 2. Budgeting GPU memory

| Source | What it says | Where | URL |
|---|---|---|---|
| CUDA Runtime API, `cudaMemGetInfo` | `*free` is "the amount of memory on the device that is free according to the OS". `*total` is "the total amount of memory available to the current context". "CUDA is not guaranteed to be able to allocate all of the memory that the OS reports as free". In a multi-tenant situation the free estimate "is prone to race condition" with other processes or threads. | 6.x Memory Management | https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html |
| CUDA C++ Best Practices Guide | "Memory allocated through the CUDA Runtime API, such as via cudaMalloc(), is guaranteed to be aligned to at least 256 bytes." | 10.2.1.2 | https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html |
| NVIDIA blog, stream-ordered allocator part 1 | The legacy calls synchronize the device. Without pooling, memory "must be allocated through an expensive system call". A pool whose request fails because of fragmentation is defragmented by remapping. | whole post | https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/ |
| NVIDIA support article 5490 | Since driver 536.40, CUDA applications "which previously crashed when running out of GPU memory" continue "albeit at lower speeds" by using shared system memory "when running close to maxing out GPU memory". The setting "CUDA - Sysmem Fallback Policy" can be set to "Prefer No Sysmem Fallback" (driver 546.01 and later). | whole article | https://nvidia.custhelp.com/app/answers/detail/a_id/5490 |
| NVIDIA forum (Robert Crovella) | The fallback is "a windows GPU driver feature". The question whether `cuMemGetInfo` changes under it went unanswered. | thread | https://forums.developer.nvidia.com/t/system-memory-fallback-and-estimation-of-gpu-memory/306185 |
| Microsoft, D3D12 residency | The per-process video memory budget "can fluctuate noticeably as background processes wake-up and sleep". Over budget, "the process will be intermittently frozen ... and/or the creation APIs will return failure". The kernel moves heaps to system memory "only as an extreme last resort. Applications should not rely on the over-budget behavior". | Residency budget; Heap resources | https://learn.microsoft.com/en-us/windows/win32/direct3d12/residency |
| Microsoft DirectX blog, GPUs in Task Manager | Dedicated memory is VRAM managed by VidMm. Shared memory is system memory: "the GPU is only allowed to use half of physical memory at any given instant", bounded by the commit limit. | Dedicated vs shared | https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/ |
| Blender manual, Cycles GPU rendering | "If the GPU memory is full Blender will automatically try to use system memory. This has a performance impact". | FAQ, out of memory | https://docs.blender.org/manual/en/latest/render/cycles/gpu_rendering.html |
| Redshift manual, Memory | "By default, Redshift reserves 90% of the GPU's free memory"; the rest is for other applications and the OS. Lower it to 80 or 70 when other GPU programs have trouble, or raise it to 100 % on a dedicated card. "Increasing the percentage beyond 90% is not typically recommended" (instability, driver crashes). | Memory | https://help.maxon.net/c4d/2025/en-us/Content/_REDSHIFT_/html/Memory.html |
| Redshift manual, out of VRAM | Redshift "cannot operate with less than 256MB of free VRAM", even with out-of-core rendering. Chrome, Photoshop and Nuke can use considerable VRAM. | whole page | https://help.maxon.net/r3d/houdini/en-us/Content/html/Dealing+with+Out-Of-VRAM+situations.html |
| Houdini environment variables | `HOUDINI_OCL_MEMORY_POOL_SIZE`: the "percentage of total device memory to allocate for the pool", default 0.125. `HOUDINI_OCL_COP_MEMORY`: a "fraction of total OpenCL device memory". | OCL entries | https://www.sidefx.com/docs/houdini/ref/env.html |
| PyTorch `set_per_process_memory_fraction` | The allowed memory is the total visible memory times the fraction, and the allocator raises out-of-memory beyond it. "In general, the total available free memory is less than the total capacity." | API page | https://docs.pytorch.org/docs/2.15/generated/torch.cuda.memory.set_per_process_memory_fraction.html |
| PyTorch CUDA notes | The caching allocator's unused memory "will still show as if used in nvidia-smi". `max_split_size_mb` "can reduce fragmentation". `expandable_segments` avoids "slivers of unusable memory". | Memory management | https://docs.pytorch.org/docs/2.15/notes/cuda.html |
| TensorFlow GPU guide | By default TensorFlow "maps nearly all of the GPU memory ... to more efficiently use ... by reducing memory fragmentation". `set_memory_growth` grows the allocation and never releases it. A hard `memory_limit` "is common practice for local development when the GPU is shared with ... a workstation GUI". | Limiting GPU memory growth | https://www.tensorflow.org/guide/gpu |

Not opened, or seen only in search snippets:
- RMM `pool_memory_resource` defaults: initial pool half of the free memory, maximum all of it (older releases; the
  doc pages returned 404).
- The 2 MiB granularity of small `cudaMalloc` calls: forum reports only
  (https://forums.developer.nvidia.com/t/how-big-is-minimum-assigned-size-in-cudamalloc/57170, not opened).
- Context overhead: from memory, several hundred MB depending on the card and the loaded modules.
- No EmberGen documentation was found.
- The Dawn D3D12 backend budgets 95 % of the OS budget (search snippet only).

**Pattern.** Renderers that share the desktop take a share of **free** memory: Redshift 90 % of free, and the old
FluidiTy code took the smaller of 70 % of total and 80 % of free. ML frameworks and Houdini take a share of
**total**, on the assumption that the card is theirs. Production tools allocate once and pool (TensorFlow,
Houdini's pool, RMM, the CUDA pools) to avoid fragmentation and the synchronizing cost of `cudaMalloc`. On Windows,
going over budget rarely fails: Cycles and the NVIDIA fallback page silently into shared system memory and slow
down. Microsoft tells applications not to rely on that.

**Owner's answers.** A45: the budget is a share of the memory free at the start, beginning at 90 %. A43: any
measured value above its planned budget prints a warning, with no margins, so the plan must give true upper
bounds.

**What the sources support.** Basing the budget on free memory, and the 90 % figure, match Redshift's documented
default and its stated reason. `cudaMemGetInfo` free is an OS figure: it can race with other processes and is not
guaranteed to be allocatable. On Windows, an allocation above the dedicated memory may succeed and run slowly.
Allocating once up front (TensorFlow, pools) is the standard way to avoid fragmentation and late failures.

**Recommended decision (confirms A45, refines A43):**

1. Create the CUDA context and call `cudaFree(0)` **before** reading free memory, so that the context and the
   loaded kernels are already outside "free".
   `allowed = min(kGpuFreeShare * free, free - kGpuMinHeadroomBytes)` with `kGpuFreeShare = 0.90` (Redshift) and
   `kGpuMinHeadroomBytes = 512 MiB`. The floor is reasoning: about twice Redshift's 256 MB minimum, so the desktop
   compositor and a browser can grow on a small card. Print total, free, allowed and the share.
2. Reserve everything at the start, as the design already does. Do not re-check free memory per allocation: after
   startup there are no allocations to check. Better: **one arena**. Make one `cudaMalloc` of the planned total and
   suballocate it at 256-byte alignment. Library work memory (cub, cuSPARSE) must also come from the arena. The
   footprint then equals the plan plus one rounding, fragmentation cannot happen, and only one synchronizing call is
   made. (Reasoning built on the pooling practice above.)
3. Measure two things, and warn on each in its own way:
   - (a) **Bytes requested** (sum of buffers, or the arena size) against the estimate. Equal by construction, so
     any overrun is a planner bug: warn, no margin. This is A43, meaningful as stated.
   - (b) **Device memory taken**: free before allocation minus free after. Print it as
     `device memory taken: X (requested Y)`. Warn when `X < Y - 2 MiB`. On Windows that means part of the arena did
     not land in dedicated memory (fallback). This test is reasoning, not documented NVIDIA behaviour, so a test on
     the Windows machine must confirm it. Do not warn on `X > Y`, because X also includes other processes.
4. Each frame, read free memory and report its smallest value. Microsoft documents that the budget fluctuates and
   that a process over budget is frozen intermittently.
5. On Windows, `check` and the add-on print once: `Windows can move GPU memory to shared system memory when the card
   is full; the run then slows down a lot instead of failing. To make it fail instead, set "CUDA - Sysmem Fallback
   Policy" to "Prefer No Sysmem Fallback" for blender.exe (and fluidity.exe) in the NVIDIA Control Panel.`

**What stays a judgment call.** 90 % against 85 %. The 512 MiB floor. Whether the add-on should offer a 100 %
"dedicated card" choice, as Redshift does. The 2 MiB tolerance in check (b) until the Windows test.

## 3. Rigid bodies with constrained motion inside an implicit solve

| Source | What it says | Where | URL |
|---|---|---|---|
| Batty, Bertails and Bridson 2007 | The solid update `V^{n+1} = V^n + Δt M_S^{-1} J p` uses V, "the generalized velocity of the solid (... a six-dimensional vector for a single rigid body, etc.)", and the boundary velocity `v = J* V`. Pressure is the Lagrange multiplier of a kinetic-energy minimisation, which gives a symmetric positive semidefinite system. Scripted solids: let `M_S^{-1}` drop to zero, keeping their term in the right-hand side. | section 2, Eqs. 2 and 3; section 3.2 | https://www.cs.ubc.ca/labs/imager/tr/2007/Batty_VariationalFluids/variationalFluids.pdf |
| Carlson, Mucha and Turk 2004 | Rigid fluid. The constraint forces `α_c, A_c` "could be used to model the forces necessary for joints"; listed as future work. | section 8 | https://www.cc.gatech.edu/~turk/my_papers/rigid_fluid.pdf |
| Robinson-Mosher et al. 2008 | Monolithic coupling. Water drives a fish with a PD-controlled articulated skeleton. The coupled system is symmetric indefinite, made positive definite by elimination. | section 5, Fig. 13 | https://www.cs.ucr.edu/~shinar/papers/2008_solid_fluid.pdf |
| Robinson-Mosher et al. 2011 | An SPD formulation of monolithic fluid-structure coupling, "at least to the extent that the uncoupled systems were symmetric positive definite". | abstract, sections 5.2 and 5.3 | https://www.cs.ucr.edu/~craigs/papers/2010-spd-sfi/paper.pdf |
| Aanjaneya 2018 | `M v^{n+1} = M v* - J^T W^T λ` with M "the generalized mass matrix" and J mapping solid degrees of freedom to fluid faces. Schur complement SPD. A plank "hinged at its center, but free to rotate" is shown; the hinge enters through generalized coordinates. Articulation is cited to [WTF06]. | sections 3 and 4, Fig. 3 | https://rutgers.box.com/shared/static/7v069ja8rxxgzq8s3ytiykm7xk7oxtev.pdf |
| Takahashi and Batty 2020, Monolith | Pressure, viscosity and contact in one SPD minimisation. Rigid constraints are posed at velocity level with a constraint Jacobian `J_λ`, with positions held fixed during the solve. Articulated coupling is cited (Lentine 2011, Tan 2011) as not handling contact. | sections 2, 3.2 and 4 | https://dspacemainprd01.lib.uwaterloo.ca/server/api/core/bitstreams/c251b0af-4999-4dfc-bbfe-9e8c1ece638f/content |

Not opened: Klingner et al. 2006; Takahashi and Batty 2021 (FrictionalMonolith); Lentine et al. 2011; Weinstein,
Teran and Fedkiw 2006.

**Pattern.** Every coupled solver writes the body in generalized velocities `q`, with `xi = J q`, and keeps the
body block `J^T M J` (plus `Δt J^T R^T E^T W E R J` with viscosity). This block is SPD when M is SPD and J has full
column rank. Prescribed components move to the right-hand side (the limit `M^{-1} → 0`). Joints are either reduced
coordinates (J) or velocity-level Lagrange multipliers (Monolith's `J_λ`, an indefinite system unless they are
eliminated).

**What "hold the forbidden components" is (reasoning).** It is reduced coordinates with a selection matrix:
`xi = S q + (I - S S^T) xi_before`. Replacing the rows and columns with the identity and folding the coefficients
into `rate_from_solids` is exactly Dirichlet elimination, which stays symmetric and SPD. It is exact when every
allowed motion is a coordinate of `xi` in the frame where `xi` is written: world axes, rotation about
`origin_at_start`. So it is exact for today's `FreeMotionLimits` (world-axis rails; all turning on or off about the
origin), provided forbidden components are zero before pressure (the commit zeroes them, solids design 5). It is
wrong for:
- a rail along a non-axis direction `d`, where `J = [d; 0]`;
- a hinge about axis `a` through a point `p ≠ origin`, where `J = [a × (origin - p); a]` is one column: selecting
  components would leave `V` and `Omega` uncoupled;
- a hinge whose axis is fixed in the body, where J changes with the pose every step;
- a body on a rail that itself moves, where `xi = J q + xi_prescribed`.

Over a step, a hinge about p drifts at first order because positions are frozen during the solve (Monolith does
the same). A position projection after commit is needed.

**Owner's answer (A46, first half).** A guided body holds its forbidden components at their value before the
solve, and only the free components are solved with the liquid.

**What the sources support.** This is the generalized-coordinate formulation of Batty 2007 and Aanjaneya 2018,
specialised to a selection matrix. It is symmetric and SPD, and exact for the constraints the settings can express
today.

**Recommended decision (confirms for now, with a cheap generalisation).**
- Keep the selection form for `FreeMotionLimits` as it stands.
- Store the guide internally as a 6 × m matrix `J` (m from 0 to 6) plus a prescribed `xi_0`. The selection becomes
  the case `J = S`, `xi_0 = 0`.
- Make the per-solid block `J^T (M + Δt R^T E_s^T W E_s R) J` (m × m Cholesky) and the per-sample coefficients
  `R J`.
- Cost: per solid, m columns instead of 6, which saves memory. One 6 × m product per solid per solve, which is
  negligible. A position projection after commit for pivot constraints.
- Then a non-axis rail or an off-origin hinge is a settings change, not a solver change. Test: a hinged plank
  equals the same plank whose 6 × 6 block is solved and then projected, in the limit of a small step.

**What stays a judgment call.** Whether to expose hinges and arbitrary rails to the user now, or only the
internal J. The drift correction (projection or Baumgarte).

## 4. Reporting the hydrodynamic force on solids

| Source | What it says | Where | URL |
|---|---|---|---|
| Uhlmann 2005 (arXiv 1809.08170) | The force on a particle comes from a momentum balance over the fluid. By Cauchy's principle, `∮ τ·n dσ = -∫ f dx + d/dt ∫_S u dx` (and the same for torque with `r ×`, r taken from the particle centre). Drag and lift are evaluated "as sums over the fluid-solid coupling terms", not by integrating the surface stress. Cylinder at Re 100, D/h = 38.4: Strouhal number within 4 %, mean drag overpredicted by 11 %, attributed to confinement. | section 4, appendix B, section 5.1 and Table 1 | https://arxiv.org/pdf/1809.08170 |
| Batty, Bertails and Bridson 2007 | The generalized force on the solid is `J p`, with J the adjoint of the operator that gives the boundary velocity, so the power exchanged balances exactly. For scripted bodies, the coupling "reduces to calculating the work done by pressure on their boundaries". | sections 2 and 3.2 | https://www.cs.ubc.ca/labs/imager/tr/2007/Batty_VariationalFluids/variationalFluids.pdf |
| Takahashi and Batty 2020, Monolith | Rigid bodies receive pressure, viscous stress and contact as forces inside one minimisation, so the same discrete terms act on fluid and body. | sections 3 and 4 | https://dspacemainprd01.lib.uwaterloo.ca/server/api/core/bitstreams/c251b0af-4999-4dfc-bbfe-9e8c1ece638f/content |

Not opened: Peskin 2002; Kempe and Fröhlich 2012 (no free PDF found; per its abstract, it integrates the
artificial flow inside the particle numerically); Ng, Min and Gibou 2009 (no free PDF). Analytic drag values,
from memory and standard:
- Stokes sphere: `F = 6π μ R U`, rotating sphere `T = 8π μ R³ Ω`.
- Plane Couette: shear force `μ A U / H`.
- Taylor–Couette torque per length: `4π μ Ω R1² R2² / (R2² - R1²)`.
- Stokes first problem: wall shear `μ U / sqrt(π ν t)`.
- Hydrostatic buoyancy: `ρ g V`.

**Pattern.** Accurate grid solvers read the force from the discrete coupling terms (momentum exchange), not by
re-integrating the stress on a reconstructed surface. Momentum exchange is exactly conservative by construction.
The CFD accuracy reached with this method is a few percent to about 10 % at 20 to 40 cells per diameter (Uhlmann).

**Owner's answer (A46, second half).** Add the viscous force and torque on every driven solid to the step report,
next to the pressure force.

**What the sources support.** Report what the discrete equations exchange. The free-solid row already is that
balance:
`M (xi_after - xi_before) = Δt Σ_e (force, torque)_per_pressure p_e - Δt R^T E_s^T W (E x + g)`.
For a driven solid, the same transposed coefficients give the force it would receive (Batty's `J p`, with
`M^{-1} = 0`).

**Recommended decision (confirms and makes it precise):**
- Per solid (free, guided and driven), report two 6-vectors, `pressure_push` and `viscous_push`:
  `pressure_push = Δt Σ_e (force_per_pressure, torque_per_pressure)_e p_e` and
  `viscous_push = -Δt R^T E_s^T W (E x + g)`. Both are impulses (N s, N m s) and are reported as mean force and
  torque over the step (divide by Δt).
- Driven solids need their R coefficients to be kept in the sample data rather than folded into `rate_from_solids`
  only. Store the 6 coefficients alongside, or evaluate them in a separate reduction pass.
- Frame: world axes. Torque about the solid's `origin_at_start`, the point `Omega` refers to. The report states
  the point, and `T_p = T_o + (o - p) × F` lets a reader move it.
- Guided solids: report the full 6 components, including the forbidden ones (the reaction the guide carries).
- Invariant tested: for a free unguided solid, `pressure_push + viscous_push == M (xi_after - xi_before)` within the
  solve tolerance. Liquid momentum change plus solid change equals the external impulse (extends
  `BoxInLiquidBlock_MomentumChangedOnlyAtAirEnds`).
- Analytic tests, in order of value:
  1. Couette plate force `μ A U / H` (a linear profile is exact on the grid, so the tolerance is near the solver
     tolerance).
  2. Hydrostatic buoyancy on a submerged box (pressure part only, a few percent).
  3. Taylor–Couette torque on an inner cylinder (tests the torque and the off-origin arm; 5 to 10 % at about
     10 cells across the gap).
  4. Stokes settling sphere in a large box, with stated wall-effect correction and tolerance, or only linearity in U
     as the design already plans.

**What stays a judgment call.** Report impulse or mean force (mean force recommended, with the step's Δt in the
report). Whether to also report torque about the centre of mass for driven solids whose origin is elsewhere. The
tolerances of tests 3 and 4.

## Sources not opened in this pass

Goldade et al. 2019; Klingner et al. 2006; Takahashi and Batty 2021 (FrictionalMonolith); Lentine et al. 2011;
Weinstein, Teran and Fedkiw 2006; Peskin 2002; Kempe and Fröhlich 2012; Ng, Min and Gibou 2009; Bridson's book,
2nd edition; Brackbill and Ruppel 1986 and Brackbill, Kothe and Ruppel 1988 (FLIP); RMM pool documentation (404);
Houdini Pyro and Vellum GPU memory pages; EmberGen documentation (none found); NVIDIA statements on context size
and `cudaMalloc` granularity (forum reports only).
