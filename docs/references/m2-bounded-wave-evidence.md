# M2 bounded swell qualification

This control uses a 5 m long, 1 m wide strip over a stationary 0.15 slope, with 0.25 m cells, a 0.75 m inlet depth, a 0.25 m swell and a 2 s period. It exercises the particle solver and its accepted-source effects at one quarter of the reference scene length and depth. The strip has a single halfspace obstacle, so fluid can leave the inlet and sides. It does not qualify wave breaking or outlet reflection.

## Simulate, Whitewater and Volumes

CPU and RTX 5060 Ti completed frame 48 at 24 fps, with 240 accepted steps, zero timestep retries and zero capacity replays. Both started with 132 particles and finished with 430. The frame-48 point exports shared all 430 IDs. Position RMS and maximum differences were 0.0109 and 0.226 mm. Velocity RMS and maximum differences were 0.00884 and 0.153 mm/s. The runs overlapped on the same host. Measured Simulate wall time was 115.90 s on CPU and 87.98 s on CUDA. CUDA peak device reservation was 113,546,240 bytes.

Whitewater initially stopped with ABORTED while the source was still publishing. Resuming after Simulate settled restored the first 15 frames and completed through frame 48. Frame 48 contained 628 foam points and 92 bubbles. Whitewater took 0.306 s for that frame and peaked at 33.15 MB process RSS.

The small beach-control volume box rejected positive deposition support at frame 18. The larger wave box extends upstream to -4 m and includes the measured minimum foam or bubble x of -3.286 m. All 49 volume samples published. The last frame held 628 foam and 92 bubble equivalents and 0.0481711 m³ of gas in 710 nonzero voxels, with 59,904 known voxels. Its frame time was 0.023 s and peak RSS was 7.35 MB. The upstream excursion is a limitation of the open strip and the current inlet treatment.

A fresh Blender 5.2.3 process reopened a saved native scene with relative Alembic and OpenVDB files. At frames 0, 24 and 48 it read 132, 261 and 430 liquid points, and 0, 137 and 628 foam points. It also resolved all four OpenVDB grids at those frames. This playback has no reconstructed surface.

## Mesh and remaining acceptance work

The regular initial particle lattice leaves a gap across shallow columns. Mesh rejects the disconnected material at frame 0 before smoothing. A second seed samples the actual wet depth in each column. It passes the material connectivity check, then fails the 0.5% volume gate within the existing 0.25-cell vertex correction bound at surface spacing 0.125 m and sigma 0.16 m. At those settings the connected raw scaffold encloses 0.992169 m³ against 1.76393 m³ of material, a 43.8% deficit. Smaller sigma values of 0.10 and 0.125 m fail topology, while 0.20 m still fails volume correction. Neither failure publishes an invalid Mesh frame. They require component-aware reconstruction and a feasible thin-water scaffold, respectively. The full reference shot still exceeds the present uniform solver capacity.

The separate three-frame stationary beach surface and foam control reopens in Blender 5.3.0 build 02f0899d78e3. Surface geometry hashes and foam counts matched the 5.2.3 receipt. Mask float hashes differed at the two populated frames while the 5.3 mask agreed with its nearest-point oracle within 6.81e-8. The OpenVDB playback receipt matched. Wave-scale surface tracking, full reference bake and artist workflow qualification remain open.

## Reconstruction direction

Hu et al.'s [MLS-MPM](https://yuanming.taichi.graphics/publication/2018-mlsmpm/) improves material-to-grid transfer and discretizes the stress divergence. The accepted particle positions and material volumes still need a separate rendering reconstruction. Yu and Turk's [anisotropic surface kernels](https://faculty.cc.gatech.edu/~turk/my_papers/sph_surfaces.pdf) show why a spherical kernel struggles with flat features. Bhattacharya et al.'s [constrained level-set skinning](https://sealab.cs.utah.edu/Papers/Bhattacharya-2011-ALM/) fits a smooth surface between particle-derived bounds. Ando et al.'s [sheet preservation](https://ryichando.graphics/sheetflip/tvcg.html) also changes the FLIP sampling during simulation. For M2 the surface stage can use those geometric ideas on a frozen cache. Changes to simulation sampling and the narrow band belong to M3.

Frame zero now partitions material before density smoothing. Each of at most 16 components has a separate density lattice and volume target. Reconstruction divides the node budget across components. Publication measures the aggregate for intersections and rejects potential nesting. Replay checks the payload and restore authenticates each component against the original particle IDs and material volumes. Tracking later multi-component frames and repairing splits or merges remain open.

The separated-component control fits two unit-volume components and rejects overlapping scaffolds. On the original frozen swell, partitioning passes the connectivity guard and reaches reconstruction of a component with 1.875 m³ of material. Its raw surface encloses 1.30507 m³. The next step is a thin-water scaffold with local support fitted to the wet layer and beach, compared with the isotropic reference on the same frozen inputs. The existing geometric correction and topology gates still apply.

## Stationary beach density experiment

The Mesh field reflects each eligible particle's density contribution across the beach plane before filtering and clips the resulting surface against the actual plane. This borrows the boundary-extension idea from [mirrored kernel support in SPH](https://doi.org/10.1007/s40571-021-00403-3). FluidiTy uses it only for the rendering field of a sphere-union scene with a stationary planar obstacle. Reflected contributions do not enter the material target. Surface budgets, intersection checks and the existing vertex correction still apply.

On the frozen partial-column seed at 0.125 m spacing and 0.16 m sigma, the raw enclosed volume rose from 0.992169 to 1.341566 m³. The corrected surface encloses 1.759954 m³ of 1.763927 m³ of material, a relative error of -0.2253%. Frame zero published with 2,878 vertices and 5,752 triangles. The measured Debug CPU frame took 19.50 s with 16.15 MB peak process RSS. Read-only replay reproduced the exact frame-zero payload. Resume rejected frame one with 30 triangle intersections and no boundary, manifold, winding or degeneracy defects. The regular seed still fails at its smaller component, with 0.088769 m³ raw against 0.1875 m³ material.

## Frame-one projection repair

A diagnostic build measured the same triangle connectivity at three points in
the frozen frame-one bake. The seed and RK2-advected meshes each had zero
intersecting pairs. Nearest-surface projection introduced 30. The
[affected patch](m2-wave-frame-one-crossing.png) shows a measured pair near
world position (4.580, 0.732, 0.900) m, using the same axes and scale for
both stages. The plotted triangles are IDs 5522 and 5583 in the frame mesh
array. Their vertices moved only millimetres during projection.

Bounded tangential redistribution and reprojection reduced the crossings
but could not remove them all. Local edge flips then removed the remaining
crossings. The repair accepts only candidates with fewer crossings and no
other measured topology defects. It keeps vertex IDs, checks the original
projection distance and residual bounds for moved vertices, and remeasures
the entire mesh. The volume correction uses geometry-derived normals when
triangle connectivity changes.

The optimized CPU bake published frame one with all 2,878 vertex IDs,
5,752 triangles, and -0.0141% volume error. Ten triangle slots changed
connectivity, spanning 0.00361 m² after repair. Read-only replay reproduced the exact frame-one payload. Frame two then failed
scaffold volume correction before tracking. Its raw scaffold encloses
1.32813 m³ against a 1.76750 m³ material target. The full Blender
breaking-wave acceptance gate remains open.

## Reproduce the fixed inputs

Use the native fixture from this branch and pass the exact solver executable to bind its digest into the scene. Create a new output directory for each run.

```powershell
fluidity_simulate_fixture scene wave.flrs fluidity_simulate.exe cpu bounded-wave
fluidity_simulate run --stage simulate --scene wave.flrs --initial-state wave.flrs.flst --cache simulate --backend cpu --frame-end 48 --measure-job
fluidity_simulate run --stage whitewater --cache simulate --output-cache whitewater --frame-end 48
fluidity_simulate_fixture wave-volume-settings wave.flvc
fluidity_simulate run --stage volumes --cache whitewater --output-cache volumes --volume-settings wave.flvc --frame-end 48
fluidity_simulate run --stage mesh --cache simulate --output-cache mesh --frame-end 0 --surface-spacing .125 --surface-sigma .16
```

For the second seed, replace `bounded-wave` with `bounded-wave-column-seed` and use a different scene and cache directory. The measured timings describe this bounded experiment. The [fixed-cache reconstruction and Blender beach receipts](m2-integrated-progress.md) cover different inputs and acceptance controls.
