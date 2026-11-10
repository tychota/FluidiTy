# M2 fixed-cache evidence

## Pinned input and reproduction

The input is the 216-frame public CUDA `oscillating-drop-l2` run, including frame zero, from M1 revision
`3e0526edb9adda66b4be965ab469d74f5042c990`. The native reader verified the complete frame range against the
archived export, including resolved fingerprint, execution ID, clock and referenced points. The two benchmark
processes left all 897 input-cache files byte-identical. A copied cache with the frame-zero manifest missing
and the same copy with one flipped point-blob byte both failed at frame zero without publishing a report.
See the [M1 closure review](m1-transport-evidence.md#m1-closure-review)
for the source simulation evidence.

The exact Windows artifacts are required for this receipt. Supply the archived run directory with
`cache/`, `scene.flrs`, `scene.flrs.flst` and `frames.jsonl`. Put the exact
`fluidity_liquid_acceptance_fixture.exe` and `fluidity_simulate.exe` together. The script contains their
SHA-256 pins and the execution ID. The original build and run directories were
`out/build/m1-piston-projected-cuda` and `out/m1/naming-closure/rtx-drop-216`. Raw caches, binaries and reports
are local artifacts.

```powershell
uv run --project reference --locked python -m pytest tools/native/tests/test_m2_surface_baseline.py tools/native/tests/test_m2_motion_interval.py -q
uv run --project reference --locked python tools/native/m2_surface_baseline.py --reader $reader --run $run --output $report
uv run --project reference --locked python tools/native/m2_motion_interval.py --output $motionReport
```

Run the surface command twice with fresh output paths. Compare the `samples` entries excluding
`reconstruction_seconds` and `metrics_seconds`. Those entries include the input points digest and an indexed
geometry hash. The measured processes used Windows 11, Ryzen 7 9700X, Python 3.13.18, NumPy 2.2.6 and SciPy 1.15.3.

## Geometric reference

The analytic seed is a determinant-one stretch of a 0.125 m sphere. Its extraction control isolates lattice
discretization. Particle reconstructions use the same tetrahedral decomposition on a world-aligned lattice.
The isotropic and determinant-one covariance unions use 1.35 times the equivalent material radius as support.
The density field splats material volume into eight surrounding grid nodes and extracts the 0.5 isosurface
of a Gaussian-smoothed volume fraction (0.025 m sigma, four-sigma cutoff). The field uses full-volume particles.

The table shows vertex-sampled radial discrepancy to the analytic ellipsoid at frame zero. This is a one-way
radial metric. Volume error is signed relative to the committed sum of material volumes, with the table giving
the range across frames 0, 48, 96, 144, 192 and 216. The analytic rows use frame zero only.

| Field | Spacing (mm) | Radial RMS / max (mm) | Volume error | Components | Reconstruction median / p95 / max (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Analytic | 12.5 | 0.128 / 0.463 | −0.50% | 1 | 0.156 / 0.156 / 0.156 |
| Analytic | 6.25 | 0.033 / 0.119 | −0.12% | 1 | 0.608 / 0.608 / 0.608 |
| Isotropic union | 12.5 | 28.444 / 125.538 | +20.18% to +21.32% | 134–414 | 0.428 / 0.482 / 0.482 |
| Isotropic union | 6.25 | 17.626 / 126.500 | +23.34% to +23.78% | 136–420 | 1.321 / 1.545 / 1.545 |
| Covariance union | 12.5 | 28.442 / 125.351 | +20.17% to +21.45% | 134–414 | 0.442 / 0.517 / 0.517 |
| Covariance union | 6.25 | 17.694 / 126.374 | +23.33% to +23.73% | 136–432 | 1.399 / 1.494 / 1.494 |
| Density | 12.5 | 4.988 / 6.297 | −11.60% to −11.12% | 1 | 0.181 / 0.201 / 0.201 |
| Density | 6.25 | 4.733 / 5.989 | −10.85% to −10.67% | 1 | 0.644 / 0.713 / 0.713 |

All sampled meshes had zero boundary edges, nonmanifold edges, orientation defects and zero-area triangles.
Analytic and density rows had Euler characteristic 2. The union controls had Euler characteristics from 264
to 860 and failed to represent a single drop. Both benchmark runs agreed exactly on all 38 geometry hashes
and numerical sample records. The first process took 30.33 s, including 0.97 s of input verification, and
peaked at 126.1 MiB RSS. The second took 30.03 s and peaked at 128.0 MiB. These are whole CPU-process working sets.

The density field is a reproducible stateless comparison for this drop. Its raw volume error exceeds the
production per-component target of 0.5%. The fixed experiment establishes neither production volume correction
nor thin-sheet, wall-contact, cavity, global intersection or persistent-ID qualification.

## Separate MotionInterval study

The manufactured carrier has an exactly integrable time factor. Constant translation checks the integrator,
time-varying translation checks sampling, and differential rotation winds a material sheet while keeping each
point's radius fixed. The study uses a 1/24 s interval and a predeclared provisional budget of 0.1 cell
(2.5 mm at 25 mm cell width). Analytic transport under the interpolated carrier separates sampling error
from the 256-step RK2 error. Peak RK2 displacement was 0.184 cell.

| Linear sampling intervals | Translation max (mm) | Folding sheet max (mm) | Folding RK2-only max (mm) | Velocity bytes per node |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 84.883 | 288.850 | 0.027 | 24 |
| 2 | 18.216 | 67.280 | 0.082 | 36 |
| 4 | 4.409 | 16.267 | 0.097 | 60 |
| 8 | 1.094 | 3.959 | 0.101 | 108 |
| 16 | 0.273 | 0.911 | 0.102 | 204 |

The folding control missed the budget at eight intervals, which triggered the sixteen-interval measurement.
Right-endpoint hold had nonmonotone errors and missed the folding budget at both eight and sixteen intervals.
Constant translation agreed within 4e-15 m. Storage estimates include three FP32 velocity components at
each sample and a shared array of eight-byte timestamps. These are manufactured-carrier results. Choosing a
production sampling policy still requires spatial validity, storage, replay and breaking-wave measurements.

## Integrated and research gates

| Gate | Status |
| --- | --- |
| M1-B entry | In progress on its separate branch. The available native Points playback fixture receipt covers reader behavior on the pinned Blender build. Installed real bake, resume and fresh render evidence belong to the separate M1-B qualification. |
| M1c before integrated M2 | Partial. M1 closure evidence covers solver and cache contracts. A focused existing-build CTest run passed 68 cache, schema, inspection, CPU replay and lifecycle cases. The checked point reader authenticates published manifests and blobs. The point publication lacks interior/cavity occupancy, timestamped grid-carrier samples and accepted Whitewater events. Repeat the architecture audit against the merged and qualified M1-B workflow. |
| Zhao CNN reproduction and promotion | Open. The published configuration and archive are identified. Code and model-weight terms remain unresolved, the research runtime is absent from the reference environment, and there are no full, band or tiled inference receipts. Keep the geometric dependency set. [Neural evidence](neural-surface-evidence.md) |
| Persistent Mesh and replay | Pending M1-B and post-M1-B M1c. |
| Whitewater, Volumes and Ocean Boundary | Pending integrated work. |
| Native breaking-wave bake | Pending. The source cache is an isolated drop. |
| Blender inspection and Foam Projection | Pending. The existing native Points fixture covers M0 reader behavior. |
