# Matched-error manufactured pressure benchmark

The uniform-finest and balanced two-level layouts tile the same sealed cube
`[-1, 1]^3` m. The finest width is 0.25 m. The two-level layout uses 0.5 m
cells for `x < 0.5 m` and finest cells in the rightmost quarter. Both use the
same incompressible, zero-source setup with a 0.5 s step and density 2 kg/m³.
Exterior wall flux is zero.

The analytic pressure is `p(x) = 0.125 + 0.75 x` Pa with `x` in meters.
Interior x-faces start at a predicted 0.1875 m/s in the positive direction.
Interior y- and z-faces start at zero. The coarse–fine interface is normal to x,
so the analytic pressure difference divided by normal center distance equals
the same gradient on both layouts, even where subface centers are offset in y
and z. The analytic correction gives zero velocity on every face.

Both layouts use the checked explicit pressure graph and bounded CPU solver.
The common gates are a maximum pressure error of **1e-4 Pa** after subtracting
the analytic pressure at each layout's zero gauge and a maximum physical
divergence error of **2e-5 s⁻¹**. Divergence is recomputed from the corrected
velocities and topology areas, then divided by each cell's physical volume.
Every attempt starts from zero pressure, including the warm-up. The executable
returns a failure if either layout exceeds the common gates.

## Debug CPU measurement

Measured with MSVC 19.44.35229 on an AMD Ryzen 7 9700X under Windows. Each
layout warmed up once. Nine unprofiled samples per layout alternate execution
order. The timer covers only `MacPressureWorkspace3D::ProjectPrepared`,
including hierarchy construction, the solve, and projection publication.
Topology and graph construction, workspace reservation and independent error
measurement are outside the timer. The workspace reservation counts both
projection banks and bounded scratch. High-water payload records actual use
under the workspace's accounting.

| Metric | Uniform finest | Balanced two-level |
| --- | ---: | ---: |
| Pressure cells | 512 | 176 |
| Coarse cells | 0 | 48 |
| Owned interior faces | 1,344 | 456 |
| Mixed-level faces | 0 | 64 |
| Maximum pressure error (Pa) | 1.19209e-6 | 1.07288e-6 |
| Maximum corrected divergence (s⁻¹) | 5.96046e-6 | 8.90096e-6 |
| Maximum corrected face speed (m/s) | 5.96046e-7 | 7.15256e-7 |
| Solver iterations in each sample | 11 | 8 |
| Reserved workspace (bytes) | 4,300,232 | 1,470,344 |
| Used workspace high-water (bytes) | 277,808 | 107,672 |
| Median projection elapsed (ms) | 10.6120 | 3.0573 |
| Projection sample range (ms) | 9.7904–26.0196 | 2.8689–3.7612 |

This measurement is for manufactured pressure in a Debug build on a shared
host. It provides no full-wave or production-throughput result. Elapsed time
has no acceptance threshold.

## Release CPU measurement

A locked Release build with the same MSVC, host, layouts and error limits
reproduced every numerical and workspace entry above. The median projection
was 0.5470 ms uniform and 0.1291 ms adaptive. The nine-sample ranges were
0.3961–0.5936 ms and 0.1114–0.1902 ms, respectively. The timer has the same
projection-only scope.

The measured source was `bb9938c036ce246817b4c1f8a149c4e30a0f2a17` and the executable SHA-256 was
`c632e1855ecd6e772d1e58d27a10d636c18ac2b069420952e028eec74661ab99`.

## Reproduce

Configure the Debug CPU preset with the
[Windows native build procedure](../native/README.md#cpu-only-build-on-windows).
From the repository root, build and run the benchmark to emit the individual
elapsed samples in JSON. The CTest label runs the same error gates. For the
Release measurement, use the `windows-release-cpu` preset and output directory.

```powershell
uv run --offline --project reference --locked --group native cmake --build --preset windows-debug-cpu --target fluidity_m3_matched_pressure
& ./out/build/windows-debug-cpu/fluidity_m3_matched_pressure.exe
uv run --offline --project reference --locked --group native ctest --test-dir out/build/windows-debug-cpu -L '^pressure-benchmark$' --output-on-failure
```

The recorded executable was built on base
`ec9ded852b6324afa98ef9e4b622cc6944defa8b`. Its SHA-256 was
`a3d51de9a6d6ce09a35d3e7491dfc5a28a94eae0416fdf43ca594a4eb49f470e`.
