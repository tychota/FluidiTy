# M3 native occupancy and mesh qualification

`ApplySurfaceOccupancy3D` combines a particle density field with authoritative
occupancy and particle-support classifications at the same world nodes. The
caller maps the accepted simulation product to the extraction lattice. The
adapter checks exact origin, metre spacing and extents, and requires a complete
classification span in `(x * ny + y) * nz + z` order. A cell-centred mask or a
classification sampled at a different time requires mapping before this call.

Supported interface samples retain the input dimensionless density residual.
Certified deep liquid and certified air use fixed values of −0.5 and +0.5.
Every mixed-sign cube under the extractor's zero tie must have all eight
particle samples supported. This keeps the artificial constants off the
interpolated zero set. The extractor can still use those constants in gradient
samples on adjacent nodes. A positive boundary shell is required. Missing
occupancy, unknown classifications, refused particle support and insufficient
crossing coverage fail before a field is returned.

The native fixtures sample two analytic fields on a translated 15 × 15 × 15
world lattice at 0.25 m spacing and call the existing `ExtractSurfaceMesh3D`
and `MeasureSurfaceMesh3D`. The raw band shell has two closed surfaces and
Euler characteristic 4. Certifying its core as liquid leaves a single closed
surface with Euler characteristic 2 and identical exterior vertex positions.
The true cavity case has two closed surfaces and Euler characteristic 4. Its
vertex normals and triangle winding point toward the cavity air and outward
at the exterior. The mesh metrics report zero boundary edges, nonmanifold
edges or vertices, orientation defects, degeneracies and self-intersections
in the authored cases.

```powershell
& ./out/build/windows-debug-cpu/fluidity_native_tests.exe '--gtest_filter=SurfaceOccupancy3D.*'
ctest --test-dir out/build/windows-debug-cpu --output-on-failure
```

All four focused tests and all 563 CTest cases passed on the qualified Windows Debug CPU configuration
with MSVC 19.44.35229. The build used the cached Debug Conan toolchain and CPU
qualification receipt. The [native build procedure](../native/README.md#cpu-only-build-on-windows)
describes setup for a clean checkout.

This qualification uses manufactured node coverage and static geometry. M2
integration still requires authenticated occupancy and particle support from
the accepted interval, obstacle closure, component volume targets and lineage,
accepted motion and valid carrier coverage, and mesh persistence and cache
replay. See the [M3 plan](../superpowers/plans/2026-11-11-m3-adaptivity-and-band.md#task-4--particle-band-and-eulerian-interior).
