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

## Accepted adaptive leaves

`ApplyAdaptiveSurfaceOccupancy3D` maps decoded adaptive leaves onto the extraction lattice using half-open bounds in represented world coordinates. Supported particle nodes retain their samples. An unsupported node requires a full liquid leaf or an explicit empty air leaf. Missing, solid and partial leaves leave coverage unknown. The extraction gate requires eight supported samples in every crossing cube.

The manufactured regression covers a 2:1 corner, signed keys on a translated lattice, false inner shell removal, and true cavity preservation. The mapped samples equal the independently authored controls. An unsupported node in a solid leaf is refused.

The manufactured extraction round-trips the completed endpoint through the
M2 motion capture codec with explicit Eulerian ownership. EffectsSource already
joins each motion payload to the committed point frame and verifies full state
continuity across adjacent motion frames. The completed motion endpoint supplies
the occupancy records at the requested epoch. Mesh still needs to derive
particle support on the extraction lattice and qualify obstacle closure and
component-volume targets from that same accepted interval.

```powershell
uv run --offline --project reference --locked --group native ctest --test-dir out/build/occupancy-cpu --output-on-failure -R '^SurfaceOccupancy3D' --no-tests=error
```

## Graded motion carrier in Mesh

The existing Mesh motion interface now samples the canonical accepted graded
subfaces. It uses valid or predicted same-side carriers and the graded affine
fit. Rank loss selects PIC, and missing compatible support fails. At a
stationary planar tank's open top the carrier continues the interior fit at
most half a finest-cell width. This covers the small outward displacement from
Mesh volume correction. The manufactured affine control passed at the 2:1
corner and across that boundary within 1e-11 m/s.

The public graded Simulate cache with five leaves and 25 subfaces published
Mesh frames 0 through 2. Frame 2 retained all 458 vertex IDs and 912 triangles.
The relative volume error was -0.00010954. A separate replay process regenerated
payload BLAKE3
`6340f73c6a36857f7876a64616ddd3979d87506840a10848e3b7b45a3b5dc049`.

Run against the public graded cache from
[the Simulate qualification](graded-simulate-runtime.md#public-cli-motion-capture),
with a fresh output directory.

```powershell
ctest --test-dir out/build/motion-cpu --output-on-failure -R '^MotionCarrier3D' --no-tests=error
& ./out/build/motion-cpu/fluidity_simulate.exe run --stage mesh --cache $gradedCache --output-cache $meshCache --surface-spacing 0.125 --surface-sigma 0.25 --frame-end 2
& ./out/build/motion-cpu/fluidity_simulate.exe replay --stage mesh --cache $gradedCache --output-cache $meshCache --frame 2 --substep 0
```

This moving-correspondence control uses the stationary planar tank's signed
surface and target volume. The adaptive occupancy mapper still needs to be
wired into particle-band extraction on moving, nonplanar surfaces.

## Accepted planar band retirement in Mesh

The native planar tank path validates each motion sample with an Eulerian
owner. Its canonical leaves must be disjoint and entirely inside the resolved
tank, each represented liquid volume must fill its leaf, and the inventory
sum must cover the planar tank. The accepted endpoint sum sets the Mesh
component volume target. A partial leaf fails closed before reconstruction.

The native control retires a physically deep leaf in Simulate, completes
the frame, and decodes that motion payload for Mesh. The 1.5 × 4 × 1.5 m tank
retains one outward component and all seed vertex IDs with the 0.5% volume
tolerance. Halving the accepted core volume in the decoded input is refused.
The separate extraction fixture checks the inward boundary of a genuine
cavity. Moving nonplanar extraction still requires complete authenticated
air and liquid coverage and a support witness on the Mesh lattice.

```powershell
ctest --test-dir out/build/band-mesh-cpu -R 'UsesAcceptedEulerianCoreInventoryForPlanarMesh|^SurfaceOccupancy3D' --output-on-failure
```
