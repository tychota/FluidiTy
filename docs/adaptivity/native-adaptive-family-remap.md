# M3 adaptive family remap

The host adapter proposes one refinement or coarsening on the geometric
adaptive MAC topology. It uses the existing family edit for the candidate
topology, retained-leaf map and pressure warm starts. It copies retained cell
inventories and uses the conservative liquid remap for the edited family.
Topology family ordinals use signed `(z,y,x)` order with x changing fastest.
The inventory kernel uses z-fast `(x,y,z)` order. Both directions explicitly
apply the permutation `[0,4,2,6,1,5,3,7]`.

Integrated face flux is signed along the positive world normal. Integer
dyadic face rectangles identify unchanged faces and four-subface partitions.
Refinement uses exact equal-area quartering. Coarsening uses the native
deterministic signed sum and checks its residual with binary64 partials so an
unrepresentable coarse scalar cannot discard a fine flux contribution.
Incomplete or overlapping face coverage fails. Existing fine faces against
retained fine neighbors keep their individual flux values.

The optional exterior inputs supply aperture areas for both topologies and
committed integrated flux in canonical exterior-face order. Exterior flux is
positive outward from its owner on either side of an axis. A face must keep
its owner side and its aperture area, either unchanged or summed across its
four subfaces. Sealed apertures carry zero flux. Exterior refinement splits
flux in proportion to aperture and preserves its exact sum. It keeps the
directly computed binary64 shares if their compensated sum in canonical order
equals the coarse flux and the exact-sum check confirms it. Otherwise, the
split rounds each computed cumulative share to the nearest coarse-flux
quantum, with ties toward the larger magnitude, then takes consecutive
differences. Each resulting share can differ from the difference of the
unrounded computed cumulative shares by at most one coarse-flux quantum.
Aperture ratios and cumulative shares also incur ordinary binary64 rounding.
The quantum is the spacing from the coarse flux magnitude to the next larger
binary64 value in an unbounded exponent range, floored at the smallest
positive subnormal. Coarsening uses the same exact-sum check as interior
faces. Omitting the exterior inputs selects the sealed mode.

Refinement gives each new sibling one eighth of the old parent's net outflow,
including the outward flux on open exterior subfaces.
That is the same volume-scaled divergence for representable equal-volume
children. The deterministic tree joins each non-root child to the child
obtained by clearing its lowest set octant bit. Processing in descending child
order satisfies each subtree balance. The five sibling faces outside the tree
carry zero flux. A second incidence pass checks the resulting child outflows
against the target using 64 binary64 machine epsilons times the larger of the
target magnitude and largest incident flux. Nonzero subnormal targets must
match exactly. Coarsening discards sibling interior faces, whose opposite
incidences cancel in the family balance.

The fixtures use asymmetric x/z inventory slopes, a signed-coordinate family,
coarse and fine retained neighbors, and mixed outer flux signs. The old
parent's net outflow of 8 m³/s becomes 1 m³/s on each child. Retained leaf
inventories, pressures and net outflows match exactly. A refine/coarsen round
trip restores the original fields. An independent coarsening case combines
`[2, -3, 1, -8]` into −8 m³/s and checks the mean of unequal sibling warm starts.
A lossy sum, an unrepresentable subnormal inventory or flux split, and missing
committed flux coverage fail without changing committed inputs. Repeating the
same refinement gives identical candidate order, warm starts and face fluxes.
The exterior fixture covers both owner orientations, unequal apertures, sealed
walls, an unchanged open retained face, and refusal of incomplete aperture
coverage.

```powershell
& ./out/build/windows-debug-cpu/fluidity_native_tests.exe '--gtest_filter=AdaptiveLiquidRemap3D.*:AdaptiveMacTopology3D.*:LiquidRemap3D.*'
ctest --test-dir out/build/windows-debug-cpu --output-on-failure
```

All 14 focused tests and all 575 CTest cases passed on the qualified Windows Debug CPU configuration
with MSVC 19.44.35229, the cached Debug Conan toolchain and CPU qualification
receipt. See the [native build procedure](../native/README.md#cpu-only-build-on-windows).

The tree is a bounded conservative reference. The adapter handles one
geometric family and assumes the caller provides a same-width reconstruction
stencil. Physical masking, cut faces, persistent accepted-step transactions,
hysteresis, and CUDA execution remain integration work. See the
[M3 plan](../superpowers/plans/2026-11-11-m3-adaptivity-and-band.md#task-2--native-parity-masking-and-conservative-remap).
