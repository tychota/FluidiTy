# M3 manufactured particle band and Eulerian core

The independent Python fixtures use a uniform 18 × 4 × 24 grid with unit-width cells,
one-second motion intervals and an incompressible material. Analytic tank and cavity
geometry supply occupancy fractions and liquid-side depth. The accepted component
target starts from the analytic volume and advances by signed inward flux, outward
flux and source. The fixture measures geometric occupancy drift and actual owned
volume against that target independently. The descending control changes the source
by +0.5 m³ and measures −0.5 m³ geometric drift with identical occupancy.

Every positive-volume cell has one inventory owner. Full cells within 6h of a free
surface own eight symmetric particle quadrature points. Deeper and partially filled
cells own Eulerian inventory. The 2h overlap blends the manufactured carrier
velocities on wet-adjacent MAC faces before saving old-time velocity history. A
deliberately unavailable particle deposit uses the valid Eulerian sample and marks
the face for PIC reset. Reseeding uses the incompressible EOS ratio J = 1 and the
destination affine velocity and gradient. Checks sum actual particle or cell mass,
current and reference volumes, momentum, constituent volumes and enthalpy.

The mixed MAC support box has half-widths 2h, 1.5h and 1.5h. Its enclosing radius
is √8.5h. The seed offsets have radius √3h/4 and the moving interfaces displace
at most 1h. Subtracting support, displacement and seed radius from 6h leaves at
least 1.652h. The 2h overlap retains at least 0.567h after subtracting displacement
and seed radius. These are geometric margins for the authored motions and seed
placement. Carrier availability in the overlap uses the fixture's prescribed
face rule.

| Case | Signed inward / outward / source volume (m³) | Eulerian cells | Particle cells | Reseeded / retired cells | Exterior / cavity wet–dry faces |
| --- | ---: | ---: | ---: | ---: | ---: |
| Deep tank | 0 / 0 / 0 | 936 | 432 | 0 / 0 | 72 / 0 |
| Translating | +73 / −73 / 0 | 404 | 888 | 120 / 120 | 144 / 0 |
| Descending | 0 / −80 / +8 | 864 | 432 | 72 / 72 | 72 / 0 |
| Closed cavity | 0 / 0 / 0 | 440 | 920 | 0 / 0 | 72 / 24 |

All four intervals had zero geometric drift and zero owned-volume defect. The
deep tank had no enclosed dry cells after the interior particles were stripped.
The closed cavity preserved its eight enclosed dry cells, 24 wet-facing sides
and the enclosing liquid component's lineage. A partially filled cell counts as
wet in the adjacency diagnostic. The check enumerates wet–dry lattice faces
inside the domain, with the roof air as the flood-fill seed.

Tests cover missing accepted and destination occupancy, understated motion,
transfer and reseed rejection, full-cell-only reseeding, and a forward/reverse
replay. Injected failures after the first trial mutation at each stage preserve
the complete accepted snapshot digest, including carrier history, time and the
next particle ID.

Run from the repository root with uv 0.13.7 or later and a populated cache.

```powershell
uv run --project reference --locked --offline python -m fluidity_reference.experiments.band_core
uv run --project reference --locked --offline pytest reference/tests/test_band_core.py -q
uv run --project reference --locked --offline pytest reference/tests tools/tests -q
```

Measured with uv 0.13.7, Python 3.13.18 and NumPy 2.2.6 on Windows. The focused
suite passed all six tests. The combined reference and tools suite passed 338 tests.

The fixture scope is one persistent liquid component, prescribed geometry and
analytic carrier samples. The checks leave M2 occupancy and motion authentication,
production support near cuts and walls, actual P2G and particle residual reset,
mesh reconstruction, topology changes, finite compressibility, and CUDA
performance to their integration gates. See the
[M3 plan](../superpowers/plans/2026-11-11-m3-adaptivity-and-band.md#task-4--particle-band-and-eulerian-interior).
