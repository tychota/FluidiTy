# Zhao ONNX CPU comparison

This is an isolated research run of the published ONNX weights with an independent implementation of the
preprocessing. The [original artifact review](neural-surface-evidence.md) identifies the paper, archive and
unresolved software and weight terms. This run used the verified archive directly and left third-party
source, data, weights and runtime files in ignored local storage.

## Reproduction

Use Python 3.13.18 and an isolated environment with `numpy==2.2.6`, `scipy==1.15.3` and `onnxruntime==1.23.2`.
Supply the authors' ZIP with SHA-256 `bc4304606c6b378bfd06dec928025e38d8cf6fd767f70a43d8bdc0f56c3e16ca`
and the exact [M1 drop reader and run](m2-fixed-cache-evidence.md#pinned-input-and-reproduction).

```powershell
python -m tools.native.m2_zhao_experiment --archive $archive --reader $reader --run $run --output $report --threads 6
```

The reader revalidates all 217 M1 publications before the experiment uses frame zero. The archive's ONNX
SHA-256 is `417263e65280fe6efbdb1cab603c99235b011d85478de5c31641c6f16b461be3`.
The script checks the archive, model and tensor shapes. It uses the source-derived quartic splat,
trilinear density normalization, eight staggered parities, occupied-cell thresholds, fixed 1024-row
batching and output multiplication by feature spacing. It accumulates features in FP64 before
converting to FP32. The CPU runtime, summation order and query order within each parity differ from
the upstream TensorRT and OpenMP execution. Both variants in a comparison use the same CPU setup.
The checkout's unit tests use a separate scalar stencil oracle and exercise an omitted particle that
affects normalization but is outside direct patch support.

On Windows 11 with a Ryzen 7 9700X, two runs matched all ten numerical records and feature, scalar and
mesh digests. Both took about 22 s including input, preprocessing, inference and extraction. Peak
working sets were about 603 and 601 MiB. The script records load time and all per-variant costs.

## Supplied dam-break control

The archive's first OBJ contains 857,314 particles. The test uses a 13³ query crop around the initial
top surface, with the supplied `−0.1 m` origin and `0.005 m` output spacing. The crop retains 39,228
particles within the conservative 9 h_N L∞ halo. All 169 vertical columns had one crossing, with
3.13 mm RMS displacement from the nominal 0.5 m plane. Monolithic and x-tiled features and scalar
values matched byte-for-byte. The monolithic crop took 1.06 s of preprocessing and 0.77 s of inference.
This is a crop of the supplied OBJ with complete query support. The upstream full-volume field,
TensorRT parity and closed extraction on that scene remain open.

## Verified M1 drop

The frame-zero authored ellipsoid supplies the radial proxy and the signed volume reference. Use
`h_phi=0.0125 m`, `h_N=0.025 m`, and one globally aligned 41³ query lattice. Synthetic bands select
particles by absolute radial distance to the authored surface. The frozen-pruning rows hold the
full-data inference mask fixed. The recomputed-pruning rows apply the same occupancy test to the
trimmed input. The simulated M1 cache is full-volume and supplies no actual solver-band or reseeding
comparison.

| Input | Particles | Field RMS / max difference (mm) | Changed pruning / sign queries | Components | Raw volume error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 552 | 0 / 0 | 0 / 0 | 1 | +18.15% |
| 25 mm, frozen | 272 | 12.92 / 146.45 | 0 / 2,043 | 7 | −17.15% |
| 25 mm, recomputed | 272 | 13.10 / 146.45 | 111 / 2,052 | 2 | −17.37% |
| 50 mm, frozen | 416 | 10.94 / 125.77 | 0 / 1,293 | 7 | −11.80% |
| 50 mm, recomputed | 416 | 10.98 / 125.77 | 9 / 1,302 | 2 | −12.10% |
| 75 mm, frozen | 520 | 5.42 / 83.20 | 0 / 80 | 2 | +18.96% |
| 75 mm, recomputed | 520 | 5.43 / 83.20 | 8 / 80 | 2 | +18.96% |

The full-data mesh had 7.73 mm radial RMS error, Euler characteristic 2 and zero boundary edges,
nonmanifold edges, winding defects or zero-area triangles. At the same extraction spacing the
[geometric density baseline](m2-fixed-cache-evidence.md#geometric-reference) had 4.99 mm radial RMS
error and −11.60% raw volume error. Full-data CNN preprocessing, inference and extraction/metrics
took 0.08, 1.54 and 0.22 s in the first run. X-tiling with complete halos preserved full-data feature,
scalar and mesh bytes. Every synthetic band changed signs even with the original pruning fixed.

## Promotion verdict

1. **Published reproduction remains open.** The pinned CPU model, preprocessing oracle and bounded
   controls ran. Full supplied-scene output and independent TensorRT parity remain unverified.
2. **Band fidelity failed on this synthetic control.** Even retaining 520 of 552 points changed 80
   query signs and added a component. Real solver-band and reseeding evidence remains open.
3. **Complete-halo tiling matched for the two tested workloads.** Production tile geometry, device
   determinism and extraction at scale remain open.
4. **Keep the geometric backend.** The full-data result missed the per-component volume target,
   combined-job cost remains unmeasured, and the source and weight terms remain unresolved.
