# FluidiTy

Particle–grid multiphysics for Blender, designed around one sparse staggered grid, independent numerical references and measurable output quality.

**Current status** — The independent Python operator reference, bounded native CPU operators and two Lean proof pilots are implemented. The Windows CUDA hardware probe exercises shared vector arithmetic. The ten complete proof families, production solver and CUDA operator validation remain outstanding.

- [English design specification](docs/superpowers/specs/2026-11-07-fluidity-design.md): product requirements, numerical and execution contracts, milestones and acceptance gates.
- [Scientific method paper](docs/papers/fluidity-method.pdf): equations, figures and detailed particle/transfer/grid flowcharts; [Typst source](docs/papers/fluidity-method.typ).
- [Ten formal contracts](docs/references/formal-contracts.md): bounded exact-arithmetic proof obligations and numerical implementation bridges.
- [Verified neural-surface evidence](docs/references/neural-surface-evidence.md): published artifacts, preprocessing dependencies, benchmark scope and first comparison.

The first meshing experiment compares a deterministic geometric reconstruction with the published Zhao–Shinar–Schroeder CNN on matched full-volume and narrow-band caches. Neural integration is gated on input coverage, quality, total resources, repeatability and artifact permissions.

## Development tooling

The Python reference uses uv >=0.13.7, Python 3.13 and a committed `reference/uv.lock`:

```sh
uv sync --project reference --locked
uv run --project reference --locked pytest reference/tests tools/tests -q
```

See the [reference instructions](reference/README.md) and [proof tooling](proofs/README.md).
The native build uses CMake presets, Ninja and locked Conan dependencies. The
[native validation guide](docs/native/README.md) covers CPU builds, the Windows CUDA
hardware gate and the evidence each run produces.
The [CUDA optimization method](docs/native/cuda-optimization.md) covers production profiling for M2–M4.
Python tooling uses uv and Lean
dependencies use Lake.

## Build the paper

With Typst 0.15.1, CeTZ 0.5.2 and the fonts named in the source:

```sh
typst compile --root . docs/papers/fluidity-method.typ docs/papers/fluidity-method.pdf
```

See [paper sources and diagrams](docs/papers/README.md). Internal review notes and downloaded third-party assets are excluded from version control. The project license must be selected before public release; no third-party source or model weights are distributed here.
