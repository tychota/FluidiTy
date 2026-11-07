# FluidiTy

Particle–grid multiphysics for Blender, designed around one sparse staggered grid, independent numerical references and measurable output quality.

**Current status:** the independent Python operator reference and two bounded Lean proof pilots are implemented. The ten complete proof families, production solver and CUDA validation remain outstanding. Research timings are not measured product performance.

- [English design specification](docs/superpowers/specs/2026-11-07-fluidity-design.md): product requirements, numerical and execution contracts, milestones and acceptance gates.
- [Scientific method paper](docs/papers/fluidity-method.pdf): equations, figures and detailed particle/transfer/grid flowcharts; [Typst source](docs/papers/fluidity-method.typ).
- [Ten formal contracts](docs/references/formal-contracts.md): bounded exact-arithmetic proof obligations and numerical implementation bridges.
- [Verified neural-surface evidence](docs/references/neural-surface-evidence.md): published artifacts, preprocessing dependencies, benchmark scope and first comparison.

The first meshing experiment compares a deterministic geometric reconstruction with the published Zhao–Shinar–Schroeder CNN on matched full-volume and narrow-band caches. Neural integration is gated on input coverage, quality, total resources, repeatability and artifact permissions.

## Build the paper

With Typst 0.15.1, CeTZ 0.5.2 and the fonts named in the source:

```sh
typst compile --root . docs/papers/fluidity-method.typ docs/papers/fluidity-method.pdf
```

See [paper sources and diagrams](docs/papers/README.md). Internal review notes and downloaded third-party assets are excluded from version control. The project license must be selected before public release; no third-party source or model weights are distributed here.
