# Scientific method and diagrams

The paper accompanies the canonical [English design specification](../superpowers/specs/2026-11-07-fluidity-design.md). It describes proposed equations and acceptance contracts; it records the bounded reference and proof pilots without claiming completion of the proof families.

- `fluidity-method.typ`: English paper, equations, ten formal contract families, independent-reference scope and bibliography.
- `fluidity-figures.typ`: reusable CeTZ figures, including neural dependency coverage, hybrid meshing and validation layers.
- `fluidity-flow.typ`: three A3 landscape algorithm pages, Figures 11a–11c, separating particles, transfers and grid operations.
- `fluidity-method.pdf`: compiled paper, including the landscape flowcharts.

Build from the project root with Typst 0.15.1 and CeTZ 0.5.2:

```sh
typst compile --root . docs/papers/fluidity-method.typ docs/papers/fluidity-method.pdf
```

The source uses Libertinus Serif/Sans, New Computer Modern Math and DejaVu Sans Mono. CeTZ is pinned in the imports. The first build may require obtaining the package if it is not already cached.

Public supporting evidence:

- [Neural surface artifacts and dependency analysis](../references/neural-surface-evidence.md).
- [Ten formal contracts and implementation bridges](../references/formal-contracts.md).

The independent Python reference and two Lean pilots are implemented; the full proof families and production tests remain future work. Internal review records and downloaded third-party assets are excluded from version control.
