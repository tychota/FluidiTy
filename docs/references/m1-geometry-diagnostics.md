# M1 geometry diagnostics

These independent checks help separate reconstruction error from solver transport. They do not change the resolved field, pressure operator, or the M1 acceptance gates.

## Piston material mesh

Run the material diagnostic on a retained primary or refined piston evaluation. Take both SHA-256 values from the trusted runner receipt.

```text
python tools/native/diagnose_piston_material.py <evaluation-directory>
    --manifest-sha256 <capture-manifest-sha256>
    --measurements-sha256 <frozen-measurements-sha256>
```

The diagnostic reconstructs vertices of the frozen 16 × 8 × 8 lattice from stable particle IDs. It partitions each material cell into six signed tetrahedra, clips the transported mesh to the tank and the y = 0.5 m plane, and reports volume and wall section separately. The sufficient injectivity check is a uniform displacement-gradient bound on the piecewise affine map. A rejected bound does not by itself prove an inversion.

The wall section samples the inward side of the wall. The swept strip clips the mesh with the piston footprint frozen at the start of each accepted step. The midpoint estimate linearly interpolates the measured endpoint positions. Near-wall offset samples measure sensitivity to the extrapolation. They are not a correction to the wall position or an independently observed contact patch.

The exact retained CPU cases give above-plane endpoint gains of 0.00789957 m³ and 0.00789073 m³. Both preserve clipped material within 0.012%, but the dry sweep and void terms in this mesh representation are 19.65% and 25.37% of the frozen wall integral. The frozen receiving comparisons continue to fail at 7.617% and 7.625% against the 1% limit. The historical 0.055% agreement was a scratch endpoint against an analytic stroke. A qualification change needs independent wet-wall evidence.

## Isolated drop reference

The checked covariance ellipsoid in `fluidity_reference.experiments.isolated_drop_geometry` is a reference for resolved, filled, separated components. The tests check true Euclidean distance separately from the quadratic level set, outward curvature, component separation, and corner visibility under translation and rotation.

On the archived public seed, the fixed-radius union has 552 wet cell centers and no wet corners. The reference ellipsoid has 528 wet centers and 511 wet corners. A separate native CPU first-step witness activated pressure on 528 cells and converged in nine iterations. Full native production integration and the public oscillation test remain open. Keep the frequency, damping, timestep and actual-device requirements in the [scene plan](../superpowers/plans/2026-11-09-real-3d-liquid-acceptance-and-throughput.md).
