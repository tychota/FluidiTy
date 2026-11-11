# M3 Conservative Remap Reference Plan

**Goal** — Check extensive inventories and signed face fluxes for a single aligned 2:1 parent/child remap before native topology integration.

- [x] Add tests for signed momentum and enthalpy, current-volume weighted constituents, conservative restricted/prolonged round trips, and a bounded affine reconstruction.
- [x] Add rejected inputs for duplicate face owners, invalid composition, nonfinite values and lossy subnormal prolongation. Verify callers retain their input data.
- [x] Implement a fixed eight-child reference with a shared limiter for total volume and constituent volumes. Use four uniquely owned subfaces for face restriction/prolongation.
- [x] Run targeted and existing reference tests. Record evidence and retain the production topology, aperture-aware divergence prolongation, EOS reconstruction, and committed-state rollback gates for native integration.

## Evidence

The manufactured affine stencil round-trips every inventory exactly on the declared dyadic case. A shared limiter preserved total and constituent volume at both zero and fractional limiting. Signed subfaces restricted to +0.25 m³/s and prolonged back to −0.25 m³/s in the reverse case. The checks rejected duplicate ownership, nonfinite flux, invalid constituents and a parent mass that could not survive division among eight children. Inputs were preserved on rejection.

The focused suite first failed on the missing module. The final locked run reported 338 passed for `reference/tests tools/tests`, including the six remap cases. Run from the repository root:

```powershell
uv run --offline --project reference --locked --group native pytest reference/tests/test_adaptive_remap.py -q
```

The reference covers fully owned 2:1 cells and four equal-area subfaces. Partial apertures, pressure-compatible interior flux prolongation, EOS reconstruction, topology hysteresis and trial publication remain native integration work.
