# Native adaptive band handoff and core transport

The graded scheduler can retire or reseed one explicitly requested liquid leaf. The caller certifies full occupancy and supplies a valid carrier from the previous accepted epoch. The candidate checks mass, reference and current volumes, momentum, and particle IDs in the scheduler's exact accumulation order. A rejection leaves the previous host and device ownership published.

In the bounded stationary planar tank, an Eulerian leaf requires valid implicit-liquid carriers and stays below the open top. Projected subface flux updates only Eulerian-owned inventories. The donor-cell transport preserves the signed incidence and refuses nonzero exchange with a particle-owned leaf. An exact drain retries before publishing a state that cannot restart. Particle motion into an Eulerian-owned leaf also retries. The old core carrier is used before the old-grid snapshot.

The independent transport tests cover nonzero coarse/fine flux, signed momentum and enthalpy, capacity and donor limits, unsupported exchange, and exact drain. The scheduler fixture exercises retirement, a closed stationary core, reseed and CUDA bank rollback. CPU and SM120 CUDA produced identical hard-kill replay receipts. The Eulerian checkpoint BLAKE3 is `0281b0da2020a984e3c67eea7f8665a8a4977da655fa0f1ae68f648e2b575d92`. The particle-owned continuation is `9f9e180832a6eedef56bc6c13112adb1eeb2458891b8488b6a72b27514a894a9`.

```powershell
uv run --offline --project reference --locked --group native ctest --test-dir out/build/band-cpu -V -R 'AdaptiveBandReplay.cpu|PublishesAdaptiveBand|RejectsEulerianDrain|AdaptiveLiquid(CoreTransport|BandHandoff)3D' --no-tests=error
uv run --offline --project reference --locked --group native ctest --test-dir out/build/band-cuda -V -R 'AdaptiveBandReplay.cuda|CudaAdaptiveSimulate3D' --no-tests=error
```

The host-only `PrepareAdaptiveLiquidOwnerExchange3D` proposal uses the projected Cartesian-normal carrier and aperture on each canonical interior subface. Each nonzero face incident to Eulerian ownership creates one donor debit and matching receiver credit for mass, reference and current volume, momentum, and signed enthalpy. Core-to-core and cross-owner faces share one step-start donor budget. The proposal checks the represented binary64 leaf balances and lists the particle-owned leaves requiring reconciliation. Publishing a proposal requires actual particles that represent the candidate inventories, plus a separate enthalpy ledger. The scheduler still refuses nonzero cross-owner flux.

The detached `PrepareAdaptiveLiquidParticleExchange3D` candidate regenerates that plan for a clock-bound step-start particle snapshot. It preserves IDs, record order, and positions, then checks that the last resident in each affected leaf can represent the exact correction at the certified constitutive state. The caller certifies support for the revised carrier. Empty recipients, complete particle drains, inconsistent EOS, and mixed enthalpy are refused. A future runtime must partition the mixed-face ledger from particle trajectories so the same transfer is applied once.

Automatic placement from the 6h band and 2h overlap, blended carriers and conservative cross-owner exchange remain integration work. Family remap on Eulerian-owned layouts remains unavailable. A handoff candidate currently requires half-open particle coverage, so it can refuse particles exactly on solid upper tank planes. Graded transfer and core transport still use transient host allocations. The matched-error full wave and simultaneous GPU memory gates remain open.

## Certified planar placement in Simulate

`AttemptWithAdaptiveBandPlacement` derives the signed distance to the frozen
planar top and the leaf occupancy from the accepted scheduler state. The caller
provides the accepted epoch, leaf key, duration-bound sweep and same-side carrier
evidence. The scheduler uses the finest locally active width to apply the 6h
particle depth and 2h carrier overlap before choosing retirement, reseeding or
the ordinary trial. Retirement and reseeding use the existing atomic publication
path. Missing carrier support and stale epochs fail before the trial.

The deep planar fixture exercises late rejection, retirement and reseeding. CPU
checks the committed epoch and represented inventory. SM120 reads the accepted
device bank after rejection and both ownership transitions. The existing
mixed-owner fixture exercises fresh-process replay.

```powershell
ctest --test-dir out/build/placement-cpu -R 'BandPlacement|AdaptiveBand|Eulerian' --output-on-failure
ctest --test-dir out/build/placement-cuda -R 'CudaAdaptiveSimulate3D|AdaptiveBandReplay.cuda' --output-on-failure
```