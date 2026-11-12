# Live graded family transaction implementation plan

Use the accepted graded state as the input to the family remap. Prepare a bounded candidate with the canonical face roster and exact particle-owned inventories. Refuse a candidate that needs uncertified particle resampling. Execute the physical trial using a separate frozen owner and install that owner only after host and device acceptance.

- [ ] In the native scheduler test, refine a particle-matched family, reject late, and check encoded state equality. Then accept refinement and coarsening, and restore an intermediate adaptive checkpoint.
- [ ] Add an explicit family-change attempt to the production scheduler. Keep the prior owner on every refusal and rejection. Use the ordinary accepted clock and backend publication.
- [ ] Run CPU and real CUDA rollback and topology-changed replay. Verify v2 checkpoint inventory and canonical slots.
- [ ] Add accepted-epoch selection, certified split and merge, and band ownership in subsequent integration. Qualify the full reference wave once M2 publishes usable accepted surface and motion data.

## Verified native entry

`AttemptWithAdaptiveFamily` accepts an explicit family request when the existing particles rebin to the conservative remap inventory exactly. The ordinary acceptance transaction publishes the new owner. Flux that cannot be represented by the candidate carrier causes refusal. Automatic selection and certified particle split and merge remain separate work.

CPU and real SM120 CUDA matched after killing the qualification process following its refined checkpoint and resuming in a fresh process. The 12-leaf checkpoint BLAKE3 is `ee9acceafbdbb4b90de89918fa259eb6c82c0e12452d5d5d981849948f5a106f`. The coarsened five-leaf result is `a9914eb25efda25dc5d8e5163743082ca1552c629b43654e23694dd6eede961b`.

```powershell
uv run --offline --project reference --locked --group native ctest --test-dir out/build/live-cpu -V -R 'AdaptiveFamilyReplay.cpu|PublishesLiveAdaptiveFamily' --no-tests=error
uv run --offline --project reference --locked --group native ctest --test-dir out/build/live-cuda -V -R 'AdaptiveFamilyReplay.cuda|CudaAdaptiveSimulate3D' --no-tests=error
```

Compute Sanitizer reported zero errors in `CudaAdaptiveSimulate3D.PublishesLiveFamilyAndPreservesRejectedDeviceEpoch`. Receipts are `out/live-final-cpu.log`, `out/live-final-cuda.log`, `out/live-final-memcheck.log`, and `out/live-cpu-replay.log`.
