# Reference-wave admission preflight

The reference design calls for an 8 × 20 × 8 m domain, a 3 m inlet tapering over a 20 m beach, a 1 m, 4 s Stokes swell, 3.5 cm simulation cells near the fixed camera, and 240 frames at 24 fps. The camera is specified as 35 mm and 1920 px, about 2 m from breaking. The repository at `6b2be6faec21c20747f68654375e9752ba6fff1e` has no authored reference scene and seed, camera refinement footprint, or complete output settings. The first admission dependency is that exact authored input and its measured accepted-state capacity. [Reference design](../superpowers/specs/2026-11-07-fluidity-design.md#21-reference-job)

The capacity diagnostic treats 8 m as width, 20 m as length, and the taper as a linear still-water wedge. The initial idealized volume is 240 m³. At 0.035 m, the full lattice spans 229 × 572 × 229 cells after rounding each extent upward. Multiplying 229 transverse columns by the sum of rounded-up wet depths sampled at the longitudinal column centers gives 5,671,872 initially wet cells. This counts the uniformly fine still-water geometry and omits support halo, free-surface motion, and sources. Public uniform execution reserves 1,024 particles, 128 blocks, 4,096 cells, and 24,576 faces. The graded topology admits 4,096 leaves and 24,576 faces. Even if every graded leaf were fine, its occupied fine volume would be at most 0.175616 m³, or 5.0176 m² in a complete layer one fine cell deep. Coarser neighbors also consume leaf and face capacity. The diagnostic gives no exact graded requirement without a camera footprint and canonical leaf roster.

## Independent bounded control

The largest authored wave available to this checkout is the M2 5 × 1 m partial-column strip at 0.25 m cells, 0.75 m inlet depth, and a 0.25 m, 2 s swell. The preflight runs its 49 samples, frame 0 through 48, with exact solver and authoring executable hashes, FLRS/FLST and volume-settings hashes, native execution IDs, stage commands, and read-only replay. It runs Simulate on SM120 and the derived native stages on CPU. The reference GPU ceiling needs simultaneous device allocations across the full pipeline. This control measures the Simulate allocator against its once-evaluated start budget and records allocator drain. [M2 bounded swell](../references/m2-bounded-wave-evidence.md)

```powershell
uv run --offline --project reference --locked --group native python tools/adaptivity/preflight_reference_wave.py --output out/reference-admission
uv run --offline --project reference --locked --group native python tools/adaptivity/preflight_reference_wave.py --solver out/wave-cuda/fluidity_simulate.exe --fixture out/wave-cuda/fluidity_simulate_fixture.exe --output out/wave-verified
```

The independently authored mainline Mesh and the M2 branch use different reconstruction source. The M2 branch at `5856e91b737d6c9f27da6543315fff52378363a8` records a fresh integrated native prefix through frame 41, independently replayed and reopened in Blender with surface, foam, and volume assets. It reports 0.49555% maximum absolute Mesh volume error and 2,753.56 s summed Mesh frame intervals. Normal continuation refused frame 42 on intersections. A separately selected fixed-cache sequence reaches frame 48 and has separate lineage. The in-progress material-entry worktree and frame-42 logs are experimental and do not extend the confirmed fresh prefix. [M2 branch receipt](https://github.com/tychota/FluidiTy/blob/5856e91b737d6c9f27da6543315fff52378363a8/docs/references/m2-bounded-wave-evidence.md)

### Measured run

| Stage | Published frames | Launch to exit (s) | Read-only terminal replay |
| --- | --- | ---: | --- |
| Simulate, SM120 | 0–48 | 73.568 | Frame 48 matched |
| Mesh, CPU | 0–1 | 13.213 | Frame 1 matched |
| Whitewater, CPU | 0–48 | 6.524 | Frame 48 matched |
| Volumes, CPU | 0–48 | 0.885 | Frame 48 matched |

The sequential attempted-stage elapsed time was 94.213 s, including the failed Mesh process. Mesh refused frame 2 because the raw scaffold enclosed 1.32813 m³ against a 1.7675 m³ target outside its correction budget. All four replays preserved cache bytes and timestamps. The terminal Simulate state digest was `426b45055474d9c314b378fc595a31ced89fd80b6c3255c77a55fd48d32fe5a3`, with 240 accepted substeps, 412 particles, 57 blocks, 380 cells, and 10,944 faces. Whitewater and Volumes reached frame 48 using that run's accepted source, with 595 foam points and 79 bubbles. The volume grid held 59,904 known voxels and 0.0413643 m³ represented gas.

The measured GPU was an RTX 5060 Ti on driver 610.88, with the executed kernel reporting SM120. Simulate allocator telemetry reported 17,102,864,384 total device bytes, 15,908,995,072 free bytes at its accounting baseline, and an enforced 11,972,005,068-byte budget. Its reservation peak was 113,726,464 bytes and zero remained after drain. A separate preflight `nvidia-smi` snapshot had 13,353,615,360 bytes free. The readings come from different measurement paths. Neither supplies a simultaneous full-reference pipeline peak.

The run receipt SHA-256 is `823aa340adaaee6eeb7f95a7143fea671c8b45112f4ea0b176290c89673752bc`. The solver and fixture SHA-256 values are `70af7859c63fb34f4f72d8fb8ad0b9d00458b5445be612ac8853c2b1d836334d` and `e0927108c4f4909739df9d322d4b09ef23f9a4fd712cd7564cf2b86bab6d221e`. The 240-frame reference, matched-error timing, and simultaneous GPU-memory gates remain open.
