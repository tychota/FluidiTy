# M1c audit after the first Blender bake

Audit baseline `4cf17151052e33a947338848cfd6bb5b2a237c3e` includes the first offline Blender bake.
The M1 core Simulate publisher, execution loop and schemas are byte-identical in Git to the baseline
before that merge. M1-B adds a separate bridge, package and extension around those contracts.
[M1-B merge](https://github.com/tychota/FluidiTy/pull/47)

## Observed

- Rechecked the M1-B ZIP SHA-256 `e1281041fe3e943c6479732ed4a25c702e3939d6416a1e166a48a90e57f2eff5`
  and ran the merged `tools/blender/verify_m1_package.py` against a fresh offline profile. The test
  exited zero on Blender 5.2.3 LTS `e13b728c6880`. Check Scene, Preview, CUDA Bake, cancellation,
  Resume, Inspect, repeat attachment, relocation, independent relink, failed replacement and
  translation checks ran. Blender reported missing-reader warnings while the old path was deliberately
  absent before relink. The subsequent fresh factory process rendered distinct frames 0 and 10
  through the saved native reader with the extension disabled. [M1-B evidence](../blender/m1-b-evidence.md)
- The prior focused native run covered 68 cache, schema, CPU replay, lifecycle and inspection cases
  against the unchanged Simulate core. The checked reader also revalidated all 217 publications
  of the pinned CUDA drop, including hash and clock checks. [M2 fixed-cache evidence](m2-fixed-cache-evidence.md)

## Authority and handoff

| Contract | M1 implementation | M2 handoff |
| --- | --- | --- |
| Accepted state | The scheduler owns accepted clock and epoch. CUDA frame snapshots keep the writer lease alive until copy completion and compare the result to committed state before publication. | Collect only accepted carrier samples and events. Discard attempt-local data on rejected trials and on frame-capacity replay. |
| Publication | `LiquidCache3D::PublishFrame` commits content and the JSON mirror before its binary no-replace manifest. Checkpoint reference publication and pruning occur independently. | Put derived products under separate stage identities. Record exact upstream content. Compare identical replay before reuse. Keep old point-only commits readable. |
| Data availability | `FLMF` references a point blob. `FLST` checkpoints carry selected cell occupancy and face velocity, validity and side metadata. Only retained checkpoints have this state. | Publish occupancy with explicit spatial validity and timestamped carrier samples as frame-local artifacts. Endpoint velocity or a retained restart checkpoint cannot supply an interval or certify a cavity. |
| Replay | Native replay opens an authenticated read-only cache and compares any existing point publication. | A Mesh checkpoint needs geometry, IDs, repair lineage, controls and upstream identity. Whitewater needs its own source, random and hysteresis state. Keep read-only inspection separate from restart authority. |
| Blender | The bridge authenticates every frame through the native reader before export. A relative Alembic Points binding and node group render without the extension. | Qualify separate mesh, whitewater and volume carriers. Bind product identity and exact content, and keep Foam Projection and material changes in saved render datablocks. |

## Remaining gates

M1-B supplies the real artist workflow needed to start the M2 Mesh and transport implementation. The
audit leaves the frozen M1 piston receiving-volume gate and final matched CPU/RTX scene matrix open.
Neither the bounded offline Blender bake nor fixed-cache reconstruction certifies moving-wall
receiving volume or the integrated M2 breaking wave. A later Mesh quality receipt needs authoritative
occupancy, interval samples, production replay, and contact and thin-feature fixtures.
