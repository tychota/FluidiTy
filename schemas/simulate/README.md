# Simulate liquid cache v1

The five little-endian roots use identifiers `FLST` (state/checkpoint), `FLPT`
(points), `FLMF` (complete frame manifest), `FLRS` (resolved scene), and
`FLCR` (independent retained-checkpoint reference).
Every table field has an explicit numeric ID. Struct layouts and enum values are
frozen; add new table fields without reusing IDs. `v1/` is the conformance baseline.
The deliberately incompatible random-seed and checkpoint schema-version types under
`incompatible/` must fail
`flatc --conform` with that specific field diagnostic.

CMake generates headers into the ignored build tree using the absolute build-context
Conan flatc 25.12.19 executable. The locked native validator requires all eight schema
CTest cases, including both negative controls, on every platform/configuration.

`fluidity/host/liquid_cache_3d.h` is the checked C++20 boundary. Bytes are bounded,
8-byte aligned, structurally verified, versioned and length checked before semantic
access. All semantic checks precede decoded record allocation. Fields preserve
physical order and FP64 values without copying ABI padding, scratch or reservation
capacity. Checkpoint resume capacity plans are separate metadata. The encoders use
a checked nothrow arena sized at 4096 bytes plus 512 per child table (two tables per
emission source); the configured byte ceiling bounds that conservative reservation.
All record families also have a hard ceiling of 2^20.

Physical state persists geometry, clock/epoch, seed and next ID; particle
ID/tag/position/stored velocity/APIC slopes/affine flag/mass/reference volume/J/side/PIC;
block indices/generation/activation and inactive ages; cell occupancy and pressure;
owned face indices/slots/velocity/validity/side/eligibility; source child ordinals and
fractional remainders. The uniform point format carries ID/tag/position/velocity/
radius/J and level zero. Checkpoints require completed-frame clocks.

Resolved scene records materialize every scalar default. Canonical ordering follows
schema field order for child construction and pinned generated table field order;
inline field/obstacle struct alignment precedes their table extent. Signed zero is
normalized, and inactive field/obstacle values are zero. Positive infinity is allowed
only for bulk modulus. The canonical fingerprint is BLAKE3-256 of ASCII
`fluidity/simulate/resolved/v1`, one NUL, then canonical `FLRS` bytes. It includes
geometry, seed, material, solver options, forces, field, obstacle, ordered sources,
model version, backend, numerical-build artifact digest, and ordered upstream content
roles/ordinals/lengths/digests. Local paths, process metadata and capacities are absent.
Execution ID uses `fluidity/simulate/execution/v1`, one NUL, and the 32 fingerprint bytes.
Checkpoint and manifest encoders reject a mismatching execution ID as InvalidArgument;
readers reject it as DataLoss. This internal consistency check is separate from
comparing valid stored metadata against the current job.
Exact content IDs instead hash artifact bytes with ordinary unkeyed BLAKE3-256.

## Downstream contracts

- The cache store calls `VerifyLiquidContent3D` with the trusted manifest/index
  reference before decoding referenced content. `LiquidContentName3D` derives the
  sole allowed name `blobs/<64 lowercase digest hex>.<extension>`; decoders reject
  every other spelling. Immutable frame manifests reference points only.
- Task 8 supplies a completed coherent host state after the snapshot fence; codec
  calls retain no input aliases. Encoded owners can move to the writer after the
  call. Codecs are setup/I/O-boundary operations and are not physical-attempt APIs.
- Task 9 resolves defaults into `LiquidResolvedSceneView3D`, hashes exact upstream
  bytes, supplies numerical build/backend identity, and compares fingerprints and
  execution IDs before resume. Decoded `LiquidResolvedScene3D::View()` borrows its
  owning arrays; reacquire after moving the owner. Capacity fields come from resume
  metadata and must be requalified against current resource availability. Process
  liveness/heartbeat never participates in immutable record bytes or fingerprints.

## Independent canonical fixture

`native/tests/fixtures/liquid_cache/resolved_defaults.json` is the explicit default
scene. Generate with pinned `flatc --binary --force-defaults -I schemas/simulate -o
<ignored-output> schemas/simulate/resolved_scene.fbs <fixture.json>`. Set root
`byte_length` to the produced file size without rebuilding; the checked-in `.hex`
records those 488 exact bytes. Domain-prefixed BLAKE3-256 is
`f9970683b8708e8963451711f5f03a62d1ec6830c5815f43dda539fec94eddd0`.
This independent parser oracle caught a real six-byte vtable-extent discrepancy in
the first C++ encoder. Raw content hashing also has standard empty and `abc`
known-answer tests.

## Filesystem publication and restart

`LiquidCache3D::Create(root_directory, resolved_scene)` validates canonical settings
and exclusively installs `resolved.flrs`. `Open(root_directory, expected_fingerprint)`
verifies canonical bytes and the caller's expected job identity. `OpenForInspection(root_directory)` instead validates portable self-consistency
without creating or writing files and requires the existing lease file. Its returned
instance can inspect resolved inputs and point frames, but rejects publication and
`ReadCheckpoint`; resume requires opening with an externally expected fingerprint.
The root is a trusted cache-owned directory; record strings are never opened as arbitrary paths.
The native lease serializes readers, publishers and pruning across processes. It
returns Aborted while another operation holds the lease. Process exit releases it;
the persistent `.writer.lock` is not a heartbeat or a content identity.

`PublishFrame` writes verified content to `blobs/<digest>.<kind>`, including the
manifest content object, and writes a deterministic JSON mirror to
`frames/<20-digit frame>.json`. Native exclusive creation writes temporaries beside
their destinations. Windows flushes file data with FlushFileBuffers and promotes
with MoveFileExW(MOVEFILE_WRITE_THROUGH), without replace-existing. POSIX uses fsync
on data, atomic no-replace link publication, unlink of the temporary, and directory
fsync. The binary `frames/<20-digit frame>.flmf` is the point-frame commit and is
published last. Existing identical bytes are verified and retain their modification
times. Reuse, including a concurrent promotion winner, completes a parent-directory
barrier before success or retention. Root creation builds missing ancestors one at
a time, synchronizing each parent before descendants; retries also synchronize
existing links left by an interrupted creation. POSIX file/lease opens are
nonblocking until fstat rejects nonregular objects, so a FIFO cannot wait for a
peer. Unequal committed content is never overwritten. Both checkpoint publication
paths compare existing immutable reference identity before any candidate writes,
including when its old state payload has already been pruned. These guarantees assume a
local filesystem honoring the native primitives, not a network share.

Frame zero explicitly publishes the initial completed state at time/step/epoch
zero before later frames. Every tenth frame supplies a full checkpoint. Exported
points may be a stable-order subset, but their shared particle fields, exact clock,
seed and geometry must agree with the checkpoint and resolved job. Readers verify
the deterministic mirror and trusted kind, exact byte length and digest before
decoding referenced records. JSON stores seconds explicitly and only cache-relative
content names, never local directories or process metadata.

After the frame commits, a separate immutable
`checkpoints/<20-digit frame>.flcr` reference binds its exact manifest and full-state
content. No unpublished future checkpoint blob is resumable. `PublishCheckpoint`
can finish this metadata step after an interrupted frame publication. A publication
error after the commit can leave the new frame readable; callers reread it, then
retry exact inputs or repair its checkpoint metadata. Cancellation hooks run at
documented real I/O boundaries, including immediately before manifest publication.

Retention keeps the immutable reference inventory and its newest two full-state
blobs. It verifies the retained states before deleting any older verified payload.
The global newest two references define eligibility even if an interrupted prune
leaves extra old blobs. This avoids relying on durable unlink ordering on Windows
and preserves historical checkpoint identity for replay. Initial
zero remains through checkpoint ten and is pruned once twenty is available.
Frame manifests, point blobs and their inspection mirrors are never pruned.
Failures during retention leave point playback intact and can leave extra valid
checkpoints or unreferenced blobs. Abandoned temporary files and valid orphan
content are ignored on restart; exact orphan content can be reused on retry.
There is no automatic broad orphan deletion.

`ReadResolvedScene`, `ReadCommittedFrame(frame)` and `ReadCheckpoint(maximum_frame)`
return independent checked owners. The latter returns the newest retained eligible
checkpoint among the global newest two references and fails closed on corrupt
metadata/content. Reading an older
point frame does not require its now-pruned checkpoint. Task 8 supplies a fenced
completed snapshot; Task 9 owns operational status, heartbeat and process liveness;
Task 10 consumes these checked restart/read interfaces.
