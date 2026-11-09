# Simulate liquid cache v1

The four little-endian roots use identifiers `FLST` (state/checkpoint), `FLPT`
(points), `FLMF` (complete frame manifest), and `FLRS` (resolved scene).
Every table field has an explicit numeric ID. Struct layouts and enum values are
frozen; add new table fields without reusing IDs. `v1/` is the conformance baseline.
The deliberately incompatible random-seed type under `incompatible/` must fail
`flatc --conform` with that specific field diagnostic.

CMake generates headers into the ignored build tree using the absolute build-context
Conan flatc 25.12.19 executable. The locked native validator requires all six schema
CTest cases, including the negative control, on every platform/configuration.

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
Exact content IDs instead hash artifact bytes with ordinary unkeyed BLAKE3-256.

## Downstream contracts

- Task 7B calls `VerifyLiquidContent3D` with the trusted manifest/index reference
  before decoding referenced content. `LiquidContentName3D` derives the sole allowed
  name `blobs/<64 lowercase digest hex>.<extension>`; the decoder rejects every other
  spelling. Filesystem confinement, exclusive writes, durability, atomic publication,
  JSON inspection mirrors and checkpoints' independent retention index remain 7B.
  The immutable frame manifest references points only, so pruning checkpoints cannot
  invalidate point playback. Compare the decoded clock/fingerprint/execution ID with
  the expected job and checkpoint index before publishing/resuming.
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
