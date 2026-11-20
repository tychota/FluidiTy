# Grid module design

Status: decided. Module 2 of the rebuild. Sections 3 and 4 were aligned with the other module designs on
2026-11-14; names shared between modules are fixed in `shared-interface.md`. Updated on 2026-11-14 with the
owner's answers (`owner-answers.md`) and `changes-to-apply.md`; every question of section 11 has a decision.
Updated again on 2026-11-14 with owner answer A17 (the grid is built on the GPU in a GPU run) and the grid entries
of `plan-decisions.md`, as carried by `plans/2026-11-14-rebuild-02-grid.md` (decisions 16 to 27). Updated again on
2026-11-14 with the reconcile pass (`plan-decisions.md` "Reconcile pass (2026-11-14)" item 11: grid buffers sized by
the planner, one builder buffers lent to several buffers; grid plan Tasks 6, 9 and 11 marked reconcile; decisions 28
to 31). Updated on 2026-11-15 with owner answers A40–A64 (`plan-decisions.md` "Owner answers A40–A64
applied": the shared cell size rule 3.8, `RunMemory` builders, `kOutOfMemory` last; decisions 36 to 40).

## 1. Purpose

The grid is the fixed scaffolding the simulation computes on: space is cut into cubic cells, pressure lives at
cell centres and each velocity part lives on the cell faces it crosses. Cells may have different sizes (each
size is half the previous one), and only the parts of space that hold material are stored, in chunks of
4 x 4 x 4 cells. The module also answers the one geometric question particles ask the grid: "which faces are
near this position, and how much weight does each get?"

**Owns:** cell and face identity and addressing; cell blocks and their neighbour tables; the size-step rule;
face pieces (identity, area, orientation, the two cells); cell corners; ranges of cell blocks and their border;
the weight curve and `NearbyFaceWeights`; grid versions; `GridBuffers`, the arrays a grid version is built
into without allocating; the per-cell, per-face, per-corner and per-block value buffers every module stores
its fields in (built on `Buffer` from `base/`, sized for a limit and bound to one grid version at a time);
pieces of a grid for the batched run, in the grid's fixed block order, and their stacked borders; the grid's
own memory estimate entries (the row type is in `base/`).

**Does not own:** any physical field (velocity, mass, pressure, `open_area`, `SolidSide`, distance to surface,
ingredient amounts, gas temperature: each is a `FaceValues`/`CellValues` owned by its module); which cells
should exist and the passes that resize them (step and cell-resize modules; the per-cell formulas of the shared
cell size rule are the grid's, 3.8, changed 2026-11-15, A41); remap and region transfer (cell
resize and batches); the sums over particles (transfer); what a face on the grid edge means physically
(pressure, solids); whether a face is outside every solid (`outside_solid`) or which plate side it is on
(`transfer::SolidsAtFaces`, filled by solids).

## 2. What exists today

| Piece on `main` | Where | What it does | What is wrong |
|---|---|---|---|
| Uniform sparse grid | `native/src/sparse_mac_grid_3d.cc` (`Create` 344-374, `FaceSlot` 388-409), `host/sparse_mac_grid_3d.h`, `base/mac_types_3d.h` | Sorted list of 4^3 blocks; each stores 64 cells and 3 x 64 low faces; a face is found by binary search of its block | One cell size only. A face's address depends on which blocks exist and carries no version. Search repeated per sample (CUDA report 9.1). |
| Uniform weights | same file, `BuildCheckedAxisStencil` 266-310, `PhysicalWeightGradient` 312-336; Python `reference/src/fluidity_reference/basis.py` | 27 quadratic tensor weights per axis with offsets and derivatives | 64 B per sample, 81 samples stored per particle = 5,184 B (CUDA report 4.1). Lines 83-198 and 236-336 are rounding-recovery machinery (A9). Needs "moved at most half a cell since grouping" and an 8^3 tile (222-234, 279-283). |
| Mixed-size cells | `native/src/adaptive_mac_topology_3d.cc` (`Create` 278-343, size-step check 138-199, `RefinementClosure` 345-424, border faces 228-274 and 434-469) | Flat sorted list of cells with `width_exponent` (0 = smallest); one record per shared face piece; uncovered sides become "exterior" pieces | Hard limits 4,096 cells and 24,576 faces (`:19-20`, `liquid_state_3d.h:170-171`). No blocks, so no ranges. Level numbering is the reverse of the decided `size_level`. Border pieces are a second record type with a `bool cell_is_tail`. |
| Mixed-size weights | `native/src/graded_mac_transfer_3d.cc` (`VisitSupport` 143-183, `BuildSupport` 185-267, `SolveAffineSlope` 271-309) | Radial weight `((1 - r^2/4) * face_width/cell_width)^2`, divided by the weight sum, offsets centred on the weighted mean, measured centred spread, 3 x 3 Cholesky solve with pivot floor `1e-12 * trace` | The procedure is right and is ported. The radial curve is not the uniform one: its spread is 167/396 and 76/165, not 1/4 (`docs/adaptivity/production-integration.md`), so a particle changes weights when it crosses into a mixed region. Containing cell found by linear scan (428-444). `solid_upper` closes the X and Z upper sides only (Y-up tank, 434-437). |
| Same-size "islands" in a mixed grid | `native/src/graded_quadratic_mac_transfer_3d.cc` | Uses the uniform weights only where all 27 faces are same-size, open and valid | Refuses everything else; a third code path. |
| Hierarchical weights | `base/coarse_fine_transfer_math_3d.h` | Cubic-normal weights across one size change, FP32, coordinates limited to 64 cells | An experiment (old paper "candidate"). Not the current method. |
| Inverse spread | `native/src/mac_gather_3d_internal.h:49-55` | Multiplies by `4 / cell_width` twice instead of inverting | Valid only when all 27 faces are used and same-size (numerics report 2.1). |

Y-up specifics: block order `(z, y, x)` is a storage order with no physical meaning; `solid_upper` and the
tank tests assume the open side is +Y. To compare a proof-of-concept result, rotate with the Foundation matrix
(`new = {old.x, -old.z, old.y}`): old Y-faces become Z-faces; old Z-faces become Y-faces with the sign of the
value flipped and `cell_low`/`cell_high` exchanged; old cell `z = k` becomes `y = -k - 1`.

## 3. Public interface

Folder `native/fluidity/grid/`, namespace `fluidity::grid`. Headers marked (S) are shared
(`FLUIDITY_CPU_AND_GPU`, C++20 subset); the others are CPU-only. Public headers: `grid_ids.h` (S),
`grid_error.h` (S), `grid_tables.h` (S), `weight_curve.h` (S), `nearby_face_weights.h` (S), `staggered_grid.h`,
`cell_block_range.h` (includes `grid_piece.h`), `grid_piece.h`, `grid_values.h`, `grid_memory_estimate.h`, and
`cell_size_rule.h` (S; NEW, A41, 3.8). Private
shared headers (A17, decision 17): `grid_build_status.h`, `grid_build_steps.h` (every per-cell, per-block and
per-face build rule, called by the CPU builder and the GPU kernels alike), `border_block_steps.h`. Private files:
`staggered_grid_cpu.cc`, `staggered_grid_gpu.cu`, `grid_cell_checks.cc`, `grid_values_cpu.cc`,
`cell_block_range_cpu.cc`, `cell_block_range_gpu.cu`, `grid_piece_cpu.cc`, `grid_piece_gpu.cu`,
`grid_memory_estimate.cc`; there is no `grid_values_gpu.cu` (values cross only through `base/` `Buffer` calls). Every
function in an (S) header is `constexpr` or `inline`, `noexcept`, and allocates nothing. Declarations that other
module designs asked for and this design did not have are collected in 3.7 (reconciliation) and marked NEW in
place (update of 2026-11-14: storage, binding, pieces, stacked borders, `ToName`, `BuildGridMemoryEstimate`).
Declarations of the A17 update are marked NEW (A17); they are copied from `plan-decisions.md` "Grid built on the
GPU (A17)". Declarations of the reconcile pass are marked NEW (reconcile); they are copied from `plan-decisions.md`
"Reconcile pass (2026-11-14)" item 11 and grid plan Task 6.

**One processor per run (A17).** Every input list, output list and working buffers of a grid operation lives on
the grid's (or the buffers') `run_on()`. Changed (A30): this is a documented `Requires:` precondition checked by a
debug assertion, never copied and never an error value (it was `kBadInput`). A grid operation
on the GPU returns to the CPU only its `GridBuildStatus` record and, for border search, a count. A GPU grid keeps
no CPU copy of its tables; `ReadGrid` is the one explicit copy. GPU tables, counts and failures equal the CPU
builder's byte for byte: the CPU builder is the oracle.

### 3.1 `grid_ids.h` (S): identity and addressing

```cpp
inline constexpr int kMaxSizeLevels = 16;        // NEW. 4 bits in an id.
inline constexpr int kBlockWidthInCells = 4;     // NEW. Ported value; tuning it is out of scope.
inline constexpr int kCellPlacesPerBlock = 64;   // NEW. 4 x 4 x 4.
inline constexpr int kFacePlacesPerBlock = 240;  // NEW. 3 axes x (5 planes x 4 x 4).
inline constexpr int kMaxCoord = (1 << 18) - 1;  // NEW. Coordinates fit 19 signed bits at every level.

struct CellCoord { std::int32_t x, y, z; };      // DECIDED. In cells of the cell's own size level.
enum class SizeLevel : std::uint8_t {};          // DECIDED word. 0 is the biggest cell.

// Stable identity: the same place and size has the same id in every grid version. Ids are opaque: other modules
// never build or split one by hand; they use ToCellId, ToCellCoord, ToSizeLevel, ComesBefore. CellId{0} and
// FacePieceId{0} are never valid ids (plan decision, decision 23), so "zero when unused" is unambiguous.
struct CellId { std::uint64_t value; };          // NEW. Packs size level and CellCoord.
struct CellBlockId { std::uint64_t value; };     // NEW. Packs size level and the block's coordinate.
struct FacePieceId { std::uint64_t value; };     // NEW. Packs axis, size level and the low-corner coordinate.

// Positions in the arrays of ONE grid version. Never store one across a version change.
struct CellBlockIndex { std::uint32_t value; };  // NEW.
struct CellPlaceIndex { std::uint32_t value; };  // NEW. block * 64 + (z * 4 + y) * 4 + x.
struct FacePlaceIndex { std::uint32_t value; };  // NEW. block * 240 + axis * 80 + local (normal axis has 5).
inline constexpr std::uint32_t kNoIndex = 0xFFFFFFFF;  // NEW. "No block / cell / face here".

struct GridVersion { std::uint64_t value; };     // NEW. Grows by one each time the set of active cells changes.
                                                 // 0 means "bound to no grid"; a built grid has 1 or more.

CellId ToCellId(SizeLevel size_level, CellCoord cell_coord);
SizeLevel ToSizeLevel(CellId cell);
CellCoord ToCellCoord(CellId cell);
CellBlockId ToCellBlockId(CellId cell);          // The block that stores this cell.
FacePieceId ToFacePieceId(Axis axis, SizeLevel size_level, CellCoord cell_high_coord);
CellId ComputeBigCell(CellId small_cell);                    // Coordinates halved, level minus one.
CellId ComputeSmallCell(CellId big_cell, int cell_eighth);   // cell_eighth = x + 2 * y + 4 * z, 0..7.
int ComputeCellEighth(CellId small_cell);                    // Which of the 8 sibling cells it is.
// Fixed order used everywhere: low corner in smallest-cell units by z, then y, then x, then bigger first.
bool ComesBefore(CellBlockId first, CellBlockId second);     // NEW verb-less predicate (decision 10).
bool ComesBefore(CellId first, CellId second);               // NEW (plan revision 2026-11-19). The same order for cells.

// Moved here from cell_block_range.h (plan decision, decision 23): a plain struct that shared CPU-and-GPU headers
// of other modules take on either processor. cell_block_range.h keeps the CPU-only functions that build ranges.
struct CellBlockRange { std::uint32_t first_block; std::uint32_t block_count; };  // Run in fixed order.
```

### 3.2 `grid_tables.h` (S): what code on either processor reads

```cpp
struct GridFrame {                 // NEW.
  Vec3 origin;                     // m, world. Low corner of the cell with coordinate (0, 0, 0).
  double biggest_cell_width;       // m, positive. Width at size level 0.
  std::uint8_t size_level_count;   // 1..16. Level size_level_count - 1 is the smallest.
};

enum class CellState : std::uint8_t {  // NEW. One per cell place.
  kAbsent,            // No cell here: outside the stored part of space.
  kInsideBiggerCell,  // Covered by an active cell one level bigger.
  kActive,            // DECIDED word: a cell that exists and is not split. Carries values.
  kSplit,             // Replaced by its 8 small cells, which all exist.
};

enum class FacePieceKind : std::uint8_t {  // NEW. One per face place.
  kNotAPiece,       // Unused place: no flow is counted here.
  kBetweenCells,    // An active cell on each side; they differ by at most one size level.
  kGridEdgeLow,     // Active cell on the high side only. What lies beyond is not the grid's to say.
  kGridEdgeHigh,    // Active cell on the low side only.
  kOutsidePiece,    // GridPiece only: a kBetweenCells piece of the whole grid whose other cell's block was not
                    // loaded with this piece (5.8).
};

// Raw arrays of one grid version on one processor. Copy by value into GPU functions. Not owning: they point
// into the GridBuffers the grid owns, whose arrays have room for more blocks than block_count.
struct GridTables {                               // NEW.
  GridFrame frame;
  GridVersion version;
  std::uint32_t block_count;                      // Blocks of this version. Every array below is read up to it.
  const CellBlockId* block_ids;                   // [block_count], in the fixed order.
  const std::uint32_t* same_level_blocks;         // [block_count * 27], (dz, dy, dx) order, kNoIndex if none.
  const std::uint32_t* bigger_block;              // [block_count], kNoIndex at level 0.
  const std::uint32_t* smaller_blocks;            // [block_count * 8], by cell_eighth of the block.
  const CellState* cell_states;                   // [block_count * 64].
  const FacePieceKind* face_piece_kinds;          // [block_count * 240].
};

double ComputeCellWidth(GridFrame frame, SizeLevel size_level);                 // m. GridFrame by value in shared functions.
Vec3 ComputeCellCenter(const GridTables& tables, CellPlaceIndex cell);          // m, world
Vec3 ComputeFacePosition(const GridTables& tables, FacePlaceIndex face);        // m, world, piece centre
double ComputeFaceArea(const GridTables& tables, FacePlaceIndex face);          // m2, whole piece, no solid
Axis ToAxis(FacePlaceIndex face);            // The piece's normal is +axis; values on it are positive along +axis.
CellId ToCellId(const GridTables& tables, CellPlaceIndex cell);
FacePieceId ToFacePieceId(const GridTables& tables, FacePlaceIndex face);
CellPlaceIndex FindCellLow(const GridTables& tables, FacePlaceIndex face);      // kNoIndex on kGridEdgeLow
CellPlaceIndex FindCellHigh(const GridTables& tables, FacePlaceIndex face);     // kNoIndex on kGridEdgeHigh
double ComputeCellCenterDistance(const GridTables& tables, FacePlaceIndex face);  // m; half a width at an edge
CellBlockIndex FindCellBlock(const GridTables& tables, CellBlockId block);      // Binary search. kNoIndex if none.
CellPlaceIndex FindCellPlace(const GridTables& tables, CellId cell);            // kNoIndex if absent.
FacePlaceIndex FindFacePlace(const GridTables& tables, FacePieceId piece);      // kNoIndex if not a piece now.
// The active cell that contains a position. Low sides belong to the cell, high sides do not.
CellPlaceIndex FindCellAt(const GridTables& tables, Vec3 position);                       // Binary search.
// Two moves from `near` through the neighbour tables, then the binary search: always the answer of the form above.
// Either form: a coordinate that is not finite or outside the id range is kNoIndex before any integer cast.
CellPlaceIndex FindCellAt(const GridTables& tables, Vec3 position, CellBlockIndex near);

struct TouchingCells { CellPlaceIndex cells[56]; int count; };   // NEW. DECIDED word `touching`.
// Every active cell that shares a face, an edge or a corner with `cell`, in ascending index order.
TouchingCells FindTouchingCells(const GridTables& tables, CellPlaceIndex cell);
struct CellFacePieces { FacePlaceIndex faces[24]; int count; };  // NEW. Up to 4 pieces on each of 6 sides.
CellFacePieces FindFacePieces(const GridTables& tables, CellPlaceIndex cell);  // Ascending index order.
```

### 3.3 `grid_error.h` (S) and `staggered_grid.h`: build, own, check

`GridError` and `GridFailure` are in the shared `grid_error.h` (plan decision: shared `FindNearbyFaces` reports
through `GridError*`); `ToMessage` and `ToName` are in the CPU-only `grid_error_message.h/.cc`, as contact is built
(`contact_error.h`, `contact_error_message.h`); the rest is in `staggered_grid.h`. A `BufferError` from `base/`
maps as follows: `kOutOfMemory` and `kRunMemoryTooSmall` (A58) to `kOutOfMemory`, `kRunOnCallFailed` to `kRunOnCallFailed`, `kSizeMismatch` to
`kInternalFault` (A106, 2026-11-20; it was `kBadInput`: the grid checks every caller's sizes itself, so a size error
from `base/` inside a grid operation is the grid's own fault) (A30: `kRunOnMismatch` no longer exists; buffers on two processors are a precondition).

```cpp
enum class GridError : std::uint8_t {  // NEW. One enum for the module.
  kNone,                  // No error. For shared functions that report through a GridError*.
  kBadInput,              // A frame value is not finite, not positive, or out of range; a version of 0; a
                          // span of the wrong size; a block list that is not ascending or out of range.
  kCoordOutOfRange,       // A cell coordinate does not fit kMaxCoord at its size level.
  kCellsOverlap,          // The same cell twice, or a cell and a bigger cell that contains it.
  kSiblingCellMissing,    // A big cell is split but one of its 8 small cells is absent.
  kSizeStepRuleBroken,    // Two touching cells differ by more than one size level.
  kTooManyCellBlocks,     // More blocks than the buffers or the values have room for (max_cell_blocks). Names no
                          // cell (`cell` zero) when a grid or piece does not fit its buffers (A17, decision 18).
  kRunOnCallFailed,       // CheckCanRunOn failed for the chosen RunOn, or a call to that processor failed.
                          // Nothing falls back to the CPU.
  kWrongGridVersion,      // A buffer or index of another grid version was passed, or values bound to no grid.
  kPositionOutsideGrid,   // No active cell contains the position.
  kTooManyNearbyFaces,    // More than kMaxNearbyFaces candidates. A guard: no valid grid reaches it (5.4).
  kInternalFault,         // NEW (A106). A state no input can produce: a size error from base/ inside a grid
                          // operation, arrays that do not match the estimate's entries, a table or status record
                          // that contradicts what the operation just wrote. Checked in every build; never
                          // kBadInput, never a debug-only assertion. A fault in FluidiTy, not in the scene: the
                          // message asks the user to report it with the scene. Stable name `internal_fault`.
  kOutOfMemory,           // Value 12 of 13. Changed (reconcile item 31): moved to the last place. Only the builders: the run's
                          // RunMemory (BufferError::kRunMemoryTooSmall) or the processor could not hold a buffer.
};
struct GridFailure {              // NEW. DECIDED shape (A2).
  GridError error = GridError::kNone;
  CellId cell{};                  // First failing cell in fixed order. Zero when unused.
  CellId other_cell{};            // The second cell of an overlap or a size-step break. Zero when unused.
  std::uint64_t magnitude = 0;    // NEW (A58, reconcile item 31). kOutOfMemory: bytes asked. Zero otherwise.
};
// The error value of every std::expected in this module is GridFailure (A2).
std::string_view ToMessage(GridError error);
// NEW. Stable code: the enumerator's name without `k`, in lower snake case ("too_many_cell_blocks"). The CLI
// and the add-on use it.
std::string_view ToName(GridError error);

// NEW (A17). Largest max_cell_blocks: every cell-place count (64 * max_cell_blocks) then fits the std::uint32_t
// orders and prefix sums of base/.
inline constexpr std::uint32_t kMaxCellBlocksLimit = (1u << 26) - 1;
struct GridSettings {             // DECIDED suffix.
  // Room, 1..kMaxCellBlocksLimit. Changed (reconcile, decision 28): no default room; the default 0 is kBadInput in
  // BuildGridBuffers and BuildGridBuilderBuffers (the test-only BuildTestGrid with 0 sizes its buffers to the list's
  // blocks, A31). A run's room comes from the planner (step::BuildRunPlan, 5.8). No compile-time room is an
  // allocation size. (The design's 1u << 22 is withdrawn.)
  std::uint32_t max_cell_blocks = 0;
  RunOn run_on = RunOn::kCpu;     // The caller's choice (A12). Never changed silently.
};
struct GridInputs {               // DECIDED suffix. NEW form (A17): the list in CPU memory or in a buffer.
  GridFrame frame;
  std::span<const CellId> active_cells;                // A list in CPU memory, any order.
  const Buffer<CellId>* active_cell_buffer = nullptr;  // Or a list in a buffer, any order.
  std::uint32_t active_cell_buffer_count = 0;          // Entries of *active_cell_buffer used.
  GridVersion version;                                 // 1 or more.
};
// When active_cell_buffer is set it is the list and active_cells must be empty; otherwise active_cells is the
// list (the only description of the grid; the caller passes previous version + 1). Requires (A30, debug
// assertion): the list is on the GridBuffers' processor (for the one-argument forms: in CPU memory), except an empty
// list, which has no processor. active_cell_buffer_count above the buffer's size is kBadInput. An empty list is a valid (empty) grid.
// So a GPU run passes a Buffer<CellId> on the GPU, a CPU run either form, tests and tools a span.
struct GridCounts {               // NEW. Everything the sizing estimate needs from the grid.
  std::uint32_t cell_blocks;
  std::uint64_t active_cells_per_size_level[kMaxSizeLevels];
  std::uint64_t active_cells;
  std::uint64_t face_pieces;      // Places whose kind is not kNotAPiece.
};

// NEW. The arrays one grid version is built into, with room for max_cell_blocks() blocks: the rows of
// BuildGridMemoryEstimate (3.6) that are not "builder buffers:": the tables and the piece map, plus on the GPU the
// build status record, all on the GPU and nothing on the CPU for a GPU grid (A17). It uses the GridBuilderBuffers
// lent to it by BuildGridBuffers (reconcile; A31: always lent, it never owns builder buffers). Move-only. It belongs
// to no grid (BuildGrid, BuildGridPiece move it into one; TakeBuffers gives it back), so its contents mean nothing.
// A moved-from GridBuffers has max_cell_blocks() == 0.
class GridBuffers {
 public:
  std::uint32_t max_cell_blocks() const;
  RunOn run_on() const;
};

class StaggeredGrid {             // DECIDED name (decision 13). Move-only. Owns its GridBuffers.
 public:
  GridVersion version() const;             // Kept on the CPU for every grid (from the status record on the GPU).
  const GridCounts& counts() const;        // Kept on the CPU for every grid.
  CellBlockRange all_blocks() const;       // {0, counts().cell_blocks}.
  // Changed (A17): tables(run_on()) only. tables(kCpu) on a GPU grid and tables(kGpu) on a CPU grid return
  // version 0 and null arrays (the design's "kCpu always works" no longer holds; ReadGrid copies on request).
  GridTables tables(RunOn run_on) const;
  RunOn run_on() const;                    // NEW. The buffers'.
  std::uint32_t max_cell_blocks() const;   // NEW. The buffers' room, at least counts().cell_blocks.
};

// Counts without building. Requires a CPU list (A30, debug assertion). Same checks as BuildGrid except the block
// limit, so a planner can count beyond any room: the stated reason this form stays beside the buffers form (A31).
// It builds no table (plan revision 2026-11-19): it sorts a CPU copy of the ids, counts face pieces by id lookups
// (per active cell and side: is the place beside it active, inside a bigger cell, split or absent), appends the
// split cells, sorts by block and counts the distinct block ids. One allocation of 8 B per active or split cell
// (at most 9.2 B per active cell); two sorts and four passes. Tables sized for the worst case would cost about
// 1.1 kB per active cell, and their size is the count being asked for.
std::expected<GridCounts, GridFailure> ComputeGridCounts(const GridInputs& inputs);
// NEW (A17). The same on buffers.run_on(), allocating nothing: steps 1 to 8 of 5.1 (face pieces counted from built
// kinds), so it fails kTooManyCellBlocks (`cell` zero) when the grid does not fit the buffers. The buffers hold no
// grid before or after; their contents become unspecified.
std::expected<GridCounts, GridFailure> ComputeGridCounts(const GridInputs& inputs, GridBuffers& buffers);

// NEW (reconcile, decision 29). The temporary arrays of grid builds and checks on one processor for grids of at
// most max_cell_blocks blocks: the "grid: builder buffers: ..." entries of BuildGridMemoryEstimate (cell ids, sort
// keys, two orders of 64 * max_cell_blocks + 1 entries and an OrderBuffers for 64 * max_cell_blocks + 1 items; the
// entry "builder buffers: order buffers" counts that OrderBuffers, reconcile item 28, as one entry, A58). Holds nothing
// between calls. Owned by the caller and lent to every GridBuffers built with it; those buffers must not outlive
// it, it must not be moved while they exist, and one build or check uses it at a time. No operation on a built
// grid reads the builder buffers, so buffers holding two live grids may share one. Move-only.
class GridBuilderBuffers {
 public:
  std::uint32_t max_cell_blocks() const;
  RunOn run_on() const;
};
// NEW (reconcile). kBadInput: max_cell_blocks 0 or above kMaxCellBlocksLimit; kOutOfMemory; kRunOnCallFailed.
// Changed (A58): every array is a piece of run_memory, the RunMemory of the processor the arrays live on (a CPU
// grid in a GPU run gets the run's cpu_run_memory). Requires: run_on == run_memory.run_on() (debug assertion).
std::expected<GridBuilderBuffers, GridFailure> BuildGridBuilderBuffers(RunOn run_on, std::uint32_t max_cell_blocks,
                                                                       RunMemory& run_memory);
// NEW. The one allocation of a grid's own arrays: the non-builder rows of 3.6 for settings.max_cell_blocks blocks
// on settings.run_on (5.8), each allocated where its MemoryEstimateEntry::where says. The buffers allocate no
// builder buffers and use `builder_buffers` for every BuildGrid, BuildGridPiece and buffers form of
// ComputeGridCounts and CheckSizeStepRule. Requires: builder_buffers.run_on() == settings.run_on (debug assertion,
// A30). Fails kBadInput (max_cell_blocks 0, the default, or above kMaxCellBlocksLimit; builder_buffers.max_cell_blocks()
// < settings.max_cell_blocks; nothing allocated), kRunOnCallFailed, kOutOfMemory. Changed (A31): the one form; the
// form that built its own builder buffers is removed. Changed (A58): every array is a piece of run_memory;
// Requires: settings.run_on == run_memory.run_on().
std::expected<GridBuffers, GridFailure> BuildGridBuffers(const GridSettings& settings,
                                                         GridBuilderBuffers& builder_buffers,
                                                         RunMemory& run_memory);
// NEW (reconcile). How many GridBuilderBuffers the builder buffers entries of BuildGridMemoryEstimate (3.6) count.
// Declared in staggered_grid.h (reconcile item 20); grid_memory_estimate.h includes it.
enum class BuilderBuffersSharing : std::uint8_t {
  kOnePerGrid,      // One GridBuilderBuffers for each of the grid_count buffers.
  kOneForAllGrids,  // One GridBuilderBuffers lent to all of them.
};
// NEW form. Builds the grid of `inputs` into `buffers` on buffers.run_on(): on the CPU by 5.1 item 8, on the GPU
// by the parallel build of 5.1 item 9 (A17), which checks, sorts and builds every table on the GPU and reads
// back only the GridBuildStatus record. The tables, counts and failures of both are equal byte for byte. Allocates
// nothing. On success the buffers move into the returned grid and `buffers` is left moved-from. On failure
// `buffers` keeps its arrays and can be used again; no grid is returned. Fails kBadInput, kCoordOutOfRange,
// kCellsOverlap, kSiblingCellMissing, kSizeStepRuleBroken (the first failure by the failure key of 5.1 item 9,
// on both processors; where two cells are involved both are named), kTooManyCellBlocks (active plus implied split
// cells above 64 * buffers.max_cell_blocks(), or blocks above buffers.max_cell_blocks(); `cell` zero),
// kRunOnCallFailed. An empty list builds an empty grid (zero blocks).
std::expected<StaggeredGrid, GridFailure> BuildGrid(const GridInputs& inputs, GridBuffers& buffers);
// NEW. Takes back the buffers of a grid that is no longer needed, for the next BuildGrid or BuildGridPiece.
// Allocates nothing. Every GridTables taken from `grid` is invalid afterwards. Changed (A29): was ReleaseBuffers.
GridBuffers TakeBuffers(StaggeredGrid&& grid);
// Removed (A31): the allocating BuildGrid(inputs, settings). Tests use the test-only BuildTestGrid of
// grid_test_support.h (grid plan Task 9), which builds a GridBuilderBuffers, a GridBuffers and the grid, and with
// settings.max_cell_blocks 0 sizes them to ComputeGridCounts(inputs).cell_blocks, at least 1; a tool builds the
// two buffers itself.
// Id checks, overlaps and the size-step rule (steps 1, 2 and 4 of 5.1), not siblings: cell resize checks lists
// while it splits (plan decision, decision 19). The span form sorts a CPU copy of the ids (8 B per cell); it is
// kept beside the buffers form because cell resize checks CPU lists where no GridBuffers exist (A31, stated reason).
std::expected<void, GridFailure> CheckSizeStepRule(GridFrame frame, std::span<const CellId> active_cells);
// NEW (A17). The same on buffers.run_on(), allocating nothing; the buffers hold no grid before or after. Also fails
// kTooManyCellBlocks (`cell` zero) when the list has more than 64 * buffers.max_cell_blocks() cells.
std::expected<void, GridFailure> CheckSizeStepRule(const GridInputs& inputs, GridBuffers& buffers);

// NEW (A17). Copies the used part of grid's five tables into cpu_buffers (456 B per block from the GPU) and returns
// a grid on the CPU with the same version and counts. The one explicit copy of a GPU grid: for tests, tools and a
// caller that accepts the transfer. Allocates nothing. Requires: cpu_buffers.run_on() == kCpu (debug assertion,
// A30). Fails kTooManyCellBlocks (room below grid.counts().cell_blocks), kRunOnCallFailed. On failure cpu_buffers keeps its
// arrays.
std::expected<StaggeredGrid, GridFailure> ReadGrid(const StaggeredGrid& grid, GridBuffers& cpu_buffers);
```

`grid_build_status.h` (S, private; written by GPU kernels, read back with one `ReadBuffer`; its `sizeof` is a
memory estimate entry, so it is stated here):

```cpp
inline constexpr std::uint64_t kNoFailureKey = ~std::uint64_t{0};
struct GridBuildStatus {        // The only thing a grid operation on the GPU returns to the CPU.
  std::uint64_t failure_key;    // kNoFailureKey, or (step << 56) | (position << 8) | detail (5.1 item 9).
  GridError error;              // kNone on success; else the failure's error.
  CellId cell;                  // GridFailure::cell; zero when unused.
  CellId other_cell;            // GridFailure::other_cell; zero when unused.
  std::uint32_t cell_blocks;    // Blocks built (grid or piece); later kernels read it as their count.
  std::uint32_t split_cells;    // Split cells added at step P7; later kernels read it. The CPU ignores it.
  std::uint32_t result_count;   // FindBorderBlocks: entries found. Otherwise 0.
  std::uint64_t active_cells;
  std::uint64_t active_cells_per_size_level[kMaxSizeLevels];
  std::uint64_t face_pieces;    // Places whose kind is not kNotAPiece.
};                              // Trivially copyable (static_assert).
```

A caller that keeps a committed and a try grid (the step) builds two grid buffers once, at the limit, on its run's
processor, and moves them between grids with `BuildGrid` and `TakeBuffers`; raising the limit
(`step::ApplyLimits`) builds new grid buffers and rebuilds the committed grid into one with the same `GridInputs`,
which gives the same tables. Reconcile (decision 29): the step's simulation state owns one `GridBuilderBuffers` (built
first, freed last, rebuilt by `ApplyLimits` with the grid buffers) lent to the committed and free `GridBuffers` and,
in a batched run, the loaded try's buffers; the batched GPU run's CPU grid copy pair shares one CPU `GridBuilderBuffers`;
cell resize's two CPU grids in a one-piece GPU run share one CPU `GridBuilderBuffers`. Every other caller (pressure's
`SolveBatches`, tests and tools) owns a `GridBuilderBuffers` for its buffers (A31: no form builds its own). In a GPU run the step calls `BuildGrid` on the GPU each time the cell set changes:
`ChooseActiveCells` hands it a `Buffer<CellId>` on the GPU and the entry count as a CPU number (from the step's own
stage status), and takes the new grid's counts from `grid.counts()` (A17).

### 3.4 `cell_block_range.h` and `grid_piece.h`: the unit every stage works on

`CellBlockRange` is declared in `grid_ids.h` (3.1). `cell_block_range.h` is CPU only and includes `grid_piece.h`,
so the spelling `cell_block_range.h` still reaches `GridPiece` (plan decision, decision 23: `GridPiece` owns
buffers and gets its own header).

```cpp
struct RangeWithBorder {          // NEW. DECIDED word `border_cells`.
  CellBlockRange range;                         // A stage writes only values stored in these blocks.
  std::vector<CellBlockIndex> border_blocks;    // FindBorderBlocks(grid, range's blocks, width). Ascending.
};
// BorderWidth::kTouchingCells. Precondition (A17): grid.run_on() == RunOn::kCpu (it has no failure value; checked
// by a debug assertion).
RangeWithBorder BuildRangeWithBorder(const StaggeredGrid& grid, CellBlockRange range);
// Cuts the grid into consecutive ranges of at most max_blocks_per_range blocks. Reads only counts(): either processor.
std::vector<CellBlockRange> BuildRanges(const StaggeredGrid& grid, std::uint32_t max_blocks_per_range);

// NEW. The blocks outside `blocks` that a stage writing values stored in `blocks` may read, at `width` (3.7
// BorderWidth, 5.5), in ascending index, each once. `blocks` is ascending with no repeats; it need not be a run,
// so borders stack: the border of a stage before is FindBorderBlocks(grid, blocks + border found so far, its
// width). CPU grid only (A17; a `Requires:` precondition checked by a debug assertion, A30, reconcile item 32;
// it was kBadInput). Fails kBadInput when `blocks` is not ascending or holds an
// index >= the grid's block count. Allocates the returned list; for planning and tests.
std::expected<std::vector<CellBlockIndex>, GridFailure> FindBorderBlocks(const StaggeredGrid& grid,
                                                                         std::span<const CellBlockIndex> blocks,
                                                                         BorderWidth width);
// NEW. Same result, allocating nothing: writes the first border_blocks.size() border blocks and returns how many
// there are in all. A count above border_blocks.size() means the span was too small; nothing beyond it is
// written. Requires a CPU grid (A17; a `Requires:` precondition checked by a debug assertion, A30).
std::expected<std::uint32_t, GridFailure> FindBorderBlocks(const StaggeredGrid& grid,
                                                           std::span<const CellBlockIndex> blocks,
                                                           BorderWidth width,
                                                           std::span<CellBlockIndex> border_blocks);

// NEW (A17). Border search on either processor. Room to search the border of any block set of a grid with at most
// max_cell_blocks blocks. Move-only.
class BorderSearchBuffers {
 public:
  std::uint32_t max_cell_blocks() const;
  RunOn run_on() const;
};
// The one allocation of border search. Fails kBadInput (0 or above kMaxCellBlocksLimit), kRunOnCallFailed,
// kOutOfMemory. Changed (A58): pieces of run_memory; Requires: run_on == run_memory.run_on().
std::expected<BorderSearchBuffers, GridFailure> BuildBorderSearchBuffers(RunOn run_on,
                                                                         std::uint32_t max_cell_blocks,
                                                                         RunMemory& run_memory);
std::vector<MemoryEstimateEntry> BuildBorderSearchMemoryEstimate(RunOn run_on,
                                                                 std::uint32_t max_cell_blocks);  // Changed (A58).
// NEW (A17). Every block outside `range` that is in border_so_far or in the border, at `width`, of range plus
// border_so_far: one more layer of a stacked border, merged with the layers before. Ascending, each once. Writes
// the first border_blocks.size() entries and returns how many there are in all. Allocates nothing. Requires: every
// buffer on grid.run_on() (debug assertion, A30). Fails kBadInput (border_so_far not ascending, holding a block of `range` or an index >= the grid's block count; border_blocks is border_so_far;
// border_so_far_count above its size), kTooManyCellBlocks (search_buffers too small for the grid),
// kRunOnCallFailed. On the GPU only the count comes back. A stacked border is one call per layer with two output
// buffers used in turn; no merge is needed between layers.
std::expected<std::uint32_t, GridFailure> FindBorderBlocks(const StaggeredGrid& grid, CellBlockRange range,
                                                           const Buffer<CellBlockIndex>& border_so_far,
                                                           std::uint32_t border_so_far_count, BorderWidth width,
                                                           BorderSearchBuffers& search_buffers,
                                                           Buffer<CellBlockIndex>& border_blocks);
```

```cpp
// grid_piece.h. NEW shape. One range of a whole grid with its border, for the batched run of the cell resize and
// batches design (5.8) and for pressure's solve batches. Move-only. Owns its GridBuffers. Its blocks are the
// loaded blocks in the whole grid's fixed order (never "range blocks first"), so the range is a run inside it and
// every sum over source blocks visits them as the whole grid does.
class GridPiece {
 public:
  const StaggeredGrid& grid() const;           // The loaded blocks. version() is the whole grid's version.
  CellBlockRange range() const;                // The range, as a run of grid()'s blocks.
  CellBlockRange range_in_whole_grid() const;  // The same blocks, as indexes of the whole grid.
  // [grid().counts().cell_blocks], ascending. The map of a CPU piece; empty for a GPU piece (A17).
  std::span<const CellBlockIndex> block_in_whole_grid() const;
  // NEW (A17). The piece map on grid().run_on(), [grid().counts().cell_blocks] entries used, ascending.
  const Buffer<CellBlockIndex>& block_in_whole_grid_buffer() const;
};
// NEW form. Builds the piece of `whole` made of `range` and `border_blocks` into `buffers`. Requires a CPU whole
// grid (A17; a `Requires:` precondition checked by a debug assertion, A30). Allocates nothing; same ownership rule as BuildGrid. `border_blocks` is
// ascending, outside `range`, below the whole grid's block count (a FindBorderBlocks result, stacked or not).
// Fails kBadInput, kTooManyCellBlocks (range plus border exceed buffers.max_cell_blocks(); `cell` zero),
// kRunOnCallFailed.
std::expected<GridPiece, GridFailure> BuildGridPiece(const StaggeredGrid& whole, CellBlockRange range,
                                                     std::span<const CellBlockIndex> border_blocks,
                                                     GridBuffers& buffers);
// NEW (A17). Same as the span form, on whole.run_on(), with the border in a buffer on whole.run_on() (a stacked
// FindBorderBlocks result). On the GPU the piece's tables are built on the GPU from the whole grid's, equal to the
// CPU piece byte for byte. Cell resize and batches, and pressure's solve batches, use this form in a GPU run.
std::expected<GridPiece, GridFailure> BuildGridPiece(const StaggeredGrid& whole, CellBlockRange range,
                                                     const Buffer<CellBlockIndex>& border_blocks,
                                                     std::uint32_t border_block_count, GridBuffers& buffers);
GridBuffers TakeBuffers(GridPiece&& piece);  // NEW. As for a grid. Changed (A29): was ReleaseBuffers.
```

### 3.5 `weight_curve.h` (S) and `nearby_face_weights.h` (S)

```cpp
struct WeightCurveValue { double weight; double slope; };   // NEW. No unit; slope per cell width.
// DECIDED word `weight_curve`. Papers: quadratic B-spline. Zero at and beyond 1.5 cell widths.
WeightCurveValue ComputeWeightCurve(double distance_in_cell_widths);
inline constexpr double kWeightCurveReach = 1.5;            // NEW. Cell widths.
inline constexpr int kMaxNearbyFaces = 216;                 // Per axis. Changed (owner decision 2026-11-19; was 76):
                                                            // no valid grid passes 212, proof in 5.4.
inline constexpr double kMinWeightSum = 0.1;                // NEW. Ported from the old paper's mask rule.
inline constexpr double kMinPivotShare = 1e-12;             // NEW. Ported: graded_mac_transfer_3d.cc:19.

struct NearbyFaceList {           // NEW. The candidates for one axis, before any is left out.
  CellPlaceIndex cell;            // The active cell that contains the position.
  double cell_width;              // m. All distances below are measured in this width.
  Axis axis;
  int count;
  FacePlaceIndex faces[kMaxNearbyFaces];   // Ascending index order.
};
struct NearbyFaceMask { std::uint64_t bits[4]; };   // Bit i (of word i / 64) set = face i of the list is kept.
                                                    // Changed 2026-11-19: four words, for 216 faces. 32 B.

enum class NearbyFaceFit : std::uint8_t {   // NEW.
  kFull,      // Weights, mean and an invertible spread: a fitted_velocity_change can be computed.
  kMeanOnly,  // Weights are valid but the kept faces are too few or too flat to fit a change.
  kNone,      // No kept face has weight. The particle cannot read or write the grid on this axis.
};
struct NearbyFaceWeights {        // DECIDED name. One axis, one particle position. 1,152 bytes (the list is 888), never stored.
  NearbyFaceList list;
  NearbyFaceMask kept;
  Vec3 position;                  // m, world.
  double weight_sum;              // No unit. Sum of kept raw weights; 1 when all 27 same-size faces are kept.
  Vec3 mean_offset;               // m. Weighted mean of (face position - position). Zero when complete.
  Vec3 raw_weight_change_sum;     // 1/m. Zero when complete.
  Mat3 weighted_spread;           // m2, symmetric. DECIDED name. cell_width^2 / 4 * identity when complete.
  Mat3 inverse_weighted_spread;   // 1/m2. Meaningful only when fit == kFull.
  NearbyFaceFit fit;
  bool is_complete;               // All faces kept, all same size as the cell, 27 of them (true at every position of
                                  // a same-size neighbourhood: the candidates are chosen in integers, 5.4).
};
struct FaceWeight {               // DECIDED name.
  FacePlaceIndex face;
  double weight;                  // No unit, never negative; 0 for a face exactly 1.5 widths away. Kept weights of
                                  // one axis sum to 1.
  Vec3 offset;                    // m. Face position minus particle position.
  Vec3 offset_from_mean;          // m. offset - mean_offset. Use this one in fits and in writes.
  Vec3 weight_change;             // 1/m. Derivative of `weight` with respect to the particle position. NEW.
};

// Fails: kPositionOutsideGrid, kTooManyNearbyFaces (returned in *error; the list then has count 0).
NearbyFaceList FindNearbyFaces(const GridTables& tables, Vec3 position, Axis axis, CellBlockIndex near,
                               GridError* error);
NearbyFaceMask ToMaskOfAll(const NearbyFaceList& list);
NearbyFaceWeights ComputeNearbyFaceWeights(const GridTables& tables, const NearbyFaceList& list,
                                           Vec3 position, NearbyFaceMask kept);
// position_in_list must be a kept position. Call in ascending order for a fixed order of sums.
FaceWeight ComputeFaceWeight(const GridTables& tables, const NearbyFaceWeights& weights, int position_in_list);
```

### 3.6 `grid_values.h` and `grid_memory_estimate.h`

```cpp
// Changed (plan revision 2026-11-19): ONE class template, GridValues<Value, places_per_block>, with Value a
// BufferValue (base/buffer.h); CellValues, FaceValues, CornerValues and CellBlockValues are alias templates of it
// for 64, 240, 125 and 1 places per block, so they are four distinct types with one implementation. The members
// below are GridValues' members, shown for CellValues; each Build...Values is a one-line form of one builder, and
// ApplyGridVersion, ReadValues and WriteValues are each written once over places_per_block.
template <typename Value>  // Concept: BufferValue.
class CellValues {         // NEW. One value per cell place; meaningful where the state is kActive. Move-only.
 public:
  GridVersion grid_version() const;       // Of the grid it is bound to. GridVersion{0}: bound to none.
  std::uint32_t block_count() const;      // NEW. Bound extent: places [0, block_count() * 64) are meaningful.
  std::uint32_t max_cell_blocks() const;  // NEW. Room, fixed when built.
  RunOn run_on() const;
  Value* data();           // Valid on run_on() only. Indexed by CellPlaceIndex of the bound grid.
  const Value* data() const;
  Buffer<Value>& buffer();              // The buffer (base/buffer.h), size max_cell_blocks() * 64. The ordered
  const Buffer<Value>& buffer() const;  // sums of base/ take it with count block_count() * 64.
};
// Same members. Places per block: 240 (FacePlaceIndex), 125 (CornerPlaceIndex), 1 (CellBlockIndex).
template <typename Value> class FaceValues;       // NEW.
template <typename Value> class CornerValues;     // NEW.
template <typename Value> class CellBlockValues;  // NEW.

// NEW forms. Values with room for max_cell_blocks blocks on run_on, bound to no grid. The only allocation of a
// value buffer. Fail kBadInput (max_cell_blocks is 0), kRunOnCallFailed, kOutOfMemory. Changed (A58): both forms
// below take run_memory last and build the one buffer as one piece of it; Requires: run_on == run_memory.run_on().
template <typename Value>
std::expected<CellValues<Value>, GridFailure> BuildCellValues(RunOn run_on, std::uint32_t max_cell_blocks,
                                                              RunMemory& run_memory);
template <typename Value>
std::expected<FaceValues<Value>, GridFailure> BuildFaceValues(RunOn run_on, std::uint32_t max_cell_blocks,
                                                              RunMemory& run_memory);
template <typename Value>
std::expected<CornerValues<Value>, GridFailure> BuildCornerValues(RunOn run_on, std::uint32_t max_cell_blocks,
                                                              RunMemory& run_memory);
template <typename Value>
std::expected<CellBlockValues<Value>, GridFailure> BuildCellBlockValues(RunOn run_on, std::uint32_t max_cell_blocks,
                                                              RunMemory& run_memory);

// NEW. Binds `values` to `grid` (a whole grid or GridPiece::grid()): grid_version() and block_count() become
// the grid's. Contents become unspecified. Runs on the CPU for every grid: it reads only grid.version() and
// grid.counts().cell_blocks, which every grid keeps on the CPU, and records them; it allocates, launches and copies
// nothing (A17). Requires: values.run_on() == grid.run_on() (debug assertion; decision 32, A30, replacing the
// kBadInput of decision 22). Fails kTooManyCellBlocks when the grid has more blocks than values.max_cell_blocks();
// `values` is then unchanged.
template <typename Value>
std::expected<void, GridFailure> ApplyGridVersion(CellValues<Value>& values, const StaggeredGrid& grid);
template <typename Value>
std::expected<void, GridFailure> ApplyGridVersion(FaceValues<Value>& values, const StaggeredGrid& grid);
template <typename Value>
std::expected<void, GridFailure> ApplyGridVersion(CornerValues<Value>& values, const StaggeredGrid& grid);
template <typename Value>
std::expected<void, GridFailure> ApplyGridVersion(CellBlockValues<Value>& values, const StaggeredGrid& grid);

// Built for exactly grid.counts().cell_blocks on grid.run_on() and bound to `grid`. Allocates; for tests,
// tools and values that are built once.
template <typename Value>
std::expected<CellValues<Value>, GridFailure> BuildCellValues(const StaggeredGrid& grid, RunMemory& run_memory);
template <typename Value>
std::expected<FaceValues<Value>, GridFailure> BuildFaceValues(const StaggeredGrid& grid, RunMemory& run_memory);
template <typename Value>
std::expected<CornerValues<Value>, GridFailure> BuildCornerValues(const StaggeredGrid& grid, RunMemory& run_memory);
template <typename Value>
std::expected<CellBlockValues<Value>, GridFailure> BuildCellBlockValues(const StaggeredGrid& grid,
                                                                        RunMemory& run_memory);

// Copies between the bound part of a buffer and CPU memory. The only way values leave or enter the GPU. The span
// has exactly block_count() * places-per-block entries (else kBadInput). Fail kWrongGridVersion (bound to no
// grid), kRunOnCallFailed.
template <typename Value>
std::expected<void, GridFailure> ReadValues(const CellValues<Value>& values, std::span<Value> cpu_values);
template <typename Value>
std::expected<void, GridFailure> WriteValues(std::span<const Value> cpu_values, CellValues<Value>* values);
template <typename Value>
std::expected<void, GridFailure> ReadValues(const FaceValues<Value>& values, std::span<Value> cpu_values);
template <typename Value>
std::expected<void, GridFailure> WriteValues(std::span<const Value> cpu_values, FaceValues<Value>* values);
template <typename Value>
std::expected<void, GridFailure> ReadValues(const CornerValues<Value>& values, std::span<Value> cpu_values);
template <typename Value>
std::expected<void, GridFailure> WriteValues(std::span<const Value> cpu_values, CornerValues<Value>* values);
template <typename Value>
std::expected<void, GridFailure> ReadValues(const CellBlockValues<Value>& values, std::span<Value> cpu_values);
template <typename Value>
std::expected<void, GridFailure> WriteValues(std::span<const Value> cpu_values, CellBlockValues<Value>* values);
// kWrongGridVersion unless `version` is grid.version().
std::expected<void, GridFailure> CheckGridVersion(const StaggeredGrid& grid, GridVersion version);

// MemoryEstimateEntry (with its MemoryWhere where field), CountedPer, KeptFor, MemoryCounts and ComputeMemoryBytes are in
// base/memory_count.h (shared-interface.md section (a)); this design first declared them here.
// Sets cell_places = 64, face_places = 240, corner_places = 125 times cell_blocks.
MemoryCounts ToMemoryCounts(const GridCounts& counts, std::uint64_t particles);   // NEW.
// NEW (replaces the constant kGridMemoryEstimate). The entries of `grid_count` GridBuffers objects on run_on, each
// counted per cell block at the limit (MemoryCounts::cell_blocks = max_cell_blocks), each with copies =
// grid_count and its `where` (table below). The step passes 2 (committed and try grid), the batched run 1 at
// max_loaded_blocks. NEW form (reconcile, decision 29): with kOneForAllGrids (BuilderBuffersSharing, 3.3) the
// "builder buffers:" entries have copies = 1 and every other entry copies = grid_count; the default keeps every
// existing caller's numbers. The step passes (run_on, 2, kOneForAllGrids); the batched run's loaded buffers count
// (run_on, 1, kOneForAllGrids) without the "builder buffers:" entries (they borrow the step's GridBuilderBuffers).
// Changed (A58, reconcile ruling): takes max_cell_blocks; an array of a per-item plus a fixed size is one
// kGivenCount entry (bytes_each 1, given_count = its exact bytes at max_cell_blocks), so ComputeRunMemoryBytes
// equals the bytes handed out for any room. MemoryCounts::cell_blocks must equal max_cell_blocks.
std::vector<MemoryEstimateEntry> BuildGridMemoryEstimate(
    RunOn run_on, std::uint32_t grid_count, std::uint32_t max_cell_blocks,
    BuilderBuffersSharing builder_buffers_sharing = BuilderBuffersSharing::kOnePerGrid);
```

These four names are the only per-cell, per-face, per-corner and per-block buffer types of the code base. They
replace `FaceField`, `CellField`, `FacePieceValues` and `ParticleValues` (per-particle arrays use `Buffer`
directly) in the other designs. A `BufferError` from `base/` is mapped as in 3.3.
A function of another module that takes values and a grid checks `grid_version()` and `block_count()` against
the grid and fails with its own `kWrongGridVersion`. A piece has the whole grid's version, so values loaded for
one piece are bound again with `ApplyGridVersion` at every load.

`BuildGridMemoryEstimate` (`BuildGridBuffers` and `BuildGridBuilderBuffers` walk these same rows, the builder
buffers rows for the second, and allocate each on `run_on` or the CPU as its `where` says, so the estimate cannot drift; there is no other table of processors). Rows in this fixed order,
names prefixed `grid: `, sizes `sizeof` of the real types. Every row is `KeptFor::kWholeRun` with
`copies = grid_count` (with `kOneForAllGrids`, the `builder buffers:` rows `copies = 1`: they are the
`GridBuilderBuffers`). Changed (A17): a GPU grid keeps nothing on the CPU; the CPU copy of the tables is gone.

| Row | Counted per | Bytes each | Per block | Where |
|---|---|---|---|---|
| block ids | cell block | 8 | 8 | `kRunOn` |
| same-size-level neighbour blocks | cell block | 27 x 4 | 108 | `kRunOn` |
| bigger and smaller blocks | cell block | 9 x 4 | 36 | `kRunOn` |
| cell states | cell place | 1 | 64 | `kRunOn` |
| face piece kinds | face place | 1 | 240 | `kRunOn` |
| **grid tables total** | | | **456 B (7.1 B per cell place)** | `kRunOn` |
| *CPU grid:* builder buffers: cell ids | cell place | 8 | 512 | `kCpu` |
| *CPU grid:* piece map | cell block | 4 | 4 | `kCpu` |
| **one CPU grid's buffers** | | | **972 B: 456 `kRunOn`, 516 `kCpu`** | |
| *GPU grid:* builder buffers: cell ids (sorted active cells, then active and split cells) | cell place | 8 | 512 | `kRunOn` |
| *GPU grid:* builder buffers: sort keys | cell place | 8 | 512 | `kRunOn` |
| *GPU grid:* builder buffers: order (`64 * max_cell_blocks + 1` entries; changed A58: one entry, was a per-place row plus half of `prefix sum ends`) | given count `4 * (64 * max_cell_blocks + 1)` | 1 | 256 (+ 4 once) | `kRunOn` |
| *GPU grid:* builder buffers: second order (also the marks and their prefix sums; same form) | given count `4 * (64 * max_cell_blocks + 1)` | 1 | 256 (+ 4 once) | `kRunOn` |
| *GPU grid:* builder buffers: order buffers (the `OrderBuffers` for `64 * max_cell_blocks + 1` items; reconcile item 28; changed A58: one entry, was a per-place row plus `order buffers, fixed part`) | given count `bytes_per_item * (64 * max_cell_blocks + 1) + fixed_bytes` | 1 | 64 x `bytes_per_item` (+ `fixed_bytes + bytes_per_item` once) | `kRunOn` |
| *GPU grid:* piece map | cell block | 4 | 4 | `kRunOn` |
| *GPU grid:* build status | given count 1 | `sizeof(GridBuildStatus)` | | `kRunOn` |
| **one GPU grid's buffers with their own builder buffers** | | | **1,996 B + 64 x order bytes per item, plus `fixed_bytes` + `bytes_per_item` + 8 + `sizeof(GridBuildStatus)`; 0 `kCpu`** (reconcile: with `ComputeOrderBuffersSize(kGpu) = {6, 4096}`, 2,380 B per block + 4,096 + 6 + 8 + `sizeof(GridBuildStatus)`) | |
| *NEW (reconcile):* **GPU buffers (builder buffers lent)** | | | **460 B (456 tables + 4 piece map) + `sizeof(GridBuildStatus)`, per buffers** | `kRunOn` |
| *NEW (reconcile):* **GPU `GridBuilderBuffers`** | | | **1,920 B (1,536 + 64 x 6) + 4,096 + 6 + 8, once** | `kRunOn` |
| *NEW (reconcile):* **CPU buffers (builder buffers lent); CPU `GridBuilderBuffers`** | | | **456 `kRunOn` + 4 `kCpu` per buffers; 512 B `kCpu` once** (the CPU builder sorts with `std::sort`, no `OrderBuffers`) | |
| a `CellValues<double>` (other modules) | cell place | 8 | 512 | run_on |
| a `FaceValues<double>` (other modules) | face place | 8 | 1,920 | run_on |
| a `CornerValues<double>` (pressure, solids) | corner place | 8 | 1,000 | run_on |

**Two grids, one `GridBuilderBuffers` (NEW, reconcile; decision 30).** The step's committed and try grids on the GPU
cost `2 x 460 + 1,920 = 2,840` B per block plus 4,096 + 6 + 8 + `2 x sizeof(GridBuildStatus)` (was `2 x 2,380 = 4,760`
with builder buffers each). On a 16 GiB GPU, `allowed_bytes = 0.7 x 17,179,869,184 = 12.03e9` B; a quarter of it, 3.01e9 B, holds
about **1.06 M blocks** (about `2^20`, 67 M cell places) for tables and builder buffers; the withdrawn default `2^22` would
take 11.9e9 B. Amended 2026-11-15 (A58): the budget is now `min(0.9 x free, free - 536,870,912)`; with 16 GiB free
that is 15.46e9 B, a quarter of it 3.87e9 B, about 1.36 M blocks.

**Reservation (NEW, A58).** Every grid array is one piece of the caller's `RunMemory` and one estimate entry;
`ComputeRunMemoryBytes` rounds each entry up to 256 B, as the piece is rounded. An array of a per-item plus a fixed
size (the two GPU orders, the GPU `OrderBuffers`, the border marks and the border `OrderBuffers`) is one
`kGivenCount` entry of its exact bytes at `max_cell_blocks` (reconcile ruling on `plan-decisions.md` item 3), so the
reservation equals the bytes handed out for any room.

`BuildBorderSearchMemoryEstimate(run_on, max_cell_blocks)` (NEW, A17; `grid: ` prefix, `KeptFor::kWholeRun`,
`copies = 1`; changed A58: one entry per array, the ones with a fixed part `kGivenCount` of their exact bytes): `border marks`
(given count `4 * (max_cell_blocks + 1)`, 1 B; was a per-block row plus `border marks end`), `where = kRunOn`; for
`kGpu` also `border order buffers` (given count `bytes_per_item * (max_cell_blocks + 1) + fixed_bytes` of
`ComputeOrderBuffersSize(kGpu)`, 1 B; the `OrderBuffers` of `BorderSearchBuffers`, reconcile item 28; was a
per-block row plus `border order buffers, fixed part`) and `border status` (given count 1,
`sizeof(GridBuildStatus)`). CPU total: `4 * max_cell_blocks + 4`.

### 3.7 Added for other modules (NEW, from the reconciliation of 2026-11-14)

Each declaration below was asked for by another design's section 4 and belongs to the grid. All are shared
(S) unless marked, and all are answered from the existing tables: `GridTables` gains no array.

```cpp
// grid_ids.h (S)
inline constexpr int kCornerPlacesPerBlock = 125;   // 5 x 5 x 5 lattice points of a block.
struct CornerPlaceIndex { std::uint32_t value; };   // block * 125 + (z * 5 + y) * 5 + x. One grid version only.
constexpr CellBlockIndex ToCellBlockIndex(CellPlaceIndex cell);   // cell.value / 64.   (particles, transfer)
constexpr CellBlockIndex ToCellBlockIndex(FacePlaceIndex face);   // face.value / 240.  (transfer)

// grid_tables.h (S)
SizeLevel ToSizeLevel(const GridTables& tables, CellBlockIndex block);  // Level of a block, so of its places.
// The 8 corners of a cell (bit 0 = high x, bit 1 = high y, bit 2 = high z) and the 4 corners of a face piece
// (bit 0 = high side on the lower of its two in-plane axes). Always the corner's one stored place: a corner
// shared by cells of two sizes or of two blocks has one index.                                  (pressure, solids)
struct CellCorners { CornerPlaceIndex corners[8]; };
CellCorners FindCornerPlaces(const GridTables& tables, CellPlaceIndex cell);
struct FaceCorners { CornerPlaceIndex corners[4]; };
FaceCorners FindCornerPlaces(const GridTables& tables, FacePlaceIndex face);
Vec3 ComputeCornerPosition(const GridTables& tables, CornerPlaceIndex corner);   // m, world

// The four cell places around one edge of `cell`. The edge is parallel to the third axis; quarter bit 0 set
// means the high side of first_axis, bit 1 the high side of second_axis. cells[0] is `cell`; the others are
// kNoIndex where that place holds no active cell of the same size. is_same_size is false when any of the
// three other places is covered by cells of another size.                                        (viscosity)
struct CellsAroundEdge { CellPlaceIndex cells[4]; bool is_same_size; };
CellsAroundEdge FindCellsAroundEdge(const GridTables& tables, CellPlaceIndex cell, Axis first_axis,
                                    Axis second_axis, int quarter);

// Every active cell whose particles can list `face` as a nearby face (5.4, step 2): cell_low, cell_high and
// the cells touching either. Ascending index order.                                              (transfer)
inline constexpr int kMaxCellsNearFace = 112;   // 2 + 2 x 56 - 2. An upper bound; a test searches for the maximum.
struct CellsNearFace { CellPlaceIndex cells[kMaxCellsNearFace]; int count; };
CellsNearFace FindCellsNearFace(const GridTables& tables, FacePlaceIndex face);

// Walks the face pieces of the same axis whose centre is within `radius` widths of `face`'s own cell width,
// in ascending FacePlaceIndex. It reads the 27 same-level blocks around the face's block and their bigger and
// smaller blocks, so the radius is at most kMaxFaceSearchRadius. No list is stored: the count has no small
// bound where sizes mix.                                                                         (transfer)
inline constexpr double kMaxFaceSearchRadius = 4.0;   // Cell widths. One block width.
struct FaceSearch { FacePlaceIndex face; double radius; std::uint32_t blocks[270]; int block_count;
                    int next_block; int next_place; };
FaceSearch BuildFaceSearch(const GridTables& tables, FacePlaceIndex face, double radius_in_cell_widths);
FacePlaceIndex FindNextFace(const GridTables& tables, FaceSearch* search);   // kNoIndex when done

// cell_block_range.h
enum class BorderWidth : std::uint8_t {
  kTouchingCells,    // Blocks storing an active cell that touches a cell of the set, or storing a face piece
                     // of a cell of the set or of such a touching cell (5.1 step 5: a piece may be stored in its other cell's block). Enough
                     // for nearby faces (5.4) and touching cells.
  kFourCellWidths,   // Also every block FaceSearch can read from a face piece stored in the set. For
                     // transfer's fills.
};
// Precondition (A17): grid.run_on() == RunOn::kCpu, as the two-argument form.
RangeWithBorder BuildRangeWithBorder(const StaggeredGrid& grid, CellBlockRange range, BorderWidth width);

// weight_curve.h (S)
// m2. Weighted spread, along one axis, of 27 same-size faces all kept: cell_width^2 / 4.         (particles)
constexpr double ComputeCompleteWeightedSpread(double cell_width);

// nearby_face_weights.h (S)
// The three sums of 5.4 step 5 and 7 for one particle and one axis, small enough to keep per particle.
struct NearbyFaceSums { double weight_sum; Vec3 raw_weight_change_sum; Vec3 mean_offset; };
NearbyFaceSums ToNearbyFaceSums(const NearbyFaceWeights& weights);
// The weight one face gets from one particle, seen from the face. `cell` is the active cell containing
// `position`; `sums` are that particle's sums for the face's axis and the caller's kept mask. Returns weight 0
// when the face is not in FindNearbyFaces' list for that position. Bit-identical to the entry
// ComputeFaceWeight(tables, weights, position_in_list) returns for the same face.                (transfer)
FaceWeight ComputeFaceWeight(const GridTables& tables, FacePlaceIndex face, Vec3 position, CellPlaceIndex cell,
                             const NearbyFaceSums& sums);
```

**Where a corner is stored.** A corner place is a point of the 5 x 5 x 5 lattice of a block. One physical corner
can be a lattice point of several blocks (two blocks of one level that share a side; blocks of two levels).
It is stored once: at the smallest size level that has a block whose lattice holds the point (whether or not a
cell of that block touches it), and among several blocks of that level in the one that comes last in the fixed
order (highest `z`, then `y`, then `x`: the block on the high side, the rule of face plane 4 in 5.1; plan
revision 2026-11-19, with three fixtures in grid plan Task 15). `FindCornerPlaces` returns that place. A corner of a small
cell in the middle of a big cell's face or edge belongs to the small cells only. Cost: a `CornerValues<double>`
is 1,000 B per block. Corner places are kept (decision 14).

### 3.8 `cell_size_rule.h` (S): the shared cell size rule (NEW, A41, A61–A63)

Moved here from `cell_resize/cell_size_rule.h` (`CellSizeSettings`, `CellSizeFacts`, `ComputeWantedCellWidth`) and
completed by `plan-decisions.md` "Owner answers A40–A64 applied" item 2. Every function is `FLUIDITY_CPU_AND_GPU`,
allocates nothing, uses `math::` and no fused multiply-add. The grid owns the per-cell formulas; the pass that fills
the facts and applies the hysteresis is cell resize's `ComputeCellSizeChanges`, and stage 2 of the step calls
`ComputeNewCellSizeLevel` in its marking pass. `cell_resize::CellResizeSettings::cell_size` is a
`grid::CellSizeSettings`. There is no `ChooseCellSize` callback (removed).

```cpp
struct CellSizeSettings {                     // NEW (A41). Chosen once per scene. Replaces cell_resize::CellSizeSettings.
  SizeLevel smallest_size_level{};            // Width w_min: the surface, pockets, gas and solids.
  double surface_reach_in_cell_widths = 8;    // No unit. Depth term: w_depth = w_min * max(1, depth / (8 w_min)).
  double solid_reach_in_cell_widths = 2;      // No unit. Solid term: w_solid = w_min * max(1, d / (2 w_min)).
  double shear_width_factor = 0.25;           // No unit. c_u of literature-cell-size.md 5(a); starting value,
                                              //   to calibrate (A62).
  double merge_width_share = 0.8;             // No unit. Merge only while 2 w <= 0.8 x wanted width.
  double merge_wait_duration = 0.25;          // s of committed time the merge test holds without a break.
  std::uint32_t min_merge_wait_steps = 8;     // Committed steps the merge test holds without a break.
  std::uint32_t max_cell_splits_per_resize = 1u << 16;   // Limit. More wanted splits wait.
};
struct CellWidthsByLevel {                    // NEW. m. width[l] = ComputeCellWidth(frame, l), computed once on the CPU.
  double width[kMaxSizeLevels];               // Compared, never recomputed with log2 or pow on either processor.
  std::uint8_t size_level_count = 1;
};
// NEW (plan decision, grid plan Task 13a). Fills width[l] for l < frame.size_level_count, 0 beyond: the one place
// the table is filled.
CellWidthsByLevel ComputeCellWidthsByLevel(GridFrame frame);
struct CellSizeFacts {                        // NEW (A41, moved). What the rule reads about one active cell.
  double cell_width = 0;                      // m.
  double depth_below_surface = 0;             // m. 0 at the surface; +infinity beyond reach. NEW (A66): a negative
                                              //   value (step::BuildStartCells' centre bound minus half the cell
                                              //   diagonal) counts as the surface: max(1, ...) gives w_min.
  double distance_to_solid = 0;               // m. +infinity: none within reach. Negative: as for the depth (A66).
  double velocity_change_size = 0;          // 1/s. Held maximum (cell resize); 0 outside liquid.
  double cell_size_factor = 1;                // NEW (A78, A79; replaces artist_size_level). No unit, 1/64..64: the
                                              //   refinement volume's smallest factor over the cell
                                              //   (CellSizeFactorTable of ComputeCellSizeFactorCode); 1: no wish.
  bool holds_particles_that_keep_their_count = false;   // Read by cell resize's merge test, not here.
  bool touches_air_pocket = false;
  bool wants_smallest_cells = false;          // Gas in or beside the cell (A10).
  bool touches_air_gas_or_solid = false;      // Merge veto (cell resize 5.3 rule 5); not read here.
};
// m/s. U_ref = max(largest_liquid_speed, sqrt(gravity_size * biggest_cell_width)) (literature-cell-size.md 5(a)).
double ComputeReferenceSpeed(double largest_liquid_speed, double gravity_size, double biggest_cell_width);
// m. CHANGED (A78, A79): wanted = clamp(min(w_depth, w_solid, w_shear) * cell_size_factor, w_min, w_max), w_min =
// widths.width[smallest_size_level], w_max = widths.width[0]; w_shear = c_u U_ref / g for g > 0, else +infinity.
// w_min where touches_air_pocket, wants_smallest_cells, or depth_below_surface < surface_reach_in_cell_widths * w_min
// (the cells within 8 smallest widths of the surface, A61: the factor does not reach them in the first version). An infinite fact gives an
// infinite term, never NaN. No FMA contraction (strict rounding of the shared headers).
double ComputeWantedCellWidth(const CellSizeSettings& settings, const CellWidthsByLevel& widths,
                              double reference_speed, const CellSizeFacts& facts);
// The biggest level (smallest index) whose width <= wanted_width, by comparison with widths.width;
// size_level_count - 1 when wanted_width is below every width.
SizeLevel ComputeWantedSizeLevel(const CellWidthsByLevel& widths, double wanted_width);
// A cell that is not in the committed grid (stage ChooseActiveCells): the level of the nearest committed active
// cell (FindCellAt probes at position + {-1,0,1}^3 * biggest width, nearest centre, ties in fixed cell order) plus
// one, capped at smallest_size_level; smallest_size_level when none is found. This is the rule above for a new cell:
// it borders air, so its depth is 0 and its wanted width is w_min (A61); the cap keeps the size-step rule against
// the committed cell it grew from. Reads the tables on their own processor.
SizeLevel ComputeNewCellSizeLevel(const GridTables& committed, Vec3 position, SizeLevel smallest_size_level);
// REMOVED (A78, A79): CellSizeRegion, ArtistCellSizes, ComputeArtistSizeLevel. NEW: the refinement volume.
inline constexpr int kCellSizeFactorStepsPerDoubling = 4;   // Code c means factor 2^(c / 4).
inline constexpr int kMaxCellSizeFactorCode = 24;           // Codes -24..24: factors 1/64..64. 0 is factor 1.
struct CellSizeFactorTable { double factor[2 * kMaxCellSizeFactorCode + 1]; };  // factor[c + 24] = 2^(c / 4).
// Changed (plan revision 2026-11-19): shared and constexpr, no library call. factor[i] = 2^(i / 4 - 6) * root[i % 4]
// (integer division), root = {1.0, 0x1.306fe0a31b715p+0, 0x1.6a09e667f3bcdp+0, 0x1.ae89f995ad3adp+0}: the doubles
// nearest to 2^0, 2^(1/4), 2^(1/2), 2^(3/4). A double times a power of two is exact in this range (2^-6..2^6), so
// every entry has the same bits on every machine and both processors. The one place the table is filled.
constexpr CellSizeFactorTable ComputeCellSizeFactorTable();
struct RefinementFactors {                    // Not owning, on one processor: the current key's codes as a min-pyramid.
  const std::int8_t* codes = nullptr;         // Every layer, layer 0 first; each z, then y, then x fastest.
  std::uint32_t layer_first_code[kMaxSizeLevels] = {};   // Index of each layer's first code.
  std::uint32_t count_x = 0, count_y = 0, count_z = 0;   // Layer 0 sample counts. 0: no volume (factor 1).
  std::uint8_t layer_count = 0;               // Layer L: min over 2x2x2 of layer L - 1, counts ceil(count / 2).
  std::uint8_t sample_spacing_log2 = 2;       // Sample spacing 2^this smallest cell widths, anchored at frame.origin.
};
// Code of the smallest factor over every sample the cell overlaps, one integer read: s = smallest_size_level -
// level; s <= spacing_log2: layer 0, index (coord << s) >> spacing_log2; else layer s - spacing_log2, index coord.
// 0 without a volume or for an index outside its layer.
std::int8_t ComputeCellSizeFactorCode(const RefinementFactors& factors, SizeLevel smallest_size_level, CellId cell);
```

## 4. Consumes from other modules

Each item is marked EXISTS (Foundation, including the `base/` additions) or PROVIDED (by a designed module).
Nothing is NEEDED.

**Foundation (`base/`), EXISTS:** `struct Vec3`, `struct Mat3`, `enum class Axis { kX, kY, kZ }`,
`math::Dot` (A11), `math::UnitVector(Axis)`, `enum class RunOn : std::uint8_t { kCpu, kGpu }` (shared
`base/run_on.h`), `std::expected<void, RunOnError> CheckCanRunOn(RunOn)` (A12, `base/check_can_run_on.h`),
`FLUIDITY_CPU_AND_GPU` (A13), `base/cpu_only.h`, `math::TreeSum` (`base/tree_sum.h`), `BuildStableOrder`
(`base/stable_order.h`), `ReadMemoryAvailable` (`base/processor_description.h`; GPU no-allocation tests read its
free bytes). NEW (A58; Foundation Tasks 17 to 19, `base/run_memory.h`): `RunMemory`, `BuildRunMemory`,
`BuildBuffer(RunOn, std::size_t, RunMemory&)`, `BuildOrderBuffers(RunOn, std::size_t, RunMemory&)`,
`ComputeRunMemoryPieceBytes`, `ComputeRunMemoryBytes(estimate, counts, MemoryWhere where)`,
`BufferError::kRunMemoryTooSmall`; every grid builder takes the `RunMemory` of the processor its arrays live on.

**Foundation additions (EXISTS: `shared-interface.md` section (a) and `changes-to-apply.md` "Additions to
Foundation `base/`"):**

```cpp
namespace math { double Floor(double); double Abs(double); double Sqrt(double);    // base/scalar_math.h (S)
                 bool IsFinite(double); bool IsFinite(Vec3); }
template <typename Value> class Buffer;                                            // base/buffer.h
template <typename Value> std::expected<Buffer<Value>, BufferError> BuildBuffer(RunOn run_on, std::size_t size);
template <typename Value> std::expected<void, BufferError> WriteBuffer(std::span<const Value>, Buffer<Value>&);
template <typename Value> std::expected<void, BufferError> ReadBuffer(const Buffer<Value>&, std::span<Value>);
// Added 2026-11-14. The grid uses the slice forms of WriteBuffer and ReadBuffer (first, count) to move only the
// used part of bound values, of the status record and of ReadGrid's tables. (A17: there is no CPU copy of GPU
// tables any more, so the grid itself needs no CpuMemory::kLockedForCopies.)
enum class CpuMemory : std::uint8_t { kOrdinary, kLockedForCopies };               // BuildBuffer's optional input
enum class CountedPer : std::uint8_t; enum class KeptFor : std::uint8_t;           // base/memory_count.h
struct MemoryEstimateEntry; struct MemoryCounts;
std::uint64_t ComputeMemoryBytes(std::span<const MemoryEstimateEntry> rows, const MemoryCounts& counts);
// Added 2026-11-14 (plan-decisions.md, Foundation): MemoryEstimateEntry has a field `MemoryWhere where { kRunOn, kCpu }`;
// the form with `where` sums only the rows kept there. Every grid row sets it.
std::uint64_t ComputeMemoryBytes(std::span<const MemoryEstimateEntry> rows, const MemoryCounts& counts, MemoryWhere where);
// A30: BufferError has no kRunOnMismatch; buffers on two processors are a Requires: precondition of CopyBuffer, the
// functions below and the ordered sums, checked by a debug assertion. The grid has nothing to map.
template <typename Value> std::expected<void, BufferError> CopyBuffer(const Buffer<Value>& from, Buffer<Value>& to);
```

**Foundation additions needed by the GPU build (EXISTS once Foundation adds them before grid plan Task 6;
`plan-decisions.md` "Grid built on the GPU (A17)", `base/stable_order.h`):**

```cpp
// Temporary arrays of BuildStableOrder and ComputeExclusivePrefixSum on one processor, for up to max_count
// items, so that neither allocates per call. Move-only.
class OrderBuffers { public: std::size_t max_count() const; RunOn run_on() const; };
std::expected<OrderBuffers, BufferError> BuildOrderBuffers(RunOn run_on, std::size_t max_count);
// Exact bytes BuildOrderBuffers allocates: bytes_per_item * max_count + fixed_bytes.
struct OrderBuffersSize { std::uint32_t bytes_per_item; std::uint64_t fixed_bytes; };
OrderBuffersSize ComputeOrderBuffersSize(RunOn run_on);
// The one form (A31): allocates nothing. kSizeMismatch when count > order_buffers.max_count(). Requires: every
// buffer on order_buffers.run_on() (A30).
std::expected<void, BufferError> BuildStableOrder(const Buffer<std::uint64_t>& keys, std::size_t count,
                                                  const Buffer<std::uint32_t>* previous_order,
                                                  Buffer<std::uint32_t>& order, OrderBuffers& order_buffers);
// sums[i] = values[0] + ... + values[i - 1] for i in [0, count]; sums[count] is the total. Exact unsigned
// arithmetic; a total above 2^32 - 1 is kSizeMismatch. sums.size() >= count + 1. `sums` may be the same buffer
// as `values` (in place). Allocates nothing. Same bytes on CPU and GPU.
std::expected<void, BufferError> ComputeExclusivePrefixSum(const Buffer<std::uint32_t>& values, std::size_t count,
                                                           Buffer<std::uint32_t>& sums, OrderBuffers& order_buffers);
```

**Every module, settled as a convention (A2):** an error with context is `std::expected<T, <Module>Failure>`
with `struct <Module>Failure { <Module>Error error; ...context... }`; here `GridFailure`. `ToName` beside
`ToMessage`; rows from `Build<Module>MemoryEstimate`. The grid has no long-running solve, so it takes no
`cancel_requested`.

**Contact, particles, transfer, pressure, viscosity, materials, solids, gas:** nothing is consumed. What they
do with the grid (PROVIDED to them by section 3): particles groups by the `CellPlaceIndex` of `FindCellAt` and
passes that cell's block as `near`; transfer builds the kept mask (same `contact::PlateSide`, face
`outside_solid`, value present, the material set's own values, A8) and owns all sums; pressure and solids store corner
samples in `CornerValues`; solids fills `grid_edge_is_open_to_sky` in a `FaceValues<std::uint8_t>`.

**Step (PROVIDED, step 3.2, 3.3, 3.10, 5.3):** stage `ChooseActiveCells` (`step::Stage::kChooseActiveCells`)
produces the `active_cells` list and calls `BuildGrid(inputs, buffers)` with `version + 1` into the try grid's
buffers, on the run's processor (A17: in a GPU run the list is a `Buffer<CellId>` on the GPU,
`GridInputs::active_cell_buffer`, with its entry count as a CPU number); `ReadyScene::grid_frame` gives the frame;
`SimulationLimits::max_cell_blocks` sizes the two grid buffers and every value buffer (reconcile: set by
`step::BuildRunPlan` from the scene, 5.8); the step owns one `GridBuilderBuffers` lent to both grid buffers and counts them with
`BuildGridMemoryEstimate(run_on, 2, max_cell_blocks, BuilderBuffersSharing::kOneForAllGrids)` (reconcile, decision 29); the step binds
`GridFields` with `ApplyGridVersion`. The step relies on `BuildGrid` accepting an empty list (`plan-decisions.md`,
Step). In a batched GPU run the step also keeps a CPU copy of the grid built with the CPU builder from the same list
(`step::FindCpuGridCopy`; `"step: grid copy on the CPU"`, two CPU buffers sharing one CPU `GridBuilderBuffers`).

**Step and cell resize read the shared cell size rule (NEW, A41, A61–A63; 3.8):** stage 2 calls
`ComputeNewCellSizeLevel` in its marking pass on the grid's processor; cell resize's `ComputeCellSizeChanges` fills
`CellSizeFacts` per active cell and calls `ComputeWantedCellWidth`, `ComputeWantedSizeLevel` and
`ComputeArtistSizeLevel`; `ResizeCells` calls `ComputeReferenceSpeed` with `step::CommittedStepLimits::
largest_liquid_speed`. The grid consumes nothing from them for this.

**Cell resize and batches (PROVIDED, cell resize and batches 3.3, 3.4, 3.7):** `cell_resize::SplitCell`, `MergeCells`,
`FindCellsThatMustSplit` produce the resized list and `cell_resize::ResizeCells` builds it with `BuildGrid`; on a
GPU grid it builds the resized grid on the CPU for its map, while the step builds the same list on the GPU (A17;
reconcile: through `step::BuildResizedGrid`, from the list cell resize uploads). It reads the committed grid as a
`ReadGrid` copy in a one-piece GPU run (its two CPU grids share one CPU `GridBuilderBuffers`) and as the step's CPU copy in a
batched GPU run, where its CPU resized grid is built into the step's free CPU copy buffers and handed over with
`step::ApplyResizedCpuGridCopy` (no `ReadGrid`; reconcile, `plan-decisions.md` coordinator item 6).
`batches::BuildBatchPlan` and `BatchedRun` use `BuildRanges`, the buffer form of `FindBorderBlocks` (stacked borders,
`BorderSearchBuffers`), the buffer form of `BuildGridPiece` and `block_in_whole_grid_buffer()` in a GPU run, the
span forms in a CPU run; the loaded try's grid buffers borrow the step's `GridBuilderBuffers` and are counted without the
`builder buffers:` entries (reconcile).

**Pressure (PROVIDED, `plan-decisions.md` "Batched solves (A15)"):** `pressure::SolveBatches` builds its loaded
piece with the buffer form of `BuildGridPiece` and cuts with `BuildRanges` and the buffer form of `FindBorderBlocks`
(one call per layer); its `GridBuffers`, `BorderSearchBuffers`, two border buffers and map copy are counted in
`pressure::BuildSolveBatchesMemoryEstimate`.

**For the sand module:** nothing specific. Sand's force sums are `FaceValues` it owns, on the same face places
and weights.

## 5. Method

### 5.1 The grid (new structure; face rules ported from `adaptive_mac_topology_3d.cc`)

1. A cell at size level `l` has width `biggest_cell_width / 2^l` and integer coordinates in cells of its own
   level, so every cell is aligned by construction. A big cell is split into its 8 small cells or not at all
   (`kSiblingCellMissing` otherwise; `main` allowed a lone small cell, which forced quarter "exterior" pieces).
2. **Size-step rule (ported):** two active cells that touch by a face, an edge or a corner differ by at most
   one size level.
3. A cell block is 4 x 4 x 4 cell places of one size level. A block exists when at least one of its places is
   `kActive` or `kSplit`. Blocks therefore exist at every level above a small cell (about 14 % extra blocks).
4. **Fixed order of blocks:** low corner in smallest-cell units by `z`, then `y`, then `x`, then bigger level
   first. With Z up, a run of consecutive blocks is a horizontal layer of the grid; `CellBlockRange` is such a run.
5. **Face places.** For each axis a block has 5 planes of 4 x 4 places at its own level. A place is a face
   piece when the two cell places beside it are: active and active (`kBetweenCells`, same size); active and
   `kInsideBiggerCell` (`kBetweenCells`, the piece is the small cell's whole face and one quarter of the big
   cell's face); active and `kAbsent` (`kGridEdge...`). Active beside `kSplit` is not a piece: the four pieces
   live one level down. So **a piece is always stored at the level of its smaller cell, has that cell's face
   area, and exists exactly once.** Plane 4 of a block duplicates plane 0 of the same-level block on its high
   side: when that block exists the piece is stored there and plane 4 is `kNotAPiece`.
6. Orientation is fixed: the normal is `+axis`, `cell_low` is on the minus side. A value stored on a piece is
   the part of the velocity along `+axis`. Flow out of `cell_low` is positive.
7. `FacePieceId` and `CellId` do not depend on which blocks exist; indexes do. `FindCellPlace` and
   `FindFacePlace` translate an id into the current version, which is what remap needs.
8. `BuildGrid` steps (CPU builder): first the input checks, in this order on both processors: frame, version,
   list shape, then the list length (more than `64 * max_cell_blocks` cells is `kTooManyCellBlocks`, `cell` zero);
   they come before every keyed check and never take part in the key minimum (ruling 2026-11-19: the CPU used
   to look at the ids first). Then 1 check the ids (form and coordinates); 2 copy the ids into the lent
   builder buffers and sort them in place, reject overlaps; 3 mark split cells and check the siblings of every active and every split cell (each active cell checks its
   own siblings and those of the split cells its eighth-0 chain adds; plan revision 2026-11-19); 4 check the size-step
   rule; 5 create and sort blocks in the buffers' block-id table, check their count against the buffers' room;
   6 fill neighbour tables; 7 fill cell states; 8 fill face piece kinds; 9 (changed, A17) on GPU buffers the
   whole build runs on the GPU by item 9 below: nothing is uploaded. No step allocates. Nothing is returned on
   failure and the buffers stays with the caller. The reported problem is the first by the failure key of item 9.
   The CPU builder writes every table entry through the shared rules of `grid_build_steps.h` (sort keys, the
   per-position checks, `FindBlockNeighbours`, `ComputeCoveredCellState`, `ComputeFacePieceKind`) and in no other
   way; the GPU kernels call the same functions, which is why the tables are identical. Frame, version and input
   shape are checked on the CPU before any work on either processor.
9. **Parallel build (new, A17; `<proc-build-grid-in-parallel>`).** Each sequential step becomes "compute a key,
   stable sort, mark, prefix-sum, scatter" (`BuildStableOrder` and `ComputeExclusivePrefixSum` with the buffers'
   `OrderBuffers`). Every kernel loops grid-stride over a count it reads from `GridBuildStatus`, so no count comes
   back mid-build, and writes nothing when an earlier step failed, except that a value a
   prefix sum reads (the split-cell counts of P7, the marks of P8, the border marks) is written as 0, so no prefix
   sum adds words of an earlier launch (ruling 2026-11-19). P1 (CPU) the input checks of item 8; `n` above
   `64 * max_cell_blocks` fails there, before any key exists; a one-thread kernel resets the status. P2 one thread per input index:
   `CheckCellId`. P3 three stable-sort passes of `ToCellSortKey` (least significant first: x with the level, then
   y, then z, in level-15 units) put the cells in fixed cell order (= `ComesBefore`). P4 overlaps, P5 siblings,
   P6 size-step rule: one thread per sorted position; each check looks only at the smaller cell of a pair, so a
   broken pair is reported at exactly one position. P7 add split cells: an active cell adds its ancestor one
   level up when it is eighth 0, the one two levels up when that ancestor's child on its path is also eighth 0,
   and so on (siblings exist together, so each split cell is added exactly once); counts, prefix sum, scatter.
   P8 three stable-sort passes of `ToBlockSortKey` group cells by block in fixed block order; mark the first
   entry of each block, prefix sum, scatter into `block_ids` (no second sort). P9 neighbour tables, one thread
   per block. P10 cell states, one thread per place and per sorted entry, then `ComputeCoveredCellState`. P11
   face piece kinds, one thread per face place, grid-stride like every other step; a thread adds up its face pieces
   and issues one integer `atomicAdd`. P7 ends by writing `CellId{0}` to every place from `n + split_cells` to the
   sort count, so the sort of P8 never reads a cell of the build before (plan revision 2026-11-19). P12 counts and status; one `ReadBuffer` of
   `GridBuildStatus`, the only transfer. Corner places need no table.
   **First failure, both processors:** the smallest key `(step << 56) | (position << 8) | detail`: step 1 id
   checks (position = input index), 2 overlaps, 3 siblings, 4 size-step rule (position = place in fixed cell
   order, detail = missing eighth or direction 0..25), 5 and 6 room (position 0, `kTooManyCellBlocks`, `cell`
   zero). The CPU runs each step to its end and scans positions in order; the GPU takes an integer `atomicMin`.
   The GPU's only atomics are integer `atomicMin` (first failure) and `atomicAdd` (counts), whose results do not
   depend on thread timing.

### 5.2 Containing cell

`coordinate = floor((position - origin) / smallest_cell_width)` per axis, then walk down from the level-0
block (or from `near`) following `kSplit`. Before any integer cast, a coordinate that is not finite or outside `[-kMaxCoord 2^(L-1), (kMaxCoord + 1) 2^(L-1))`
(`L` = `size_level_count`; exactly the cells a level-0 id can name) gives `kNoIndex` (plan revision 2026-11-19).
This one formula is the only definition of "inside", so a particle
one rounding step from a cell side is assigned consistently everywhere. The exact-edge recovery of
`sparse_mac_grid_3d.cc:83-198` is not ported.

### 5.3 The weight curve (ported unchanged from `sparse_mac_grid_3d.cc:285-296`, `basis.py`)

For `t` = distance in cell widths: `Q(t) = 3/4 - t^2` if `|t| < 1/2`; `(3/2 - |t|)^2 / 2` if
`1/2 <= |t| < 3/2`; `0` otherwise. Slope `Q'(t) = -2t`, or `-(3/2 - |t|) sign(t)`.

### 5.4 Nearby faces and their weights

Procedure for one position `x_p` and one axis (ported: the scale-centre-measure procedure of
`graded_mac_transfer_3d.cc:185-267` and of the old paper's `<sec-masks>`; changed: the raw weight uses the
uniform curve instead of the radial one; new: the bounded candidate rule):

1. Find the active cell `c` that contains `x_p`; `h` is its width and `p` its size level.
2. **Candidates:** the face pieces of this axis that belong to `c` or to a cell touching `c`, whose own size
   level is `p - 1`, `p` or `p + 1`, and whose centre `x_f` satisfies `-1.5 h < x_f - x_p <= 1.5 h` on each axis (owner decision
   2026-11-19: open at the low end, closed at the high end, because a cell owns its low side; so every position of
   a same-size neighbourhood, a cell side, centre or corner included, has exactly 27 candidates, those at
   exactly `1.5 h` with weight 0). The test is made in integers, never on float offsets (ruling 2026-11-19: a
   float test gives 2 or 4 planes within a rounding step of a side or centre): per axis `n_p = floor(4 ((x_p -
   origin) / h - c))` clamped to 0..3, `c` the coordinate of the cell `FindCellAt` returned, and a face is a
   candidate when its centre's coordinate `n_f` in quarter widths from that cell's low corner (an integer, from
   the ids) has `n_p - 5 <= n_f <= n_p + 6` (the paper's symbols). List them in ascending `FacePlaceIndex`.
3. The caller removes candidates it may not use (other side of a plate, not `outside_solid`, no value) by
   clearing bits of the mask. The grid never looks at physics.
4. Raw weight: `w_f = Q(d_x) Q(d_y) Q(d_z) * (area_f / h^2)` with `d = (x_f - x_p) / h`. Raw change with the
   particle position: `g_f = -(area_f / h^3) * (Q'(d_x) Q(d_y) Q(d_z), Q(d_x) Q'(d_y) Q(d_z), ...)`.
5. Sums over kept faces in list order: `W = sum w_f`, `G = sum g_f`. If `W <= 0`, no kept face has weight: the
   fit is `kNone`, every weight, change, mean and spread is zero, and steps 6 to 9 do not run, so nothing divides
   by `W` (as `<proc-nearby-face-weights>`).
6. Scaled weight and change: `weight = w_f / W`, `weight_change = (g_f - weight * G) / W`. Then
   `sum weight = 1` and `sum weight_change = 0` whatever was left out.
7. `mean_offset = sum weight * (x_f - x_p)`; `offset_from_mean = (x_f - x_p) - mean_offset`.
8. `weighted_spread = sum weight * offset_from_mean * offset_from_mean^T` (m2).
9. Invert by Cholesky on `weighted_spread / h^2`. If a pivot is at most `kMinPivotShare * trace`, or
   `W < kMinWeightSum`, the fit is `kMeanOnly`; else `kFull`, with the inverse.

**What replaces `4 / h^2`:** `inverse_weighted_spread`, always measured from the faces actually kept. When all
27 same-size faces are kept, steps 5 to 8 give `W = 1`, `G = 0`, `mean_offset = 0` and
`weighted_spread = h^2/4 * identity` up to rounding, so the old shortcut is the special case and never a
separate code path. Transfer then computes, per axis, `fitted_velocity_change row =
(sum weight * value * offset_from_mean) * inverse_weighted_spread`, the value at the particle as
`sum weight * value - row . mean_offset`, and writes `value + row . offset_from_mean`, which keeps the
particle's momentum exactly. `exact_velocity_change row = sum value * weight_change`.

Properties to state in the header and test, for every grid and every mask whose fit is not `kNone`: constants are
reproduced for any kept set; linear fields are
reproduced whenever `fit == kFull`; weights are never negative, and 0 only for a face exactly `1.5 h` away; weights jump when the particle changes cell across a
size change (as on `main`), which is the known open physics "traction across cells of different sizes".

**Why at most 212, and 124 in practice** (owner decision 2026-11-19; the earlier bound of 76 and its argument
"replacing a small neighbour by a same-size one removes at least as many pieces as it adds" were wrong: a
same-size neighbour's far faces can be split by smaller cells beyond it, and those pieces are candidates).

*What is known.* Complete same-size neighbourhood: 27. All 26 neighbour places split: 76 in every window. The
largest case known is 124 on one axis: a cell whose 8 corner neighbours and 6 of its 12 edge neighbours have its
size, with every other neighbour and everything beyond one level smaller. 124 is the exact maximum where only two
cell sizes meet: the count is a quadratic function of the 26 neighbours being whole or split, and it was maximised
over all `2^26` cases, 64 windows and 3 axes. A neighbour one level bigger is believed to give fewer pieces; that
case is not searched exhaustively.

*What is proven: no valid grid passes 212.* Work on one axis, in quarter widths `h / 4` from the low corner of `c`.
1. The smallest candidate is a piece of level `p + 1`: a square of side `h / 2`. Call its possible places slots:
   planes at even `n_f` along the axis, centres at odd `n_f` across.
2. A candidate lies on a face of `c` or of a cell touching `c`. Its plane is an even `n_f` inside the window and
   inside `-4..8` (a touching cell one level bigger has its other faces at `-8` or `12`, outside every window):
   at most 6 planes. Across, it covers at least one slot whose centre is inside the window and inside `-3..7`: a
   piece of level `p + 1` is one slot; a piece of level `p` or `p - 1` has its centre in the window, so the slot
   one quarter to one side of that centre is in the window too, and it lies in `-3..7` because the piece is on a
   cell that touches `c`. The window holds at most 6 odd `n_f` in `-3..7`.
3. Pieces of one axis do not overlap, so two candidates never get the same slot.
4. So there are at most `6 x 6 x 6 = 216` candidates, less the 4 slots of the plane `n_f = 2` that lie inside `c`
   itself, where no piece can be: 212.

`kMaxNearbyFaces` is 216, the count before the last subtraction; the mask has four 64-bit words.
`kTooManyNearbyFaces` therefore guards against a broken table, not against a grid. A cell has 64 candidate lists
per axis (the values of `n_p`); tests go through all of them on the 76 case, on the 124 case and on random grids,
and report the largest count found.

### 5.5 Ranges and border

A stage receives `(grid, range)`. It writes only values stored in blocks of the range and may read values of
the range and of its border blocks. Because every face piece is stored in exactly one block, every value has
one writer. A particle in a range cell can have nearby faces stored in a border block: transfer therefore
groups by destination face and reads the particles of border blocks (its design). In the one-piece run the
range is `all_blocks()` and the border is empty.

**Border blocks** (`FindBorderBlocks`). `kTouchingCells`: a block outside the set is in the border when it
stores an active cell that touches a cell of the set, or stores a face piece of such a cell. The second part
matters: a piece is stored with its smaller cell, and plane 4 of a block moves to plane 0 of its high
same-size-level neighbour (5.1 step 5), so a nearby face of a cell touching the range can lie one block beyond that
cell. `kFourCellWidths` adds every block `FaceSearch` reads from a face piece stored in the set: the 27
same-level blocks around the piece's block and their bigger and smaller blocks. Both are computed by walking
the grid's blocks once in index order and testing each against the sorted set through its own neighbour
tables (the reverse of the same relation), so the result is ascending with no repeats and the walk allocates
nothing. The per-block test is the shared `ComputeIsBorderBlock(tables, set, block, width)` of `border_block_steps.h`,
through the block's own 27 same-size-level neighbours, its bigger block and its 8 smaller blocks (the grid's spatial
index: no other search structure is built). **On the GPU (A17)** one thread per block marks it (outside the range,
and in `border_so_far` or `ComputeIsBorderBlock`), an exclusive prefix sum over the marks gives each marked block its place,
and a scatter writes the list; index order is kept, so the list is the CPU walk's, and only the count comes back.
**Stacked borders:** a pass that runs several stages on one load needs, around the range, the border
of its last stage, then around range plus that border the border of the stage before, and so on; each layer is
one more `FindBorderBlocks` call on the set found so far (cell resize and batches, 5.8). The buffer form does one
layer per call and merges it with the layers before, two output buffers used in turn.

**Sums over the grid** that are not sums per destination are one partial per cell block, in the fixed block
order, kept in a `CellBlockValues<double>`, then `math::TreeSum` over the block partials (`changes-to-apply.md`,
every design). The grid computes no such sum; it fixes the order, so a total is the same in one piece and in
any batch cut.

### 5.6 What the grid exposes for changing cell sizes (cell resize and batches)

Ids that survive a version change with `FindCellPlace`/`FindFacePlace`; `ComputeBigCell`, `ComputeSmallCell`,
`ComputeCellEighth`; `FindTouchingCells`; `CheckSizeStepRule` on a proposed list (its `GridFailure` names the
two cells, which is what a "which cells must also split" search needs); `ComputeGridCounts` before
building; `BuildGrid` with `version + 1` into grid buffers, and `ApplyGridVersion` for the remapped values, so a
resize allocates nothing; `FindFacePieces` for the flux through a big face as the sum of its pieces;
`FindBorderBlocks` and `BuildGridPiece`. `SplitCell`, `MergeCells`, `ResizeCells`, the closure search
(`RefinementClosure` on `main`) and all value remap belong to the cell resize and batches design.

### 5.7 Every material

Liquids and mixtures, snow, springy solids, solids being cut, moving solids and gas (also beside liquid in one
scene, A10) all use the same face places and the same weights. Per-ingredient amounts, snow and springy-solid
force sums, `SolidSide` per face, `open_area`, solid velocity, gas temperature and fuel are
`FaceValues`/`CellValues` owned by their modules; materials that do not mix are separate material sets, each
with its own values on the same places (A8). Cutting enters only through the kept mask. Gas scalars at cell
centres use `ComputeCellCenter` and `FindTouchingCells`; the gas design asks for no weights of cell-centred
values (gas 5.3), so the grid provides none. Sand (A9) uses the same places and weights in its own module.

### 5.8 Storage, bound values and pieces (new; step question 4, cell resize and batches)

**No allocation per grid version.** `BuildGridBuffers` allocates, once, every array a grid of at most
`max_cell_blocks` blocks needs (the rows of 3.6). `BuildGrid` fills it, and the buffers move into the grid;
`TakeBuffers` gives it back for the next version. Value buffers are built once with room for the same
limit and bound to each new version by `ApplyGridVersion`, which only records the version and the block count:
the arrays keep their size and their contents become unspecified (the step and the cell resize design then fill them, by
`CellId` and `FacePieceId` or by remap). A bind runs on the CPU and launches nothing; on the GPU a version costs
the parallel build of 5.1 item 9 and the read of one status record, and no table crosses (A17). Raising the limit
is the only path that builds new grid buffers and values (`step::ApplyLimits`).

**One `GridBuilderBuffers`, lent (new, reconcile).** The builder buffers are needed only while a build or check runs,
so a caller with several grid buffers on one processor builds one `GridBuilderBuffers` and lends it to each with
`BuildGridBuffers(settings, builder_buffers)` (A31: every `GridBuffers` borrows one; none owns its own); one build or check uses it at a time, and a build into one set of grid buffers leaves a
live grid in other grid buffers unchanged.

**Room sized by the planner (new, reconcile).** The grid has no default room. `limits.max_cell_blocks` = the blocks
`step::BuildRunPlan` counts from the scene times `start_limit_headroom` (1.25), capped by the domain's blocks at the
smallest cell, at least 1. Above `kMaxCellBlocksLimit`, or when the estimate does not fit, the plan has
`fit = kDoesNotFit` and the budget prints the does-not-fit lines before the run. Outgrowing the room during a run is
`kTooManyCellBlocks` (the step's `kRaiseLimits`, then `ApplyLimits`).

**Pieces keep the fixed order.** `BuildGridPiece` lists the loaded blocks (the range and the border) in the
whole grid's index order. The range is a run of the whole grid, and every border block lies before or after
it, so the range is also a run of the piece, and a sum per destination visits its source blocks in the same
order as the whole grid does (cell resize and batches, 5.10). The piece's tables are the whole grid's, block by block:
`block_ids`, `cell_states` and `face_piece_kinds` are copied, neighbour indexes are translated to the piece
(`kNoIndex` for a block not loaded), and a `kBetweenCells` piece whose other cell lies in a block not loaded
becomes `kOutsidePiece`. A plane-4 place stays `kNotAPiece` when its high neighbour exists in the whole grid but
was not loaded, so no piece is created that the whole grid does not have, and every face place keeps its kind
and its place within its block. `FindCellAt` on a piece returns `kNoIndex` for a position in a block not loaded.
The piece map (`block_in_whole_grid`) is in the buffers; `range_in_whole_grid` and the map are what the batched
run's slice copies read. Both forms write every entry through shared rules (`ToPiecePlace`: border entries below
the range first, then the range, then the rest, so each entry's place is computed alone; `ToPieceNeighbour`:
binary search in the map, `kNoIndex` when not loaded; `ComputePieceFaceKind`). **On the GPU (A17)** the buffer form
needs no sort and no prefix sum (the border is already ascending): one thread per border entry checks it, one per
loaded block writes the map, one per piece block copies the id and translates its 36 neighbour entries, one per
cell place copies the state, one per face place writes the kinds (the same grid-stride launch as every other step); one status record
comes back. The map stays on the GPU (`block_in_whole_grid_buffer()`); a caller that needs it on the CPU reads it
back (4 B per loaded block).

### 5.9 The shared cell size rule (NEW, A41, A61–A63)

One rule decides the wanted width of every cell, read by stage 2 for new cells and by cell resize for existing
ones, so the two never disagree. Per active cell, `w = clamp(min(w_depth, w_solid, w_shear) x f, w_min,
w_max)` (A78, A79; was `w_artist` in the minimum): the depth term keeps 8 smallest cells below the surface and then grows linearly with depth
(`w_min max(1, depth / (8 w_min))`), the solid term likewise within 2 smallest cells of a solid, the shear term is
`c_u U_ref / |∇u|` (literature-cell-size.md 5(a); `c_u = 0.25` is a starting value to calibrate, A62; `U_ref =
max(u_max, sqrt(g w_max))`). CHANGED (A78, A79): the artist's wish is a refinement volume of cell size factors
built outside the solver by I/O (zones, optional camera, or a user volume); the physical width is multiplied by the
cell's smallest factor, `w = clamp(min(w_depth, w_solid, w_shear) x f, w_min, w_max)`, except in the cells within 8 smallest
widths of the surface (`depth < 8 w_min`, A61), where it stays `w_min`; the lookup reads one byte of a min-pyramid per cell
(`ComputeCellSizeFactorCode`), so a thin zone inside a big cell is never missed and both processors agree. Pockets and gas take
`w_min`. A new cell borders air, so its depth is 0 and its wanted width `w_min` (A61); `ComputeNewCellSizeLevel`
gives it the nearest committed cell's level plus one, capped at the smallest, which keeps the size-step rule
against the cell it grew from, and the next resize refines it to `w_min`. Levels come from comparing the wanted
width with `CellWidthsByLevel`, never from `log2`; every term is a local read and exact min and max, so CPU and GPU,
and any batch cut, give the same bytes. The merge hysteresis, the held gradient and the curvature term are cell
resize's (the last "later").

NEW (A66): the same rule builds the start grid. `step::BuildStartCells` (CPU, before any step) computes facts from
the start shapes alone, with the depth and solid distance at a cell's centre minus half its diagonal (`sqrt(3) w /
2`; both distances are 1-Lipschitz, so the value bounds every point of the cell from below and may be negative),
`velocity_change_size` 0 (a start region's velocity is uniform) and the first refinement key's factor over the
cell's box (A78, A79); it splits
every cell wider than `ComputeWantedCellWidth`, one level per pass, then closes the one-size-step rule
(`CheckSizeStepRule` passes on the result). The grid's first version is built from `StartCells::cells` by
`BuildGrid` like any version; `ComputeNewCellSizeLevel`'s "none found" case no longer decides a start grid (it
remains for a new cell far from every committed cell). Under A65 every liquid cell then holds about eight particles
of its own size (rest volume `(w/2)³`), so the batch plan's 512 particles per full block hold at every level.

## 6. CPU and GPU

- **Shared headers:** `grid_ids.h`, `grid_tables.h`, `weight_curve.h`, `nearby_face_weights.h`. All formulas
  of section 5 live there once. Sums run in list order with plain doubles; no fused multiply-add (Foundation
  flags), so CPU and GPU are expected to give the same bits.
- **A run is all CPU or all GPU (A17).** `BuildGrid`, `ComputeGridCounts` and `CheckSizeStepRule` (buffers
  forms), the buffer form of `FindBorderBlocks` and the buffer form of `BuildGridPiece` run on the processor of
  their buffers. The CPU builder (one thread, fixed order, no allocation) serves CPU runs and is the oracle the GPU
  tables are compared with bit for bit. The span forms of `FindBorderBlocks` and `BuildGridPiece`,
  `BuildRangeWithBorder`, and the one-argument `ComputeGridCounts` and `CheckSizeStepRule` are CPU only.
  `ApplyGridVersion` runs on the CPU for every grid. Stages loop `for block in range`, `for place in block`.
- **GPU layout:** the six arrays of `GridTables` and the builder buffers as device arrays sized for the buffers'
  `max_cell_blocks`, element types as declared, owned by the `GridBuffers` the `StaggeredGrid` holds (the builder
  buffers by the lent `GridBuilderBuffers`, reconcile, A31), alive for the whole run; the used part is rewritten per version by the GPU build. `CellValues`/`FaceValues` are single
  flat device arrays sized for the limit, owned by the module that built them, and stay on the device; only
  `ReadValues`/`WriteValues` (and the batched run's slice copies of `buffer()`) cross. Work is divided one thread
  per item (block, cell place or face place), or one thread per particle for weights; weights are computed in
  local memory (1,152 B per axis, 3,456 B when three axes are held at once; above the driver's default stack of
  1,024 B per thread, so each module whose GPU function calls the weights declares a `kDriverStack` entry as
  contact does; the grid's own GPU functions do not call them and have none) and never stored. Every GPU function of the build uses Foundation's `ComputeLaunchSize`: thread
  blocks of 256 threads, at most 65,535 of them, looping grid-stride.
- **Returned to the CPU:** one `GridBuildStatus` per grid operation on the GPU (counts, or the first failure);
  for border search, the count. No table, no block id. A stage's failure is its own compact record.
- **Per step for the GPU run (changed, A17):** the cell list is already on the GPU; `BuildGrid` builds the next
  version into the try grid buffers on the GPU and reads back the status record. No block id comes down, no table goes
  up, no field value moves. `ReadGrid` (456 B per block down) is the one explicit copy, for tests, tools and a
  caller that accepts the transfer (cell resize).
- **Batched run:** `BuildGridPiece` builds a small `StaggeredGrid` for a range plus its stacked border into one
  set of buffers of `max_loaded_blocks`, reused by every batch; on a GPU whole grid it builds the piece's tables on the
  GPU (buffer form), and the loaded values are bound to it with `ApplyGridVersion`. Stage code is unchanged
  because it only ever sees `(grid, range)`.

## 7. Tests

Analytic, CPU (label `cpu`), discrete results compared exactly:

| Test | Setup | Expected |
|---|---|---|
| `BuildGrid, OneBigCell_HasSixEdgeFacePieces` | 1 level, 1 cell, width 1 m | 6 pieces, each 1 m2, 3 `kGridEdgeLow`, 3 `kGridEdgeHigh` |
| `BuildGrid, SplitCellBesideBigCell_HasFourQuarterFacePieces` | 2 cells of 1 m, the +X one split | 4 `kBetweenCells` X-pieces of 0.25 m2 with the same `cell_low`; their areas sum to 1 m2; centre distance 0.75 m |
| `BuildGrid, AnyInputOrder_GivesSameTables` | 200 random valid grids, ids shuffled | identical bytes |
| `BuildGrid, EveryCellSide_IsCoveredOnce` | 200 random valid grids, up to 4 levels | for every active cell and side, piece areas sum to the cell's face area (exact: powers of two) |
| `BuildGrid, ClosedSurface_AreasCancel` | same grids | per cell, sum over pieces of `+-area * axis` is exactly zero |
| `CheckSizeStepRule, CornerTouchTwoLevelsApart_IsSizeStepRuleBroken` | level 0 cell and a level 2 cell touching at a corner | `kSizeStepRuleBroken`, both ids reported; same for edge and face |
| `BuildGrid, LoneSmallCell_IsSiblingCellMissing` | 7 of 8 siblings | `kSiblingCellMissing` |
| `CheckSizeStepRule, BadIdAfterRepeat_IsBadInputOfFirstBadId` NEW (A17) | input `{a, a, bad1, bad2}` | `kBadInput` naming `bad1` (step 1 outranks step 2; input order within step 1) |
| `ToCellSortKey, ThreePasses_GiveComesBeforeOrder` NEW (A17) | 10,000 random ids, levels 0..3 | the three stable passes give the `ComesBefore` order |
| `ComputeSplitCellCountToAdd, RandomGrids_AddEachSplitCellOnce` NEW (A17) | 200 random grids | added ancestors distinct, equal to the brute-force set of split cells and to the places the CPU build marks `kSplit` |
| `GridInputs, TwoLists_IsBadInput`, `..., BufferCountAboveSize_IsBadInput`, `ComputeGridCounts, CpuBufferList_GivesSameCountsAsSpan` NEW (A17) | the two list forms | as named |
| `ComputeGridCounts, IntoBuffers_EqualsSpanForm` NEW (A17; renamed, A31) | 200 random grids, CPU buffers for 4,096 blocks; a grid of 4,097 blocks | equal counts; `kTooManyCellBlocks` with `cell` zero while the one-argument form gives `cell_blocks == 4097` |
| `CheckSizeStepRule, IntoBuffers_EqualsSpanForm` NEW (A17) | the corner, edge and face lists and 200 random grids | identical results |
| `ComputeGridCounts, IntoBuffers_AllocatesNothing` NEW (A17) | 100 grids through both buffers forms | allocation counter stays 0 |
| `BuildGrid, EmptyList_GivesEmptyGrid` NEW | empty list, version 1 | zero blocks, `all_blocks() == {0, 0}` (both `ComputeGridCounts` forms give zero counts) |
| `ReadGrid, CpuGrid_CopiesIntoOtherCpuBuffers`, `ReadGrid, BuffersTooSmall_IsTooManyCellBlocks`, `StaggeredGrid, CpuGrid_HasNoGpuTables` NEW (A17) | CPU grids | same tables, version and counts; `kTooManyCellBlocks`; `tables(kGpu)` version 0, null arrays |
| `FindCellAt, PositionOnLowSide_BelongsToCell` / `OnHighSide_BelongsToNext` | exact binary positions | exact |
| `ToCellId, RoundTrip_AtCoordLimits` | `+-kMaxCoord`, levels 0 and 15 | exact |
| `ComputeWeightCurve, AnyPosition_ThreeNeighboursSumToOne` | 1,000 offsets | `|sum - 1| <= 4e-16`; slopes sum `<= 4e-16` |
| `ComputeNearbyFaceWeights, CompleteSameSize_MatchesClosedForm` | uniform 8^3, 1,000 positions, all kept | 27 faces; `W = 1`, `mean_offset = 0`, `weighted_spread = h^2/4 I` within `1e-14 h^2`; `is_complete` |
| `..., AnyKeptSet_WeightsSumToOne` | random mixed grids, random masks | `|sum weight - 1| <= 1e-14`, `|sum weight_change| <= 1e-13 / h` |
| `..., LinearField_IsReproduced` | value `a + b.x` on faces, mixed sizes and masks, `fit == kFull` | value at particle within `1e-13 (|a| + |b| h)`; fitted row equals `b` within `1e-11 |b|` |
| `..., WeightChange_MatchesFiniteDifference` | centred difference, step `1e-6 h`, away from cell sides | within `1e-7 / h` |
| `..., FacesOnOnePlane_IsMeanOnly` | mask keeps one plane | `kMeanOnly`; none kept: `kNone` |
| `FindNearbyFaces, AllNeighboursSplit_Has76`, `..., AllNeighboursSplitEveryWindow_Has76`, `..., TwoSizesWorstCase_Has124`, `..., SmallRandomGrids_NeverExceedTheProvenBound`; `ComputeNearbyFaceWeights, TwoSizesWorstCase_WeightsSumToOne` (owner decision 2026-11-19) | all 26 neighbours split, every window; level-1 cell `{2,2,2}` of a 6^3 box with 14 neighbours kept, position `{4.75, 4.25, 4.75}` m, Y axis; 200 small random grids, every cell and window | 76; 124 and weights summing to 1; never above 212, largest count reported |
| `FindNearbyFaces, UniformGridOffCentre_Has27`, `..., UniformGridAtCellCentre_Has27`, `..., UniformGridOnCellSide_Has27`, `..., UniformGridAtCellCorner_Has27`; `ComputeNearbyFaceWeights, AtExactCellCentre_IsCompleteWithNineZeroWeights`, `..., OnCellSide_IsCompleteWithNineZeroWeights`, `..., AtCellCorner_IsCompleteWithFifteenZeroWeights` (owner decision 2026-11-19) | uniform 5^3, positions off the centre, at the centre, on a low side, at a low corner | 27 per axis each; `is_complete`; 0, 9, 9 (cross axes) and 15 faces of weight 0; `W = 1` |
| `FindNearbyFaces, CellCentreInNonDyadicFrame_Has27`, `..., OneStepBelowCellSide_Has27`, `..., NonDyadicFrameAtSidesAndCentres_AlwaysHas27`; `ComputeNearbyFaceWeights, NonDyadicFrameAtSidesAndCentres_IsComplete` (ruling 2026-11-19) | origin -17.616723516683763, width 0.0375, a cell centre; origin -0.5, width 0.0625, x 1.6249999999999998; positions within 3 rounding steps of sides and centres | 27 per axis; `is_complete` |
| `FindNearbyFaces, RangeWithBorder_HoldsEveryFace` | random grids cut by `BuildRanges` | every listed face is stored in the range or its border |
| `FindBorderBlocks, TouchingCells_EqualsGeometricReadSet`, `..., FourCellWidths_EqualsBlockRuleByCoordinates`, `..., FourCellWidths_HoldsEveryFaceWithinFourWidths` NEW | random grids, random ascending block sets, both widths | equals the test-only geometric checker; ascending, no repeats, no block of the set; the span form gives the same list and the full count when the span is short |
| `FindBorderBlocks, PlaneFourPieceBeyondTouchingCell_IsInBorder` NEW | range ends at a block side; the touching cell's high face is stored in the next block (5.1 step 5) | that block is in the `kTouchingCells` border |
| `FindBorderBlocks, StackedBorders_HoldEveryRead` NEW | random grids, the pass-1 stack (touching, touching, four widths, touching) | every face and cell a stage of the stack reads from its run set is loaded |
| `FindBorderBlocks, StackedBufferForm_EqualsSpanFormLayerByLayer` NEW (A17) | 100 random grids, ranges of 4 blocks, the pass-1 layers, CPU buffers | at each layer the buffer form's list equals `std::merge` of `border_so_far` and the span form on range plus `border_so_far` |
| `FindBorderBlocks, StackedBufferFormShortOutput_WritesFirstEntriesReturnsTotal`, `..., StackedBufferFormSameBufferTwice_IsBadInput`, `..., BorderSoFarInsideRange_IsBadInput`, `..., StackedBufferForm_AllocatesNothing` NEW (A17) | as named | as named |
| `BuildBorderSearchMemoryEstimate, MatchesAllocatedBytes` NEW (A17) | CPU search buffers | `4 * max_cell_blocks + 4` |
| `ComputeIsBorderBlock, PlaneFourFixture_MarksBlockC` NEW (A17) | the plane-4 fixture built by hand (shared-formula test) | block C marked |
| `BuildGridPiece, BufferForm_EqualsSpanForm` NEW (A17) | the grids of `KeepsWholeGridOrder`, border from the stacked buffer form on CPU buffers | same tables, map and counts as the span form |
| `BuildGridPiece, SameWeightsAsWholeGrid` | positions in the range | identical bytes |
| `BuildGridPiece, KeepsWholeGridOrder` NEW | random grids, ranges, stacked borders | `block_in_whole_grid` ascending; `range` is a run; per block, states and kinds equal the whole grid's except `kOutsidePiece` exactly where the other cell's block was not loaded |
| `BuildGridPiece, BlockSumsInPieceOrder_EqualWholeGrid` NEW | per-block partials summed with `math::TreeSum` over pieces from 3 different cuts | identical bytes to the one-piece sum |
| `BuildGrid, IntoBuffers_AllocatesNothing` NEW | buffers for 4,096 blocks; 100 random grids built and their buffers taken back in turn; allocations counted through a replaced global `operator new` in the test executable (plan decision) | zero allocations after `BuildGridBuffers`; buffer addresses unchanged; tables equal those of `BuildTestGrid` byte for byte |
| `BuildGrid, AboveBuffersRoom_IsTooManyCellBlocks` NEW | buffers for 8 blocks, a 9-block grid | `kTooManyCellBlocks` with `cell` zero, no grid; the same grid buffers then builds an 8-block grid |
| `GridInputs, VersionZero_IsBadInput` NEW | `version = 0` | `kBadInput` |
| `ApplyGridVersion, RebindsWithoutAllocating` NEW | each value type with room for 4,096 blocks, bound to 100 versions | no allocation; `grid_version()`, `block_count()` follow the grid; above the room `kTooManyCellBlocks` and the values unchanged |
| `ReadValues, NotBound_IsWrongGridVersion`, `..., SpanOneEntryShort_IsBadInput` NEW | values bound to no grid; a span one entry short | `kWrongGridVersion`; `kBadInput` |
| `ToName, EveryError_IsStableSnakeCase` NEW | every `GridError` | lower snake case of the enumerator without `k`, unique; `ToName(kTooManyCellBlocks) == "too_many_cell_blocks"` |
| `BuildGridMemoryEstimate, CpuGrid_IsNineHundredSeventyTwoBytesPerBlock`, `BuildGridBuffers, CpuBuffers_HoldTheEstimatedBytes` | `BuildGridBuilderBuffers` and `BuildGridBuffers` on the CPU at 1,000 blocks | `1000 * 972`; with `where`, `kRunOn` `1000 * 456`, `kCpu` `1000 * 516` |
| `BuildGridMemoryEstimate, GpuRun_KeepsNothingOnTheCpu` NEW (A17) | `BuildGridMemoryEstimate(kGpu, 1, 1000)` at 1,000 blocks | `kCpu` 0; `kRunOn` `1000 * (1996 + 64 * bytes_per_item) + fixed_bytes + bytes_per_item + 8 + sizeof(GridBuildStatus)` |
| `BuildGridMemoryEstimate, NoEntryIsACpuCopy`, `..., TwoGrids_DoublesEveryEntry` NEW (A17) | both processors | no row named a CPU copy; `copies == 2`, totals per `where` double |
| `BuildGridBuffers, ZeroBlocks_IsBadInput`, `..., AboveLimit_IsBadInput` NEW (A17) | 0 and `kMaxCellBlocksLimit + 1` | `kBadInput` |
| `BuildGridMemoryEstimate, SharedBuilderBuffers_CountedOnce` NEW (reconcile) | both processors, 2 grids, `kOneForAllGrids` | every `builder buffers:` row `copies == 1`, every other row `copies == 2`; GPU `kRunOn` at 1,000 blocks `1000 * (2 * 460 + 1920) + 4096 + 6 + 8 + 2 * sizeof(GridBuildStatus)` |
| `BuildGridBuffers, TwoBuffersShareBuilderBuffers_HoldTheSharedEstimate` NEW (reconcile) | `BuildGridBuilderBuffers(kCpu, 1000)`, two `BuildGridBuffers({.max_cell_blocks = 1000}, builder_buffers)` | bytes of the builder buffers and both buffers equal `ComputeMemoryBytes(BuildGridMemoryEstimate(kCpu, 2, 1000, kOneForAllGrids))` at 1,000 blocks; neither `GridBuffers` holds a builder buffers array |
| `BuildGridBuffers, LentBuilderBuffersTooSmall_IsBadInput` NEW (reconcile; other-processor half dropped, A30) | builder buffers for 500, buffers for 1,000 | `kBadInput`, nothing allocated |
| `BuildGridBuilderBuffers, ZeroBlocks_IsBadInput`, `BuildGridBuffers, ZeroBlocks_IsBadInput` NEW (reconcile) | 0; `GridSettings{}` | `kBadInput` |
| `BuildGrid, SharedBuilderBuffers_OtherLiveGridUnchanged` NEW (reconcile) | one `GridBuilderBuffers` for 4,096 blocks lent to two buffers; grid A in the first, 100 random grids built (buffers taken back) in turn in the second (and both buffers-form checks) | A's tables, counts and `FindCellAt` on 1,000 positions byte-identical before and after; each B equals B built with its own builder buffers |
| `BuildTestGrid, DefaultSettings_SizedToTheList` NEW (reconcile; was `BuildGrid, ...`, A31) | `BuildTestGrid(inputs, GridSettings{})` of a 9-block list | a grid with `max_cell_blocks() == 9` |

**Cell size rule (NEW, A41, A61–A63; grid plan Task 13a, label `cpu`; widths 1, 0.5, 0.25, 0.125, 0.0625 m,
smallest level 4, far facts unless named):**

| Test | Setup | Expected |
|---|---|---|
| `ComputeCellWidthsByLevel, EqualsComputeCellWidthPerLevel` | frame 1 m, 5 levels | bit for bit; 0 beyond |
| `ComputeWantedCellWidth, SurfaceCell_IsSmallest` | depth 0, also with distance 10, factor 4 (A78) | 0.0625 exactly |
| `ComputeWantedCellWidth, UniformTranslation_NoShearSplit` | gradient 0, `U_ref` 5 | 1.0, level 0 |
| `ComputeWantedCellWidth, NothingWithinReach_IsBiggestWidth` | gradient 1e-300; depth `DBL_MAX` | 1.0, never NaN |
| `ComputeWantedCellWidth, DepthBeyondReach_GrowsLinearly`; `..., SolidBeyondReach_GrowsLinearly` | depth 1.0; 0.25; 100; then distance 0.5; distance 0.5 with depth 1.0 | 0.125; 0.0625; 1.0; then 0.25; 0.125 |
| `ComputeWantedCellWidth, ShearTerm_IsShearFactorTimesSpeedOverVelocityChange` | frame 0.5 m, 6 levels, gradient `sqrt(2) x 10`, `U_ref` 1 | 0.0176776695296637 within 1e-15 relative; level 5 |
| `ComputeWantedCellWidth, PocketOrGas_IsSmallest` | flags with depth 100 | 0.0625 |
| `ComputeWantedCellWidth, FactorOne_SameBitsAsNoFactor`, `..., FactorHalf_HalvesDeepWidth`, `..., FactorTwo_DoublesDeepWidth`, `..., FactorTwoNearSurface_IsSmallest` (A78, A79) | factor 1 over 10,000 fact sets; depth 1.0 with factor 0.5; with 2; depth 0.25 with 2 | bit for bit; 0.0625; 0.25; 0.0625 |
| `ComputeWantedSizeLevel, WidthEqualToALevel_GivesThatLevel`, `..., BelowSmallestWidth_GivesSmallestLevel` | 0.125, `nextafter(0.125, 0)`, 0.2, 5; 0.01 | 3, 4, 3, 0; 4 |
| `ComputeReferenceSpeed, SlowLiquid_UsesGravityWaveSpeed` | (0.1, 9.81, 1); (5, 9.81, 1) | `sqrt(9.81)` bit for bit; 5 |
| `ComputeCellSizeFactorTable, QuarterDoublings_AreExact` (A78) | | `factor[24] == 1`, `factor[28] == 2`, `factor[20] == 0.5`, `factor[0] == 0x1p-6` exactly |
| `ComputeCellSizeFactorCode, NoVolume_IsZero`, `..., SmallCell_ReadsItsSample`, `..., BigCell_IsSmallestCodeOverItsBox`, `..., OutsideTheVolume_IsZero` (A78, A79) | frame 1 m, 5 levels, spacing 4 smallest cells (8 x 4 x 4 samples), a one-sample-wide line of code -8 at sample x 5 | 0; the sample's code; -8 for the level-0 cell holding x 5, 0 for the other; 0 |
| `ComputeNewCellSizeLevel, NextToBigCell_IsOneLevelSmaller` | level 0 cell beside 8 level-1 cells, smallest 2 | 1 beside level 0; 2 beside level 1; capped at 1 when smallest is 1; 2 when nothing is near |
| `ComputeWantedCellWidth, NegativeDepthOrDistance_IsSmallest` (NEW, A66) | depth `-0.03` alone; distance `-0.03` alone; depth `-0.03` with factor 4 | 0.0625; 0.0625; 0.0625, never NaN |
| `ComputeWantedCellWidth, OnGpu_SameBytesAsCpu` (`gpu`) | 10,000 fact sets from a fixed formula | identical bytes, levels and factor codes (`ComputeCellSizeFactorCode` on the fixture above) |
| `ComputeNewCellSizeLevel, OnGpu_SameBytesAsCpu` (`gpu`) | random 3-level grid on both processors, 10,000 positions | identical levels |

**Reservation (NEW, A58; label `cpu` unless named):** `BuildGridBuffers, IntoRunMemory_HandsOutExactlyTheEstimate`
(1,000 blocks, both sharings, CPU block and GPU piece list: handed out == reserved),
`BuildGridBuffers, RunMemoryOnePieceShort_IsOutOfMemory`, `BuildFaceValues, IntoRunMemory_TakesOnePieceOfPlacesTimesSize`,
`BuildCellValues, RunMemoryTooSmall_IsOutOfMemory` (`magnitude` 512,000), `BuildBorderSearchBuffers,
IntoRunMemory_HandsOutExactlyTheEstimate` (1,000 blocks), `GridError,
OutOfMemory_IsTheLastValue`; `gpu`: `BuildGridBuffers, OnGpuIntoRunMemory_HandsOutExactlyTheEstimate`,
`BuildBorderSearchBuffers, OnGpuIntoRunMemory_HandsOutExactlyTheEstimate`. The GPU free-byte tests measure the drop
across `BuildRunMemory`.

The planner side of the room (reconcile item 11) is tested in the step plan: Task 8 `BuildRunPlan,
BlockRoomFromSceneNotFromDefault`, `BuildStepMemoryEstimate, TwoGridBuffersShareBuilderBuffers`, `BuildRunPlan,
RoomAboveGridLimit_DoesNotFit`; Task 9 `BuildSimulation, GridBuffersShareBuilderBuffers`.

**Reference and proofs:** no Python piece and no Lean proof. The uncertainty here is discrete (does the table
walk find exactly the geometric set of pieces and neighbours?). A test-only C++ checker that scans all cells
geometrically, with no blocks and no tables, removes it and runs on every platform; it backs the "random
grids" tests above. `proofs/Fluidity/UniformTile.lean` (the 8^3 tile) describes machinery that is gone.

**Comparison with `main`** (supporting only): the 27 weights, offsets and derivatives of
`main`'s Python reference `quadratic_stencil` (`reference/src/fluidity_reference/basis.py`, the formula of its
C++ `BuildMacFaceGeometryStencil3D`; read with `git show`, nothing of `main` is built) for 100 positions,
rotated to Z-up, against the complete-neighbourhood case within `1e-15 + 1e-13 |value|`; piece count and areas of `test_adaptive_mac_topology_3d.cc`'s two-size fixture with
all 8 siblings present. The radial weights of `main` are not compared: the curve changed on purpose.

**GPU** (label `gpu`; every grid built twice from one list, on the CPU as the reference and on the GPU from a GPU
copy of the list made in the test's setup; GPU tables read with `ReadGrid`). Changed (A17): the tables are built
on the GPU, not uploaded.

| Test | Expected |
|---|---|
| `CheckSizeStepRule, OnGpu_EqualsCpu` | the corner lists, `OneLevelApart`, the empty list, 200 shuffled random grids: identical results |
| `CheckSizeStepRule, OnGpuSeveralFailures_SameFirstFailureAsCpu` | 500 random grids with 1 to 4 mutations (repeat, drop, add a big cell, add a two-levels-smaller cell with siblings, clear bit 63, coordinate `kMaxCoord + 1`), with and without the sibling check: `GridFailure` equal to the CPU's |
| `CheckSizeStepRule, OnGpuTwice_GivesIdenticalFailure`, `..., OnGpu_AllocatesNothing` (A30 removed `OnGpuBuffersWithCpuList_IsBadInput` and `ComputeGridCounts, GpuBufferListInOneArgumentForm_IsBadInput`) | as named (GPU free bytes unchanged after the first call) |
| `BuildGrid, OnGpu_EqualsCpuTablesBitForBit` | empty list, one cell, the two-size fixture, 200 shuffled random grids into GPU buffers for 4,096 blocks: version, counts and the used part of all five tables equal |
| `BuildGrid, OnGpuFailures_EqualCpuFailures` | the 500 mutated lists, 9 blocks into room for 8 (step 6), `64 * 8 + 1` cells (step 5): identical `GridFailure` |
| `BuildGrid, OnGpuFailedBuildThenRebuild_EqualsFreshBuffers`, `BuildGrid, RebuildIntoUsedGpuBuffers_EqualsFreshBuffers` | the second grid equals the same grid built into fresh buffers |
| `BuildGrid, IntoGpuBuffers_AllocatesNothing`, `BuildGrid, OnGpuTwice_GivesIdenticalBytes`; `ComputeGridCounts, OnGpu_EqualsCpu` | free bytes unchanged, `operator new` counter 0; identical bytes; equal counts |
| `ComputeMemoryBytes, MatchesAllocatedGpuBytes` | the drop in GPU free bytes equals the `kRunOn` rows of `BuildGridMemoryEstimate(kGpu, 1, 1000)` at 1,000 blocks for one `GridBuilderBuffers` and one `GridBuffers` (up to the driver's stated granularity); no CPU array allocated |
| `BuildGrid, OnGpuSharedBuilderBuffers_OtherLiveGridUnchanged` NEW (reconcile) | the CPU shared builder buffers test on GPU buffers, compared through `ReadGrid`; the drop in free GPU bytes for one `GridBuilderBuffers` and two buffers equals `ComputeMemoryBytes(BuildGridMemoryEstimate(kGpu, 2, 1000, kOneForAllGrids), counts, kRunOn)` (granularity as above) |
| `ComputeNearbyFaceWeights, OnGpu_MatchesCpu`, `..., OnGpuTwice_GivesIdenticalBytes`; `FindCornerPlaces, OnGpu_EqualsCpu` | weights for 10,000 positions on random mixed grids within `|difference| <= 1e-15 + 1e-13 * |value|` (offsets: times `h`), discrete fields exact; identical bytes; corner places identical |
| `WriteValues, OnGpuThenReadValues_ReturnsTheSameValues`; `ApplyGridVersion, OnGpuGrid_LaunchesNothing` | `ReadValues(WriteValues(x)) == x`; 100 GPU grids bound in turn: free bytes and contents unchanged |
| `FindBorderBlocks, OnGpu_EqualsCpuStackedLayers`, `..., OnGpuShortOutput_SameCountAsCpu`, `..., OnGpuFailures_EqualCpuFailures`, `..., OnGpu_AllocatesNothing`, `..., OnGpuTwice_GivesIdenticalBytes`; `BuildBorderSearchMemoryEstimate, MatchesAllocatedGpuBytes` | lists and counts equal the CPU's layer by layer; identical failures |
| `BuildGridPiece, OnGpu_EqualsCpuPiece`, `..., OnGpuFailures_EqualCpuFailures`, `..., OnGpu_AllocatesNothing`, `..., OnGpuTwice_GivesIdenticalBytes` (A30 removed `SpanFormOnGpuWhole_IsBadInput`) | piece tables, map, version and counts equal the CPU piece's byte for byte |
| `StaggeredGrid, SlopedSurfaceInBoxOnGpu_EqualsCpu` | the box of liquid with a sloped surface (below): GPU grid, stacked borders and pieces equal the CPU's; weights on the GPU piece within the CPU-versus-GPU tolerance |

**Runs in the container without a GPU** (label `no_gpu`): `BuildGridBuilderBuffers`, `BuildBorderSearchBuffers` and
`BuildFaceValues` with `run_on = kGpu` return `kRunOnCallFailed` and nothing;
every `cpu` test, since the shared headers are the same code the
GPU kernels compile.

**Scene** (label `cpu`): `StaggeredGrid, SlopedSurfaceInBox_PassesEveryCheck`: width 1 m, 3 levels, level-0 box
16 x 4 x 8, level-2 cells within 1 m of `z = 4 + 0.25 x` with the size-step closure; counts equal
`ComputeGridCounts`; ranges of 8 blocks, a piece with a stacked touching border per range; at 1,000 positions,
piece weights equal whole-grid weights byte for byte and sum to 1 within `1e-14`.

## 8. Method paper section

Labels: `<sec-grid>`, `<sec-cell-sizes>`, `<sec-face-pieces>`, `<sec-cell-blocks>`, `<sec-nearby-faces>`,
`<eq:weight-curve>`, `<eq:scaled-weights>`, `<eq:weighted-spread>`, `<fig-staggered-grid>`,
`<fig-face-pieces>`, `<fig-nearby-faces>`, `<proc-build-grid>`, `<proc-nearby-face-weights>`, and (A17)
`<sec-parallel-build>`, `<fig-parallel-build>`, `<proc-build-grid-in-parallel>`. Adds the `cell-grid(...)` helper
Foundation left open in `figure_style.typ`.

Outline: 1 where values live (cells, faces, +axis orientation); 2 cells of mixed sizes, size levels, the
size-step rule and why one level; 3 face pieces: one identity, one area, stored with the smaller cell; 4 cell
blocks, the fixed order, id versus index, grid versions, ranges and border; 5 the weight curve; 6 nearby
faces: candidates, leaving faces out, scaling, mean, spread, the three fit outcomes, and the complete case as
a special case; 7 memory per block (with the GPU build rows); 8 (A17) building the tables in parallel: a run is
all CPU or all GPU, each sequential step becomes "key, stable sort, mark, prefix-sum, scatter" (example: blocks
`[A, A, B, C, C, C]` give marks `[1, 0, 1, 1, 0, 0]` and exclusive sums `[0, 1, 1, 2, 3, 3]`), the fixed order kept
because the sort keys are the fixed order, the first failure as the smallest `(step, position)` key, and why the
tables equal the CPU's (each entry is a function of the sorted list, not of the order work was done).

`<fig-parallel-build>`: six cells of two sizes above four rows of boxes (block keys in input order, after the
stable sort, the marks, the exclusive sums), an arrow from each marked entry to its block index.
`<proc-build-grid-in-parallel>`: steps P1 to P12 of 5.1 item 9, each naming its one-thread-per-what.

Figures: `<fig-staggered-grid>` one cell in the Z-up triad with pressure at the centre and the three face
values, and a 2D slice showing `cell_low`/`cell_high`. `<fig-face-pieces>` a big cell beside four small
cells: the big face drawn as four pieces, each with its arrow, plus a forbidden two-level jump crossed out.
`<fig-nearby-faces>` a 2D slice of a particle near a size change and a plate: kept faces sized by weight,
left-out faces hollow, the mean offset arrow and the spread ellipse, beside the complete 3 x 3 case.

NEW (A41, A61–A63): `<sec-cell-size-rule>` with `<eq:wanted-cell-width>` (5.9): the four terms, the clamp, `c_u`
as a starting value, `U_ref`, the new-cell rule, the artist inputs, levels by comparison.

Pseudocode: `<proc-build-grid>` (steps 1 to 8 of 5.1 item 8, with a closing line "on the GPU, see
`<proc-build-grid-in-parallel>`"), `<proc-count-grid>` (the four steps of the count without tables, 3.3) and `<proc-nearby-face-weights>` (the 9 steps of 5.4).

Cite from the old list: [2] Ding 2020 (weights per face axis), [1] Jiang 2015 (the fit and its spread), [3] Hu
2018 (leaving out faces across a cut), [18] Stomakhin 2014 (all materials on one staggered grid), [5] Wang
2020 and [15] Fei 2021 (sparse blocks), [6] Wang 2026 (cells of mixed sizes). Harlow and Welch 1965 is not in
the old list: cited by author and year only (decision 11).

## 9. Vocabulary

| Old name on `main` | New name | Source |
|---|---|---|
| `SparseMacGrid3D`, `AdaptiveMacTopology3D`, `UniformMacGeometry3D` | `StaggeredGrid`, `GridFrame` (`staggered_grid.h`) | DECIDED (`StaggeredGrid`, `grid`) |
| `CellIndex3D`, `AdaptiveMacCellKey3D` | `CellCoord` + `SizeLevel`, `CellId` | DECIDED / NEW |
| `width_exponent` (0 = smallest) | `size_level` (0 = biggest) | DECIDED; reversed, VERIFIED in `adaptive_mac_topology_3d.h:14-19` |
| leaf, `AdaptiveMacLeaf3D` | active cell, `CellState::kActive` | DECIDED |
| parent, children, family | `big_cell`, `small_cells`, `sibling_cells`; `cell_eighth` | DECIDED |
| 2:1 grading | `size_step_rule` | DECIDED |
| subface, `AdaptiveMacInteriorFace3D`, `AdaptiveMacExteriorFace3D` | face piece, `FacePieceKind`, `FacePieceId` | DECIDED / NEW |
| `tail_cell_ordinal`, `head_cell_ordinal`, `cell_is_tail` | `cell_low`, `cell_high`; `kGridEdgeLow/High` | DECIDED / NEW |
| `face_slot`, ordinal | `FacePlaceIndex`, `CellPlaceIndex`, `CornerPlaceIndex`, `CellBlockIndex` | DECIDED (`index`; `place` A3) |
| (none) | `place` (`cell_place`, `face_place`, `corner_place`): a fixed position in a cell block that may or may not hold an active cell, a face piece or a corner | DECIDED (A3). Glossary row |
| arena, capacity of the topology | `GridBuffers` (`BuildGridBuffers`, `TakeBuffers`); values bound with `ApplyGridVersion` | NEW |
| (none) | `GridBuilderBuffers` (`BuildGridBuilderBuffers`), `BuilderBuffersSharing`: the builder buffers lent to several `GridBuffers` | NEW (reconcile) |
| replaced 2026-11-14 (A29, A31): `ReleaseBuffers`, `GridScratch`, `BuildGridScratch`, `GridScratchSize`, `ComputeGridScratchSize`, `ScratchSharing { kOnePerBuffers, kOneForAllBuffers }`, `scratch_sharing`, "build scratch", the allocating `BuildGrid(inputs, settings)` | `TakeBuffers`, `GridBuilderBuffers`, `BuildGridBuilderBuffers`, `GridBuilderBuffersSize`, `ComputeGridBuilderBuffersSize`, `BuilderBuffersSharing { kOnePerGrid, kOneForAllGrids }`, `builder_buffers_sharing`, "builder buffers" (prose: "working buffers"), test-only `BuildTestGrid` | owner answers A29, A31 |
| `cell_resize::CellSizeSettings`, `cell_resize::CellSizeFacts`, `cell_resize::ComputeWantedCellWidth`, `cell_resize::ChooseCellSize`, `step::StepCallbacks::choose_cell_size` | `grid::CellSizeSettings`, `grid::CellSizeFacts`, `grid::ComputeWantedCellWidth` (moved); the callback is removed: stage 2 calls `grid::ComputeNewCellSizeLevel` | owner answers A41, A61; `plan-decisions.md` item 2 |
| (none) | `CellWidthsByLevel`, `ComputeCellWidthsByLevel`, `ComputeReferenceSpeed`, `ComputeWantedSizeLevel`, `ComputeNewCellSizeLevel`, (A78, A79) `CellSizeFactorTable`, `ComputeCellSizeFactorTable`, `RefinementFactors`, `ComputeCellSizeFactorCode`; glossary rows `wanted cell width`, `reference speed`, `cell size factor`, `refinement volume` | NEW (A41, A61–A63; A78, A79 replaced `CellSizeRegion`, `ArtistCellSizes`, `ComputeArtistSizeLevel`) |
| memory rows `prefix sum ends`, `order buffers, fixed part`, `border marks end`, `border order buffers, fixed part` | folded into the one entry of their array (`order`, `second order`, `order buffers`, `border marks`, `border order buffers`) | A58, reconcile ruling |
| `kGridMemoryEstimate` (first draft) | `BuildGridMemoryEstimate` | NEW (every module's `Build<Module>MemoryEstimate`) |
| `BlockIndex3D` | `CellBlockId` | DECIDED (`cell_block`, `id`) |
| `face_area`, `cell_center_distance`, `world_anchor` | `ComputeFaceArea`, `ComputeCellCenterDistance`, `ComputeFacePosition` | NEW |
| topology epoch | `GridVersion` | NEW (pattern of `state_version`) |
| halo | border cells, `RangeWithBorder` | DECIDED |
| `MacFaceStencil3D`, support | `NearbyFaceWeights`, `NearbyFaceList` | DECIDED / NEW |
| `MacFaceStencilSample3D` | `FaceWeight` | DECIDED |
| quadratic basis, kernel | `weight_curve`, `ComputeWeightCurve` | DECIDED |
| `weight_gradient` | `weight_change` | NEW |
| `face_offset_from_particle` | `offset` | NEW |
| `mean_offset` | `mean_offset` | VERIFIED: first weighted sum of offsets, `graded_mac_transfer_3d.cc:127,206,228` |
| `covariance`, `centered_covariance`, paper `D'` | `weighted_spread` | VERIFIED: `graded_mac_transfer_3d.cc:131,257` is `sum weight * centred * centred^T`. Units change from cell-widths squared to m2. |
| `full_rank`, `affine_enabled`, `uses_low_mass` | `NearbyFaceFit` | NEW |
| masked, `structurally_eligible`, same-side filter | kept, `NearbyFaceMask` | NEW |
| `MacOwnerBlock3D`, `owner_slot` | "the block that stores" (`ToCellBlockId`) | see below |

**`moment` (Verify).** Only one of its uses is the spread. Checked: (a) centred second sum of offsets =
`weighted_spread` (above; also `moment_xx/xy/yy` in the 2D `transfers.cc:525-539`, not ported); (b)
`normalized_offset_velocity_moment` (`mac_gather_3d_internal.h:46`) and `velocity_covariance` are
`sum weight * value * offset`: a different quantity, named by transfer (suggest
`weighted_value_times_offset`); (c) `GradedMacSupportMomentRow3D` bundles weight sum, mean and spread, now
`NearbyFaceWeights`; (d) `moment` in `rigid_body_3d` is a turning force or inertia, and in
`liquid_isolated_drop_3d.cc:180` a mass-weighted particle spread: both belong to other modules. A blanket
rename `moment -> weighted_spread` would be wrong in (b) and (d).

**`owner` (Decided, O5, decision 3).** Five meanings found: (1) the block that stores a face (`MacOwnerBlock3D`,
`owner_slot`); (2) the one cell of a border piece (`AdaptiveMacExteriorFace3D`); (3) which kind of region
holds the liquid (`AdaptiveLiquidMaterialOwner3D { kParticles, kEulerian }`); (4) region transfer (`owner
exchange`, decided); (5) C++ memory ownership. The grid needs no identifier containing `owner` and uses none.

## 10. Not ported

- `MacFaceStencilSample3D` arrays and every stored per-particle weight record (`mac_types_3d.h`,
  `host/sparse_mac_grid_3d.h`): weights are computed when used.
- Rounding recovery: `NormalizedCoordinate`, `SubtractionRoundoff`, `QuadraticSplineFactor::Mantissa`,
  `WithinHalfCell` (`sparse_mac_grid_3d.cc:83-336`). One finite check at the module boundary replaces them.
- The 8^3 transfer tile, `ComputeParticleSortBin3D`, the half-cell motion rule, `ParticleBinIndex3D`.
- `SparseMacGridView3D`: a `StaggeredGrid` is passed by reference; `GridTables` is the non-owning form.
- Allocation of the grid tables and of grid fields per grid version: replaced by `GridBuffers` and
  `ApplyGridVersion`.
- The limits 4,096 cells and 24,576 faces; replaced by `GridSettings::max_cell_blocks`.
- The radial weight and `kSupportRadiusInLeafWidths = 2` (`graded_mac_transfer_3d.cc:18,171-176`); the
  16-face bounding tree (`:37-123`), replaced by block neighbour tables.
- `GradedQuadraticMacTransfer3D` (same-size islands): the complete case is no longer a separate path.
- `coarse_fine_transfer_math_3d.h`, `host/coarse_fine_transfer_3d.h`, `native/cuda/coarse_fine_transfer_3d.cu`:
  the cubic-normal hierarchical experiment.
- Quarter border pieces beside a lone small cell (`AppendExteriorFaces`): siblings now exist together.
- `solid_upper` in particle containment; `MeasureSupportMoments`, `CheckUniformResamplingSupport`,
  `CheckCompleteRadialSupport` (their facts are fields of `NearbyFaceWeights`).
- `reference/src/fluidity_reference/grid.py` (2D), `basis.py`, `host/grid.h`, `host/basis.h`,
  `proofs/Fluidity/UniformTile.lean`; `docs/adaptivity/` milestone notes.
- `RefinementClosure`: moves to the cell resize and batches design, built on `CheckSizeStepRule` and `FindTouchingCells`.

## 11. Decisions

Each question keeps its text. "Recommended option" means: recommended option (owner's standing instruction,
2026-11-14). The owner question numbers (Qn) are those of `owner-questions.md`.

1. **Raw weight where cell sizes mix.** `main` has two curves that disagree. (A, recommended) the uniform
   curve measured in the particle's cell width, times the face's area share, then scaled:
   `w = Q(dx) Q(dy) Q(dz) * area / h^2`. Same-size regions behave exactly as the uniform solver; one code
   path; but the mixed-size numbers are new and untested on `main`. (B) keep the radial curve
   `((1 - r^2/4) * width/h)^2` everywhere: tested on `main` for mixed sizes, but changes uniform results and
   has no clean derivative. (C) allow particles only where all 27 faces are same-size and reject the step
   otherwise: simplest, but region transfer then needs a wide buffer.
   Decision: A, the uniform quadratic curve in the particle's cell width times the face piece's area share,
   weights divided by their sum, fit with the measured weighted spread. Source: owner answer A7 (Q19).
2. **Bounded candidate rule.** Nearby faces come only from the containing cell and cells touching it, levels
   `p - 1..p + 1` (at most 212 per axis, 124 the largest case known; one layer of border cells; revised 2026-11-19). Alternative: every face within 1.5 widths
   whatever its level: no fixed bound (over 1,000 possible) and two border layers. Recommend the bounded rule.
   Decision: the bounded rule, `kMaxNearbyFaces = 216` (owner decision 2026-11-19; it was 76, which a valid grid
   passes), with `kMinWeightSum = 0.1` as the floor below which the
   fit is `kMeanOnly`. Source: recommended option (owner's standing instruction, 2026-11-14), Q20 and Q51.
3. **`owner` (O5).** Recommend banning it as an identifier in grid, transfer and pressure: say `stored in`
   for a block (`ToCellBlockId(cell)`), `cell_low`/`cell_high` with `kNoIndex` for a border piece, and
   `enum class RegionKind { kParticleRegion, kGridOnlyRegion }` for meaning (3) in module 9 (the grid-only region
   is not in the first version, A64). "Owns:" in file
   headers keeps its ordinary meaning. Alternative: allow `owner` for meaning (3) only.
   Decision: banned as an identifier in grid, transfer and pressure, as recommended. Source: recommended
   option (owner's standing instruction, 2026-11-14), Q5.
4. **`weighted_spread` scope.** Confirm it names only the centred sum `sum weight * offset_from_mean *
   offset_from_mean^T` in m2, that the first sum is `mean_offset`, and that transfer names
   `sum weight * value * offset` itself.
   Decision: confirmed as written. Source: recommended option (owner's standing instruction, 2026-11-14), Q9.
5. **New word `place`.** `cell_place` / `face_place`: a fixed position in a block that may or may not hold an
   active cell or a face piece; hence `CellPlaceIndex`, `FacePlaceIndex`. Alternatives: `CellIndex`,
   `FaceIndex` (short, but `CellIndex3D` is on the never-say list and it hides that unused positions exist).
   Decision: `place` is accepted: `CellPlaceIndex`, `FacePlaceIndex`, `CornerPlaceIndex`; glossary row added
   (section 9). Source: owner answer A3 (Q6).
6. **Sibling cells exist together.** A split cell always has all 8 small cells. This removes quarter border
   pieces and one cell state, and constrains `ChooseActiveCells` and `SplitCell` (cost: up to 7 empty small
   cells at a grid edge). Alternative: allow partial siblings with a fifth state and quarter edge pieces.
   Decision: sibling cells exist together. Source: recommended option (owner's standing instruction,
   2026-11-14), Q39.
7. **`active_cell` and `ChooseActiveCells`.** Proposed single meaning: the active cells are exactly the cells
   that exist and are not split; `ChooseActiveCells` produces that list (new cells arrive at a size chosen
   by the shared cell size rule, `ComputeNewCellSizeLevel`, 3.8; changed 2026-11-15, A41, A61: no callback) and so
   makes a new `GridVersion` before `ParticlesToGrid`;
   `ResizeCells` changes sizes of existing cells after commit. Confirm, or name block existence differently.
   Decision: confirmed as proposed. Source: recommended option (owner's standing instruction, 2026-11-14), Q40.
8. **`BuildGrid` runs on the CPU, also for GPU runs.** Per step only block ids come down and 456 B per block
   go up; no field value moves. Recommend accepting this as not being "a copy between dependent stages";
   the alternative is a device-side builder (sort, search and table fill on the GPU) in this module.
   Decision: accepted; `BuildGrid` is CPU code, now writing into a `GridBuffers` (5.8). Source: recommended
   option (owner's standing instruction, 2026-11-14), Q41.
   **Replaced by decision 16** (owner answer A17): the device-side builder is built; no block ids come down and
   no tables go up.
9. **Error value with context.** `std::expected<T, GridFailure>` with `struct GridFailure { GridError error;
   CellId cell; CellId other_cell; }` and `ToMessage(GridError)`. Alternative: `std::expected<T, GridError>`
   plus a separate out-parameter. One convention should hold for every module.
   Decision: `GridFailure` through `std::expected<T, GridFailure>`, with `ToMessage` and, for every module,
   `ToName` (stable codes). Source: owner answer A2 (Q2); `ToName` from `changes-to-apply.md`.
10. **Small names.** `weight_change` (was `weight_gradient`), `NearbyFaceFit { kFull, kMeanOnly, kNone }`,
    `KeptFor`, `CountedPer`, `ComesBefore` (a predicate that starts with no listed verb; alternative
    `CheckComesBefore`), `ToMaskOfAll`.
    Decision: the names as listed, `ComesBefore` included. Source: recommended option (owner's standing
    instruction, 2026-11-14), Q14.
11. **Harlow and Welch 1965** is required by the naming review's header comment but is not in the old
    bibliography. Please supply the entry (DOI) or agree to cite it by author and year only.
    Decision: cited by author and year only; the full entry is added if the owner supplies it. Source:
    recommended option (owner's standing instruction, 2026-11-14), Q15 (the option that needs no entry).
12. **Block width 4 and 240 face places per block** (25 % of face places unused) are kept as ported
    constants; measuring other widths is deferred with the rest of performance tuning. Confirm.
    Decision: confirmed. Source: recommended option (owner's standing instruction, 2026-11-14), Q49.
13. **(Reconciliation) The grid's type name.** The naming review decides `StaggeredGrid`; this design wrote
    `class Grid` to avoid `grid::StaggeredGrid` beside `GridTables`, `GridFrame`, `BuildGrid`. The reconciled
    set uses `grid::StaggeredGrid` for the owning type and keeps the short `Grid...` prefix on the rest.
    Decision: `grid::StaggeredGrid` for the owning type, short `Grid...` on the rest (option a). Source:
    recommended option (owner's standing instruction, 2026-11-14), Q4.
14. **(Reconciliation) Corner places.** Pressure and solids sample distances at cell corners and need one index
    per physical corner; 3.7 adds 125 corner places per block. Alternative: no corner storage, each cell
    samples its 8 corners itself (8 times the samples, no shared index).
    Decision: corner places, 125 per block, as in 3.7. Source: recommended option (owner's standing
    instruction, 2026-11-14), Q43.
15. **(Update) No allocation inside a try.** Step question 4: a) the grid gains buffers for a limit and values
    bound to a version; b) allocate in stage 2 when the list changed. Module 9 asks the same for pieces.
    Decision: a, for whole grids and pieces: `GridBuffers`, `BuildGridBuffers`, `BuildGrid(inputs, buffers)`,
    `ReleaseBuffers` (now `TakeBuffers`, decision 33), `BuildCellValues(run_on, max_cell_blocks)` and the other forms, `ApplyGridVersion`
    (5.8). Source: recommended option (owner's standing instruction, 2026-11-14), step question 4;
    `changes-to-apply.md` section 02.
16. **(A17) The grid is built on the GPU in a GPU run.** A run is all CPU or all GPU; transfers between them are
    costly. `BuildGrid(inputs, buffers)` keeps its signature; with GPU buffers it validates, sorts and builds every
    table on the GPU (5.1 item 9) and reads back only `GridBuildStatus`. The CPU builder serves CPU runs and is the
    oracle, compared bit for bit. A GPU grid keeps no CPU copy of its tables (`tables(kCpu)` is empty); `ReadGrid`
    is the one explicit copy. `GridInputs` gains `active_cell_buffer` and `active_cell_buffer_count`; the step calls
    `BuildGrid` on its run's processor each time the cell set changes. Replaces decision 8 and the old text of 5.1
    step 9, 5.8 "upload" and section 6 "CPU" and "Per step for the GPU run". Source: owner answer A17;
    `plan-decisions.md` "Grid built on the GPU (A17)".
17. **(A17) One set of build rules for both builders.** Every per-cell, per-block and per-face rule is in the private
    shared `grid_build_steps.h` (sort keys, id check, the three per-position checks, failure key, split cells from
    eighth-0 paths, `FindBlockNeighbours`, `ComputeCoveredCellState`, `ComputeFacePieceKind`, the piece rules);
    `FindCellBlock` is the same binary search the build uses. Source: plan decision, 2026-11-14 (grid plan Task 7).
18. **(A17) First failure and the block limit.** The first failure is the smallest key `(step << 56) | (position << 8)
    | detail` on both processors (5.1 item 9). `kTooManyCellBlocks` from building a grid or a piece names no cell
    (`cell` zero): the GPU cannot name "the first cell of the first block beyond the limit" without working buffers
    beyond the room, and the answer is to raise the limit. Replaces the 3.3 text "`cell` is the first cell of the first
    block beyond the limit". Source: `plan-decisions.md` "Grid built on the GPU (A17)"; plan decision, 2026-11-14.
19. **(A17) Checks before building.** `ComputeGridCounts(inputs, buffers)` and `CheckSizeStepRule(inputs, buffers)`
    run on either processor, allocating nothing; the buffers form of `ComputeGridCounts` fails `kTooManyCellBlocks`
    when the grid does not fit, while the one-argument form (CPU list only; amended by decision 34) counts beyond any room.
    `CheckSizeStepRule` (both forms) runs id checks, overlaps and the size-step rule, not siblings. The
    one-argument forms sort a CPU copy of the cell list (8 B per cell) and are not allocation-free. Source:
    `plan-decisions.md` "Grid built on the GPU (A17)" and "Grid".
20. **(A17) Border search and pieces on either processor.** `BorderSearchBuffers`, `BuildBorderSearchBuffers`,
    `BuildBorderSearchMemoryEstimate`, the stacked buffer form of `FindBorderBlocks` (one layer per call, only the count
    comes back), the buffer form of `BuildGridPiece` and `GridPiece::block_in_whole_grid_buffer()`. The span forms
    of `FindBorderBlocks` fail `kBadInput` on a GPU grid (now a precondition, decision 32); the span form of
    `BuildGridPiece` needs a CPU whole grid;
    `block_in_whole_grid()` is empty for a GPU piece; `BuildRangeWithBorder` requires a CPU grid; `BuildRanges`
    works on either. Cell resize and batches, and pressure's solve batches, use the buffer forms in a GPU run.
    Source: `plan-decisions.md` "Grid built on the GPU (A17)".
21. **(A17) Memory and limit.** The buffers of a GPU grid keep everything on the GPU and nothing on the CPU: 456 B tables +
    1,536 B builder buffers + 4 B piece map = 1,996 B per block, plus the order buffers, `fixed_bytes`, 8 B and
    `sizeof(GridBuildStatus)`; those of a CPU grid are unchanged (456 B `kRunOn`, 516 B `kCpu`). Every row carries its
    `MemoryWhere`; sizes are `sizeof` of the real types. `max_cell_blocks <= kMaxCellBlocksLimit = (1u << 26) - 1`.
    Source: `plan-decisions.md` "Grid built on the GPU (A17)" and "Foundation `base/`", "Applies to every plan".
    **Amended by decision 30** (reconcile): the 1,996 B is the old one-buffers figure without the order buffers;
    the builder buffers can be one `GridBuilderBuffers` counted once.
22. **Values on another processor.** `ApplyGridVersion` runs on the CPU for every grid (records version and block
    count, launches and copies nothing); values on another processor than the grid's are `kBadInput` (now a
    precondition, decision 32). Source:
    `plan-decisions.md` "Grid built on the GPU (A17)" and "Materials" (rule for every module's batch operations).
23. **Headers and ids.** `GridError`, `GridFailure`, `ToMessage`, `ToName` in the shared `grid/grid_error.h`;
    `GridPiece` in `grid/grid_piece.h` (included by `cell_block_range.h`; separate glossary rows for "face piece"
    and "grid piece"); `CellBlockRange` in the shared `grid/grid_ids.h`; `CellId{0}` and `FacePieceId{0}` never
    valid; ids opaque to other modules. Source: `plan-decisions.md` "Grid" and "Grid (continued)".
24. **Empty grid.** `BuildGrid` accepts an empty cell list (an empty scene is valid): zero blocks. Source:
    `plan-decisions.md` "Step".
25. **No-allocation tests** count allocations through a replaced global `operator new` in the test executable; on
    the GPU they read `ReadMemoryAvailable(kGpu)` free bytes. Source: `plan-decisions.md` "Grid"; plan decision,
    2026-11-14.
26. **A15 (batched solves) changes nothing in the grid's interface.** Pressure's `SolveBatches` uses
    `BuildRanges`, the buffer form of `FindBorderBlocks` and the buffer form of `BuildGridPiece` as they are (4).
    Source: `plan-decisions.md` "Batched solves (A15)".
27. **Method paper.** Adds `<sec-parallel-build>`, `<fig-parallel-build>`, `<proc-build-grid-in-parallel>` (8).
    Source: plan decision, 2026-11-14 (grid plan Task 1).
28. **(Reconcile) No default room.** `GridSettings::max_cell_blocks` defaults to 0; the design's `1u << 22` is
    withdrawn. `BuildGridBuffers` and `BuildGridBuilderBuffers` with 0 are `kBadInput`; the allocating
    `BuildGrid(inputs, settings)` (now the test-only `BuildTestGrid`, decision 34) with 0 sizes its buffers to the list's blocks (at least 1). No compile-time room is
    an allocation size. Closes the former open point 1 (GPU grid buffers at `1u << 22` held about 8.4 GB, two of them
    above a 16 GB GPU). Decision: no default room; the planner sizes it (decision 31). Source: `plan-decisions.md`,
    Reconcile pass item 11.
29. **(Reconcile) One `GridBuilderBuffers` lent to several buffers.** `GridBuilderBuffers`, `BuildGridBuilderBuffers`,
    `BuildGridBuffers(const GridSettings&, GridBuilderBuffers&)`, `BuilderBuffersSharing { kOnePerGrid, kOneForAllGrids }` and
    `BuildGridMemoryEstimate(run_on, grid_count, builder_buffers_sharing = kOnePerGrid)` (3.3, 3.6). The caller owns the
    builder buffers; the buffers must not outlive them; they are not moved while lent; one build or check uses them at a
    time; no operation on a built grid reads them. Owners: the step's simulation state (committed, free and loaded-try buffers),
    the batched GPU run's CPU grid copy pair, cell resize's two CPU grids in a one-piece GPU run.
    `BuildGridBuffers(settings)` still owned its own (removed by decision 34). Decision: as listed. Source: `plan-decisions.md`, Reconcile
    pass item 11; grid plan Task 6.
30. **(Reconcile) Memory numbers.** GPU: per buffers 460 B per block (456 tables + 4 piece map) plus the status
    record; builder buffers 1,920 B per block (`1536 + 64 x 6`) plus 4,096 + 8 B, once (`ComputeOrderBuffersSize(kGpu)
    = {6, 4096}`; plus 6 B, decision 35). CPU: 456 B `kRunOn` + 4 B `kCpu` per buffers, builder buffers 512 B `kCpu`.
    Two GPU grids 2,840 B per block (was 4,760). 16 GiB GPU: a quarter of `allowed_bytes` holds about 1.06 M blocks for
    tables and builder buffers. Closes the former open point 2 (the order buffers size). Decision: as listed (3.6). Source: `plan-decisions.md`, Reconcile
    pass item 11; "Foundation as built: the rest" (`stable_order.h`).
31. **(Reconcile) Room sized by the planner.** `limits.max_cell_blocks` = the blocks `step::BuildRunPlan` counts from
    the scene x `start_limit_headroom` (1.25), capped by the domain's blocks at the smallest cell, at least 1; above
    `kMaxCellBlocksLimit` or when the estimate does not fit, `fit = kDoesNotFit` and the budget prints the
    does-not-fit lines; outgrowing it in a run is `kTooManyCellBlocks` (`kRaiseLimits`, `ApplyLimits`). Decision: as
    listed (5.8). Source: `plan-decisions.md`, Reconcile pass item 11.

32. **(A30) Buffers on different processors.** `BufferError::kRunOnMismatch` no longer exists and the grid maps
    nothing for it. Every grid function whose arguments must share one processor states it as `Requires:` and
    checks it by a debug assertion: the cell list and the `GridBuffers` (span forms: a CPU list), the lent
    `GridBuilderBuffers` and `BuildGridBuffers`' settings, `ReadGrid`'s CPU buffers, bound values and their grid
    (`ApplyGridVersion`), the stacked `FindBorderBlocks` buffers, the span forms of `FindBorderBlocks` and
    `BuildGridPiece` on a CPU grid. Removed tests: `ApplyGridVersion, ValuesOnOtherProcessor_IsBadInput`,
    `CheckSizeStepRule, OnGpuBuffersWithCpuList_IsBadInput`, `ComputeGridCounts,
    GpuBufferListInOneArgumentForm_IsBadInput`, `BuildGridPiece, SpanFormOnGpuWhole_IsBadInput`; the other-processor
    half of `BuildGridBuffers, LentBuilderBuffersTooSmall_IsBadInput` (which keeps its too-small half). Amends decisions 19, 20, 22 and the A17 "one processor per run" text.
    Source: owner answer A30 (`owner-answers.md`, 2026-11-14).
33. **(A29) Verbs.** `ReleaseBuffers(StaggeredGrid&&)` and `ReleaseBuffers(GridPiece&&)` become `TakeBuffers` (they
    get preallocated buffers back out of an object no longer needed); prose says "take back". Source: owner answer
    A29.
34. **(A31) Working buffers, one form.** The grid's build working memory is `GridBuilderBuffers` with
    `BuildGridBuilderBuffers`, `GridBuilderBuffersSize`, `ComputeGridBuilderBuffersSize` and `BuilderBuffersSharing
    { kOnePerGrid, kOneForAllGrids }`; its entries are "grid: builder buffers: ..." and the ordering entries "order
    buffers" (replaced names listed once in section 9; `GridBuildBuffers` was not taken: it would give
    `BuildGridBuildBuffers`). One form:
    `BuildGridBuffers(settings)` (owned its builder buffers) and the allocating `BuildGrid(inputs, settings)` are
    removed; `BuildGridBuffers(settings, builder_buffers)` is the one form, and tests use the test-only
    `BuildTestGrid` (grid plan Task 9). The span forms of `ComputeGridCounts` and `CheckSizeStepRule` stay beside
    their buffers forms, for a stated reason: the first counts beyond any room (the planner), the second checks CPU
    lists while cell resize splits, where no grid buffers exist. `IntoBuffers_EqualsAllocatingForm` is renamed
    `IntoBuffers_EqualsSpanForm`; `BuildGrid, DefaultSettings_SizedToTheList` becomes `BuildTestGrid,
    DefaultSettings_SizedToTheList`. Source: owner answer A31.
35. **(Reconcile items 20 and 28) Where the sharing enum lives; the order buffers counted.** `BuilderBuffersSharing`
    is declared in `staggered_grid.h` (included by `grid_memory_estimate.h`). The GPU `OrderBuffers` (6 B per item)
    of the build is the entry "builder buffers: order buffers"; it is built for `64 * max_cell_blocks + 1` items, so
    its fixed-part entry is `fixed_bytes + bytes_per_item` (6 B more than before: 4,096 + 6 + 8 fixed bytes on the
    GPU). The `OrderBuffers` of `BorderSearchBuffers` (`max_cell_blocks + 1` items) is the entry "border order
    buffers", fixed part `fixed_bytes + bytes_per_item`. Source: `reconcile-todo.md` items 20 and 28.

36. **(A41, A61, A62, A63) The shared cell size rule is the grid's.** `cell_size_rule.h` (S) holds
    `CellSizeSettings` and `CellSizeFacts` (moved from cell resize), `CellWidthsByLevel`, `ComputeReferenceSpeed`,
    `ComputeWantedCellWidth`, `ComputeWantedSizeLevel`, `ComputeNewCellSizeLevel`, ~~`CellSizeRegion`,
    `ArtistCellSizes`, `ComputeArtistSizeLevel`~~ (replaced by the refinement volume, decision below, A78, A79) (3.8, 5.9), built in grid plan Task 13a; the grid owns it because
    the step and cell resize both read it and both come after grid. The `ChooseCellSize` callback is removed
    (decision 7 amended). Plan decisions: `ComputeCellWidthsByLevel`; `artist_size_level` 0 means none; regions
    inclusive; ties of the new-cell probe by `ComesBefore`. Source: owner answers A41, A61, A62, A63;
    `plan-decisions.md` "Owner answers A40–A64 applied", item 2.
37. **(A58) One reserved block per run.** Every builder (`BuildGridBuilderBuffers`, `BuildGridBuffers`,
    `BuildBorderSearchBuffers`, all `Build...Values` forms) takes `RunMemory& run_memory` last, the block of the
    processor its arrays live on; every array is one piece and one estimate entry. An array of a per-item plus a
    fixed size is one `kGivenCount` entry of its exact bytes (the rows `order buffers, fixed part`, `prefix sum
    ends`, `border marks end`, `border order buffers, fixed part` are folded in), so `BuildGridMemoryEstimate` and
    `BuildBorderSearchMemoryEstimate` take `max_cell_blocks` and the reservation is exact for any room (3.6
    "Reservation"). Amends decision 25: GPU free-byte tests
    measure the drop across `BuildRunMemory`. Source: owner answer A58; `plan-decisions.md` item 3 and its
    same-day correction and reconcile ruling.
38. **(Reconcile item 31) `kOutOfMemory` last.** `GridError::kOutOfMemory` moves to the last place (12 values then; 13 since A106 added `kInternalFault` just before
    it, 2026-11-20);
    only the builders return it, also for `BufferError::kRunMemoryTooSmall`; `GridFailure` gains `magnitude` (bytes
    asked; plan decision, the grid's failure had no number). Source: `plan-decisions.md` item 11.
39. **(Reconcile items 32, 33)** The span forms on a GPU grid (`FindBorderBlocks`, `BuildGridPiece`) and a CPU
    list with GPU buffers in the grid checks are `Requires:` preconditions (3.4 text corrected: it said `kBadInput`).
    Accepted as written: `BuildTestGrid` replaces the allocating `BuildGrid(inputs, settings)`; the span forms of
    `ComputeGridCounts` and `CheckSizeStepRule` are kept. Source: `plan-decisions.md` item 11.
40. **(A64) Particles everywhere** changes nothing in the grid. Source: `plan-decisions.md` item 1.
41. **(A65) Particle size follows the cell.** Every liquid cell holds about eight particles of its own size: a
    particle's `size_level` is its cell's level, its rest volume `(w/2)³`, so a cell k levels coarser than the smallest holds 8 particles each
    of `8^k` times the smallest particle's volume. Nothing in the grid reads particle sizes; the planner's room
    (`max_cell_blocks`) comes from the exact start cells (A66). Source: `plan-decisions.md` "Owner answers A65–A68
    applied", A65.
42. **(A66) The first grid comes from `step::BuildStartCells`.** Decision: the grid's first version is built from
    `StartCells::cells` (not from particles at `new_cell_size_level`); `ComputeWantedCellWidth` takes a negative
    depth or solid distance as the surface or the solid (5.9; test `ComputeWantedCellWidth,
    NegativeDepthOrDistance_IsSmallest`); `CheckSizeStepRule` is the start grid's test oracle. No new grid name.
    Source: owner answer A66; `plan-decisions.md` A66.
43. **(A67) Drifted particles** change nothing in the grid. Source: `plan-decisions.md` A67.
44. **(A68) Batches cut by columns or tiles (later).** A tile cut needs a batch to be a set of blocks that is not one
    `CellBlockRange` of the slab order (an x-y tile spanning every height), so `GridPiece` and `FindBorderBlocks`
    would gain a tile form; that is the later group's work, with no grid task in the first version. Source:
    `plan-decisions.md` A68.

**Open**

None. (The former points 1, default room, and 2, order buffers size, are decided by decisions 28 and 30.)

**Decision (A78, A79; reshapes A72, A73, A77; 2026-11-15): the artist's wish is a refinement volume of factors.**
The artist regions and the top-view map (A63) are removed. The rule reads one keyed volume of cell size factors over
the domain box, stored as a min-pyramid of one-byte codes (`f = 2^(c/4)`, 1/64 to 64; `RefinementFactors`,
`ComputeCellSizeFactorCode`, `CellSizeFactorTable`), and multiplies the physical width by the cell's smallest factor:
`w = clamp(min(w_depth, w_solid, w_shear) x f, w_min, w_max)`, `w_min` in the cells within 8 smallest widths of the surface, pockets and gas.
Why a product: the artist scales what the flow needs instead of overriding it; a harmonic mean (the owner's other
idea) cannot say "leave it to the physics" without a special value and keeps the result between the two widths.
Limit of the first version: the factor does not reach the cells within 8 smallest widths of the surface, so a hero region finer than `w_min`
needs a smaller scene `w_min` (with factors above 1 elsewhere); a zone that lowers `w_min` locally waits for coarse
surface cells (A61, A68). Zones, meshes and the camera are I/O's builder, outside the solver. Source: A72, A73, A77,
A78, A79; `plan-decisions.md` "Refinement volume (A78, A79; reshapes A72, A73, A77)".
