# Mesh solids module design

Status: rewritten 2026-11-15 for the owner's simple route (A94). Module 1b of the rebuild (A80). The first version
is for meshes with real thickness: no thin-wall mechanism is built (section 5.9). Decisions A80 to A94 are in
section 11; no owner question is open. The contact part has a plan,
`plans/2026-11-14-rebuild-01b-mesh-solids.md`; what the other modules must change is in
`01b-mesh-solids-cross-module.md` and, as an edit list, in `changes-to-apply.md` "01b Mesh solids (A80)".
Literature claims are marked (checked) when the citation was confirmed on the web on 2026-11-15 and (memory) when
not.

## 1. Purpose

A real scene's solids are not boxes, spheres and cylinders: they are meshes made in Blender. This module lets a
solid be a closed surface of triangles, of any shape (a bowl, a pipe, a statue), so that particles stop at it and
slide along it exactly as they do on a box, and so that every other module can ask the same questions of it:
how far is this point from the solid, which way is out, how fast is the surface moving there.

**Owns:** the mesh shape (`SolidShape::kMesh`): its triangles in the solid's own axes, the tree of boxes that finds
the triangles near a point or a path, the distance and normal to it, the inside test, the first hit of a path on
it, the slide along it (concave corners included), the checks that a mesh is usable, the measure of its thinnest
wall, the builder that turns vertices and triangles into the arrays the solver reads, and the buffers that hold
them on the run's processor.

**Does not own:** reading a mesh file, the scene file fields, the files kept for resume, mass, centre of mass and
inertia of a mesh (I/O); exporting the mesh from Blender (add-on); how a solid moves from step to step and the
force on it (solids); blocked areas (pressure, filled by solids); the finding about thin parts (the step's scene
check, from this module's measure); everything module 1 owns. Not in the first version: a mesh that changes shape
(5.8), any mechanism for walls thinner than the pressure solve can see (5.9).

On record (A80): the grid does not follow surfaces. Cubes stay cubes; the grid knows a mesh only through numbers
per cell, face and corner (blocked areas, the distance at corners), and the particles carry where the material is.

## 2. What exists today

- `main`: nothing. Solids are analytic shapes only (solids design sections 1 and 10; add-on design section 10).
- Rebuild, module 1, built through Task 8a (`native/fluidity/contact/`): `solid_shapes.h` (six shapes behind one
  `Solid`; `sizeof(Solid)` is 200 with 7 trailing padding bytes after `reports_reaction` at offset 192; no
  `static_assert` on it yet, module 1's Task 15 adds one), `contact_tolerance.h`, `contact_error.h`
  (`ContactFailure`: two enum bytes, then 6 padding bytes before `particle_id`), `solid_contact_types.h`
  (`SolidContactInputs`, `FirstHit`, `ComputeContactResponse`, `ComputeLargestApproachLeftAfterResponse`),
  `solid_contact_steps.h` (private steps: `Sweep`, `OwnPath`, `HitTime`, `StartCase`, `ComputeStartCase`,
  `ComputeOwnPath`, `ComputeDistanceForParticle(inputs, solid_index, pose, point, plate_side)`,
  `FindFirstHitOnSolid(inputs, solid_index, start, velocity_for_this_step, start_elapsed, duration, FirstHit*)`
  returning `ContactError`, whose `switch (solid.shape)` writes every case out and ends in an assertion and
  `kBadInput`), `solid_contact.h` (`FindFirstHit`, six and seven arguments). A turning solid returns
  `kTooCloseToCall` until Task 9. `MoveParticle`, `MoveParticles` and the GPU half are not built (Tasks 9 to 18).
- The only mesh input planned elsewhere is the closed-mesh refinement zone (I/O plan Tasks 29b, 29d; add-on
  `zone_mesh.py`): a binary PLY, a closed test, an inside test by ray parity. Reused: the PLY layout and reader,
  the add-on's evaluated-mesh export with the fan rule, the listed `inputs/` array file a resume reads.
- Module 1's arguments that rest on convexity, which a mesh breaks: the touching rule (P2, P16); "each convex
  solid is hit at most once per sweep"; the closed-form hit times; `kMaxHitsPerStep = 4`; one move along the
  normal reaches the rest gap.
- A lesson of module 1's Task 8a review that shapes 5.4: the cylinder's first hit computed two candidates (wall,
  cap) and checked each candidate's hit point against the other surface with a separately rounded test; a path
  through the grown rim edge fell between the two and passed through the solid, in about 5 % of paths within a
  few rounding units of the edge. The fix was the box's form: an interval `entry <= exit`, where rounding is
  monotone.

Z-up: nothing specific. A mesh is stored in the solid's own axes; the pose places it.

## 3. Public interface

Folder `native/fluidity/contact/`, namespace `fluidity::contact`; contact still includes only `base/`. All names
NEW. Headers follow module 1's layout as built: a types header, a private steps header, a public header, no
forward declarations. Include order: `solid_shapes.h` and `mesh_shape.h` (independent) <- `mesh_shape_steps.h` <-
`mesh_contact.h` <- `solid_contact_steps.h`; `solid_contact_types.h` includes `mesh_shape.h` for one field.

### 3.1 Appended to module 1's headers (no earlier value or offset moves)

```cpp
// solid_shapes.h
enum class SolidShape : std::uint8_t { /* ..., kCylinder, */ kMesh };          // value 6
enum class SolidFace : std::uint8_t { /* ..., kHighRim, */ kMeshTriangle };    // value 9
inline constexpr std::uint32_t kNoMesh = 0xFFFFFFFFU;   // "No mesh", as an index into the run's mesh shapes.
struct Solid {
  // ... the existing fields, unchanged, then:
  std::uint32_t mesh_index = kNoMesh;   // kMesh only. At offset 196, inside the trailing padding:
};                                      // static_assert(sizeof(Solid) == 200).
// kMesh: half_size holds, per own axis, the largest |coordinate| of any vertex, so the reach Length(half_size)
// bounds the surface from the origin, as for a box.
// CheckSolid(solid, step_duration[, close_call]) gains the kMesh case without the table: half_size finite and
// positive, mesh_index != kNoMesh, the turn limit on the whole angular velocity. (FindFirstHit asserts it.)
// The functions of solid_shapes.h and contact_tolerance.h that take no MeshShapes (ComputeDistanceToSolid,
// ComputeSolidVelocitySample, ComputeContactScale) gain "Requires: no solid is a kMesh" with its assertion; in a
// release build they read a mesh as the box of its half_size, which contains it, so a caller's mistake stops
// particles early and never lets one through.

// contact_error.h
enum class ContactError : std::uint8_t { /* ..., kOutOfMemory, */ kMeshNotUsable };   // appended
enum class MeshProblem : std::uint8_t {
  kNone,
  kEmpty,                // No triangle.
  kTooManyTriangles,     // More than kMaxMeshTriangles.
  kNotClosed,            // An edge is not shared by exactly two triangles. magnitude: the count of such edges.
  kFacesTurnedBothWays,  // Two triangles run along a shared edge in the same direction.
  kTriangleTooFlat,      // A triangle's smallest height is below 2^-20 of its longest edge, or below 1024 tol.
  kEdgeTooSharp,         // Two faces meet at an inside angle below 2^-6 rad.
  kCrossesItself,        // Two triangles that share no vertex intersect.
  kPartTooThin,          // The thinnest wall is below 1024 contact tolerances.
};
// ContactFailure gains `MeshProblem mesh_problem = MeshProblem::kNone;` at offset 2 (existing padding; sizeof
// unchanged, static_assert). ToMessage and ToName cover MeshProblem. ContactCloseCall is unchanged.

// solid_contact_types.h: SolidContactInputs gains, last, `MeshShapes meshes{};`.
// move_particles.h (module 1 Task 15): MoveParticlesInputs gains `MeshShapes meshes{};` on the call's processor.
// FindFirstHit, MoveParticle, MoveParticles keep their signatures. PlateSide, FindSolidSide: unchanged.
```

### 3.2 `contact/mesh_shape.h` (shared; types and constants only; includes `base/` only)

```cpp
struct MeshTreeBox {                  // 32 bytes. One box of the tree. Papers: bounding volume hierarchy node.
  float low[3], high[3];              // m, own axes. Rounded outward: the box contains its triangles.
  std::uint32_t first = 0;            // count == 0: index of the first of its two inner boxes (the second is +1).
  std::uint32_t count = 0;            // count > 0: an end box holding triangles [first, first + count), <= 4.
};
struct MeshShape {                    // One mesh on one processor. Borrowed arrays; never allocates. 120 bytes.
  const Vec3* vertices = nullptr;             // m, own axes. vertex_count.
  const std::uint32_t* triangles = nullptr;   // 3 vertex indexes per triangle, counter-clockwise seen from
                                              // outside, in tree order (each end box's triangles contiguous).
  const std::uint32_t* neighbours = nullptr;  // 3 per triangle: the triangle across edge (0,1), (1,2), (2,0).
  const Vec3* vertex_normals = nullptr;       // Unit, own axes: the triangle normals around the vertex, each
                                              // weighted by its angle there (Baerentzen and Aanaes 2005).
  const MeshTreeBox* tree = nullptr;          // tree_box_count. Box 0 is the root.
  std::uint32_t vertex_count = 0, triangle_count = 0, tree_box_count = 0;
  Vec3 half_size{};                   // m. Largest |coordinate| of a vertex per own axis.
  double thinnest_wall = 0;           // m. Smallest measured thickness between two faces back to back (5.2);
                                      // +infinity when the builder found none.
  Vec3 thinnest_wall_point{};         // m, own axes. Where it was measured (on the surface). Zero when none.
  double smallest_triangle_height = 0;// m.
};
struct MeshShapes { const MeshShape* meshes = nullptr; std::uint32_t mesh_count = 0; };   // Passed by value.

inline constexpr std::uint32_t kMaxMeshTriangles = 1U << 24;   // 16.8 million per mesh.
inline constexpr int kMaxTrianglesPerEndBox = 4;
inline constexpr int kMaxMeshTreeDepth = 24;       // ComputeMeshTreeDepth(kMaxMeshTriangles) is 22; two spare.
inline constexpr int kMaxMeshHitsPerStep = 16;     // Hits on meshes one particle may have in one step (5.6).
inline constexpr int kMaxMoveOutCount = 4;         // 5.5.
inline constexpr double kMeshNearStartInTolerances = 1.5;   // 5.4: a sweep that starts this near is "near".
inline constexpr double kMeshRestingSweepShare = 0.5;       // 5.4: a near sweep grows triangles by tol / 2.
inline constexpr double kMeshPieceOverlapShare = 0.125;     // 5.4: pieces overlap by an eighth of the radius.
// How many times a count of triangles is halved (the larger half kept) before it is at most
// kMaxTrianglesPerEndBox: the depth of the builder's tree and the most box indexes a search keeps.
constexpr int ComputeMeshTreeDepth(std::uint32_t triangle_count);
static_assert(ComputeMeshTreeDepth(kMaxMeshTriangles) <= kMaxMeshTreeDepth);

struct DistanceToMesh {
  double distance = 0;                // m. Positive outside, negative inside. Exact to rounding (5.3).
  Vec3 own_normal_out_of_solid{};     // Unit, own axes.
  Vec3 own_nearest_point{};           // m, own axes, on the surface.
  std::uint32_t triangle_index = 0;   // The nearest triangle; ties keep the lower index.
};
struct MeshSweep {                    // A straight path in a mesh's own axes, the mesh at rest.
  Vec3 own_start{};                   // m.
  Vec3 own_velocity{};                // m/s, relative to the solid.
  double duration = 0;                // s, zero or more.
  double contact_tolerance = 0;       // m, zero or more.
};
struct MeshFirstHit { bool has_hit = false; double elapsed = 0; DistanceToMesh at_hit{}; };
```

### 3.3 `contact/mesh_contact.h` (shared, public), `contact/mesh_shape_steps.h` (shared, private to `contact/`)

```cpp
// `largest_distance`: the search gives up beyond it and returns distance == largest_distance, positive, with a
// zero normal, so a far point costs one box test. Pass +infinity for the true value everywhere.
DistanceToMesh ComputeDistanceToMesh(const MeshShape& mesh, Vec3 own_point, double largest_distance);
// First time sweep.own_start + sweep.own_velocity * t, 0 <= t <= duration, meets the mesh grown by the sweep's
// radius, with the start cases of 5.4. at_hit is ComputeDistanceToMesh at the hit position.
MeshFirstHit FindFirstHitOnMesh(const MeshShape& mesh, MeshSweep sweep);
Solid ToSolid(const MeshShape& shape, std::uint32_t mesh_index, SolidPose pose, SolidMotion motion,
              std::uint32_t solid_id);
// The mesh-aware forms of module 1's point functions: the same results for the six shapes, and kMesh through
// meshes.meshes[solid.mesh_index].
DistanceToSolid ComputeDistanceToSolid(const Solid& solid, SolidPose pose, Vec3 point, MeshShapes meshes);
SolidVelocitySample ComputeSolidVelocitySample(const Solid* solids, std::uint32_t solid_count, double elapsed,
                                               Vec3 point, double search_distance, MeshShapes meshes);
ContactScale ComputeContactScale(const Solid* solids, std::uint32_t solid_count, double step_duration,
                                 double smallest_cell_width, MeshShapes meshes);
ContactError CheckSolid(const Solid& solid, double step_duration, MeshShapes meshes, ContactCloseCall* close_call);
// The checks that need the run's tolerance (5.2 checks 4 and 8). On failure *failure names the problem.
ContactError CheckMeshShape(const MeshShape& mesh, double contact_tolerance, ContactFailure* failure);
```

`ComputeContactScale(..., meshes)`: a mesh adds `Length(half_size)` to `largest_coordinate` and gives the
`thinnest_solid` candidate `min(thinnest_wall, 2 * min(half_size))`. `CheckSolid(..., meshes, ...)` adds
`mesh_index < mesh_count` and `half_size` equal to the shape's. Private steps (names for the plan): nearest point
on a triangle, distance to a tree box, the search for the nearest triangle, the interval of one piece of a grown
triangle, the search for the earliest entry, the mesh's start case.

### 3.4 `contact/mesh_shape_builder.h`, `.cc` and `contact/mesh_shape_buffers.h`, `.cc` (CPU only)

```cpp
struct MeshShapeArrays {              // Owns one mesh's arrays in CPU memory. Movable.
  std::vector<Vec3> vertices; std::vector<std::uint32_t> triangles, neighbours;
  std::vector<Vec3> vertex_normals; std::vector<MeshTreeBox> tree;
  Vec3 half_size{}; double thinnest_wall = 0; Vec3 thinnest_wall_point{}; double smallest_triangle_height = 0;
};
MeshShape ToMeshShape(const MeshShapeArrays& arrays);          // The borrowed form, CPU memory.
enum class SelfCrossingCheck : std::uint8_t { kRun, kSkip };  // kSkip: tests and tools that know the mesh.
// Builds the arrays from vertices (m, own axes) and triangles, then runs the checks of 5.2 that need no
// tolerance. Same input, same bytes. Fails kMeshNotUsable (mesh_problem, magnitude), kBadInput (a value not
// finite, an index out of range).
std::expected<MeshShapeArrays, ContactFailure> BuildMeshShape(std::span<const Vec3> vertices,
    std::span<const std::array<std::uint32_t, 3>> triangles, SelfCrossingCheck self_crossing_check);

class MeshShapeBuffers;   // The run's meshes on run_on(): one buffer per array kind, all meshes end to end, and
                          // the MeshShape table. Move-only. Members: RunOn run_on() const; MeshShapes meshes() const.
std::expected<MeshShapeBuffers, ContactFailure> BuildMeshShapeBuffers(RunOn run_on,
    std::span<const MeshShapeArrays> meshes, RunMemory& run_memory);        // kOutOfMemory, kRunOnCallFailed
std::vector<MemoryEstimateEntry> BuildMeshShapeMemoryEstimate(std::uint64_t vertex_count,
    std::uint64_t triangle_count, std::uint64_t tree_box_count, std::uint32_t mesh_count);
```

Memory rows (all `kRunOn`, `kWholeRun`, `kGivenCount`, one buffer each): "contact: mesh vertices" 24 B, "contact:
mesh triangles" 12 B, "contact: mesh neighbours" 12 B, "contact: mesh vertex normals" 24 B, "contact: mesh tree"
32 B, "contact: mesh table" 120 B. A closed mesh has about half as many vertices as triangles and between a half
and 0.8 tree boxes per triangle, so **64 to 74 B per triangle**: 7 MB for 100,000 triangles, 65 to 74 MB for one
million (checked with the tree's count formula for 12 to 2^24 triangles). I/O keeps the same arrays in CPU memory
for the CPU-only consumers ("io: mesh solids", `kCpu`).

## 4. Consumes from other modules

**Foundation (EXISTS):** as contact section 4; also `math::Min`, `math::Max` (a NaN loses: a comparison that must
fail on NaN is written out), `Buffer`, `BuildBuffer`, `WriteBuffer`, `RunMemory`, `MemoryEstimateEntry`.

**Contact, module 1.** EXISTS (Tasks 1 to 8a): everything in section 2. NEEDED from its unbuilt tasks, as its
plan states them: Task 9 the stepping search behind `FindFirstHitOnSolid`, reading the gap only through
`ComputeDistanceForParticle`; Task 10 `MoveParticle` with a surface list whose entries hold the solid index and
the normal recorded at the hit; Task 11 `MoveToRestGap(inputs, solid_index, pose, point)`; Task 12 the
earlier-surface rule; Task 13 the carry in a turning solid's axes as one private function; Task 14 the
"ends outside, control enters" helper; Task 15 `MoveParticlesInputs`, `ContactBuffers`, the size assertion;
Task 16 `MoveParticlesView`, `ApplyMoveParticleAt`.

**Nothing else.** I/O, solids, the step and the add-on consume this module (`01b-mesh-solids-cross-module.md`).

## 5. Method

Notation: `tol` the contact tolerance, `d` the distance to the solid (negative inside), `p` a point in the
solid's own axes, `u` the particle's velocity for this step minus the solid's, in own axes, `r` the radius a
sweep grows triangles by.

### 5.1 Representation (decided, A81)

**The triangles are the solid, for every module**, searched through a tree of boxes. Reasons kept from the
comparison with a distance volume: (1) it is the only choice that keeps module 1's tolerance and its safety
property; the no-miss results in the literature are exact or conservative tests on the geometry (Wang et al.
2021, Brochu et al. 2012, both checked), never push-out from a volume. (2) Viscosity, the grid's size rule and
the open-to-sky flood need a true distance (1-Lipschitz, never an overestimate); an interpolated lattice
guarantees neither. (3) Memory is about a hundredth of a 2 mm volume and does not grow with smaller cells.
(4) One representation, one set of tests. Given up: a tree search is more GPU code than eight reads, and
grid-side stages pay searches where a volume would pay reads. A volume for points farther than two cells from
the surface stays the measured fallback behind `ComputeDistanceToMesh` (the size test of section 7 is its
trigger); it is not designed here.

### 5.2 Building and checking a mesh (CPU, once per mesh)

Steps: 1 copy the vertices; drop none, merge none (the exporter's job). 2 Checks 1 to 3. 3 Order the triangles
by the Morton code of their centre (21 bits per axis within the mesh's bounds; ties by input index). 4 The tree:
a box holds a contiguous range of that order; a range of more than 4 triangles is split by count into its first
`ceil(n / 2)` and its last `floor(n / 2)`; boxes are stored parent before children, the two children adjacent.
Its depth is `ComputeMeshTreeDepth(triangle_count)`, at most 22. 5 Neighbours across each edge. 6 Vertex
normals. 7 Checks 4 to 7 and the measures. 8 `half_size`.

Checks, in this order; the first failure is reported with the lowest triangle index:

1. `kEmpty`, `kTooManyTriangles`.
2. `kNotClosed`: every edge is shared by exactly two triangles.
3. `kFacesTurnedBothWays`: the two triangles of every edge run along it in opposite directions. Then the summed
   volume (`Dot(a, Cross(b, c)) / 6` over triangles) is positive, else the builder turns every triangle over.
4. `kTriangleTooFlat`: every triangle's smallest height is at least `2^-20` of its longest edge (builder), and
   `smallest_triangle_height >= 1024 tol` (`CheckMeshShape`).
5. `kEdgeTooSharp`: the angle between the two faces of every edge, measured inside the solid, is at least
   `2^-6` rad (0.9 degrees). Sharper, the inside test's dot product at a distance `tol` falls toward rounding.
6. `kCrossesItself` (unless skipped): no two triangles that share no vertex intersect: each edge of one against
   the other, both ways, pairs found tree against tree. Floating point: two parts that touch within rounding
   may be reported either way. A mesh made of two closed parts pushed into each other is refused (the inside
   test is wrong inside the overlap); the artist joins them or exports two solids.
7. Measure `thinnest_wall`: from four points of each triangle (its centre and the three points halfway from the
   centre to a vertex) a ray goes inward along the triangle's normal to the first triangle it meets. When that
   triangle faces the other way (`Dot` of the two normals at most `-0.5`: two faces back to back, or a wedge of 60 degrees or sharper
   near its edge; not the two sides of a wider wedge) the ray's length is a wall thickness. `thinnest_wall` is the smallest, and
   `thinnest_wall_point` its starting point (ties: the lower triangle, then the first point). It is a measure:
   a thin spot between the sample points can be missed.
8. `kPartTooThin` (`CheckMeshShape`): `thinnest_wall >= 1024 tol` (module 1's `kMinFeatureInTolerances`).

### 5.3 Distance, normal and the inside test

`ComputeDistanceToMesh`: depth-first search of the tree from the root, nearer inner box first (equal: the
first), skipping a box whose distance to `p` is not below the best found (start: `largest_distance`). The search
keeps the indexes of the boxes still to visit in a fixed array of `kMaxMeshTreeDepth` entries: it pushes at most
one box per level it descends. At an end box, the nearest point of each triangle by the region form (Ericson
2005, 5.1.5): vertex, edge or face. Keep the smallest squared distance; ties keep the lower triangle index. Then:

- Outward direction at the nearest point `c`: the triangle's normal when `c` is in its face; the sum of the two
  triangles' normals (through `neighbours`) on an edge; `vertex_normals` at a vertex. The point is inside exactly
  when `Dot(p - c, that direction) < 0` (Baerentzen and Aanaes 2005, checked: correct for a closed surface that
  does not cross itself, given the true nearest point).
- `distance = +-Length(p - c)`; `own_normal_out_of_solid = +-(p - c) / Length(p - c)`; when `p == c` as computed,
  the direction above, normalised (tested with exact equality, as module 1's degenerate normals).

This is the true distance, like the sphere's and unlike the box's: 1-Lipschitz, exact to rounding. Only
`+ - * /`, `Sqrt` and comparisons: same bits on both processors for a given pose. Near an edge or a vertex the
normal is sensitive: a rounding of `1e-15` m in a point `tol` from the edge turns it by about `1e-6`, so a test
can ask for a normal near an edge only to about `1e-5`.

`ComputeDistanceToSolid(..., meshes)` turns the world point into own axes, searches with `+infinity`, turns the
normal back and reports face `kMeshTriangle`. `ComputeSolidVelocitySample(..., meshes)` searches with
`largest_distance = Max(search_distance, 0)` lowered to the best distance already found among earlier solids,
so a mesh far from the point costs one box test. `ComputeDistanceForParticle` (module 1, private) passes
`inputs.meshes` to the mesh-aware form: one line, and every start rule, settle and end check then sees meshes.

### 5.4 First hit on a mesh that does not turn

In own axes the mesh is at rest and the path is the straight segment `p0 + u t`.

**The mesh grown by `r`** is every point within `r` of some triangle: the union of the triangles grown by `r`.
A path that starts outside it first meets it at the earliest time it enters any grown triangle. That needs no
convexity of the mesh and no argument about which triangle is nearest.

**One grown triangle is covered by seven convex pieces, each swept as an interval.** With `n` the triangle's
unit normal, `o = kMeshPieceOverlapShare * r`:

- the *face piece*: `|Dot(n, p - a)| <= r` and, for each edge, `Dot(m_k, p - a_k) <= o`, `m_k` the unit vector
  in the triangle's plane pointing out of edge `k`: five half spaces;
- three *edge pieces*: within `r` of the edge's line, and between the two planes across the edge's ends, each
  moved outward by `o`;
- three *vertex pieces*: within `r` of the vertex.

For each piece the path's times inside are one interval, the intersection of the intervals of its conditions (a
half space gives a one-sided interval; a cylinder or a sphere gives the two roots of its quadratic, or nothing).
The piece is met when `entry <= exit`, `exit >= 0` and `entry <= duration`, at `Max(entry, 0)`. The triangle's
time is the earliest over its pieces, the mesh's the earliest over the triangles whose boxes the path's own
box, grown by `r + o`, overlaps (tree order; equal times keep the lower triangle index). No piece's hit point is
ever tested against another piece. This is the box's form in module 1, and the lesson of its cylinder.

**Why no path passes between pieces or between triangles.** Every point within `r` of the triangle lies in a
piece whose conditions it meets with room to spare except possibly the radial one: a point nearest the face is
in the face piece at least `o` inside each edge plane; a point nearest an edge is in that edge's piece at least
`o` inside each end plane; a point nearest a vertex is in its sphere. So a path that comes within `r - e` of the
mesh is, at that moment, inside one piece by `e` in the radial condition and by `o = r / 8` in the others. The
interval arithmetic of a piece is off by a few rounding units of the coordinates, far below `tol / 8` (the
tolerance is 4096 rounding units of the largest coordinate), so that piece's interval is not empty and the hit
is reported. Neighbouring triangles share their edge and vertex pieces exactly (same doubles, same result), and
the face pieces overlap them. The pieces reach at most `Sqrt(1 + 1/64) r`, 0.8 % beyond the true grown triangle
(the sharp corners of the overlaps), so a hit is never reported farther than that from the mesh. Stated as the
guarantee the tests check against exact arithmetic: **a reported hit is no later than the first moment the path
is within `r - tol / 8` of the mesh, and the mesh is within `r + tol / 8` of the reported hit position; a path
reported free never comes within `r - tol / 8`.**

**Start cases** (the mesh's own; they replace P2 and P16, which need a convex solid). With
`at_start = ComputeDistanceToMesh(mesh, p0, 2 tol)`:

1. *Far start*, `d > 1.5 tol`: sweep with `r = tol`. The hit position has `d = tol`.
2. *Near start*, `d <= 1.5 tol`:
   a. `d < tol / 2`: a hit at 0 (defensive: `MoveParticle` never sweeps from there).
   b. the path pushes in, `Dot(normal at start, u) * duration < -tol`: a hit at 0, as P2.
   c. otherwise the *resting sweep*: `r = tol / 2`. The hit position has `d = tol / 2`, and `MoveParticle` moves
      it to the rest gap before the response (5.5).

Why two radii. A particle in contact rests at `d = tol`, exactly on the surface of the mesh grown by `tol`: a
sweep with `r = tol` from there would be decided by the last bits at every shared edge of a flat face (the
draft's per-triangle touching rule had this defect: entering the neighbour's grown edge at a height one part in
`1e7` under `tol` counted as a hit, again at every re-sweep). With `r = tol / 2` a resting start is `tol / 2`
outside everything the sweep can meet, a sliding path over a flat or convex part meets nothing, and a path that
truly closes on another part of the mesh is stopped while still `tol / 2` outside. Every threshold is at least
`tol / 4` from where `MoveParticle` puts a particle (`tol`, and up to `1.25 tol` after the stepping search), so
no case is decided by rounding. Along a free resting sweep `d >= tol / 2 - tol / 8`: stronger than module 1's
`d >= 0`.

The reported distance, normal and nearest point are `ComputeDistanceToMesh` at the hit position: the mesh's, not
one triangle's. `FindFirstHitOnSolid` gains `case SolidShape::kMesh`: a turning mesh goes to 5.7; otherwise
`ComputeOwnPath`, `FindFirstHitOnMesh`, and the hit turned back to world axes.

### 5.5 Start rules, the rest gap and the end check

Module 1 moves a point with `d < tol` along the normal to `d = tol` once (`MoveToRestGap`). In a concave corner
of a mesh that move can bring the point within `tol` of the neighbouring face. For a mesh the move is repeated:
along the mesh normal by `tol - d`, recompute, at most `kMaxMoveOutCount = 4` times, stopping when `d >= tol`.
For two faces meeting at an inside angle `theta` each move leaves a shortfall of at most `cos(theta)` times the
last, so from a start short of one face by `tol` four moves reach `tol / 2` (the end check's bound) for every angle
above 32.8 degrees (`cos^4 = 1/2`), and two moves reach `tol` at a right angle or wider. A sharper groove can end in
`kTooCloseToCall/kEndsTooCloseToSolid` from the end check, as two flat solids forming the same wedge can today.

A hit on a mesh is passed through the same move before the response (a resting sweep's hit is at `tol / 2`; a
far sweep's hit and a stepping search's hit are at `tol` or more and do not move). **The response uses the
normal recorded at the hit as found**, not the normal at the moved position: at the hit the surface just
entered is nearer than every other by `tol / 2`, while after the move two faces of a right-angled corner are
equally near and the nearest triangle would be chosen by rounding. The reaction's `hit_position` is the moved
position, as for the cylinder's rim. Start classification (`kStartsInsideSolid`, `kStartedInsideSolid`, the
silent move) is otherwise module 1's, with the mesh's `d`.

### 5.6 Slide, concave corners, and the hit limit

After a hit the particle joins the surface list with the recorded normal; the re-sweep runs 5.4 again over the
time left (`duration = step_duration - elapsed`, computed fresh before each re-sweep), against the same mesh
among the others. A mesh may be hit several times in one step, each time with a new normal: a particle in a
groove hits one side, slides, hits the other. Module 1's rule for earlier surfaces applies between two normals
of one mesh exactly as between two solids: approach to an earlier surface is removed without friction; when the
newest is then approached again (a corner sharper than a right angle) the particle takes the solid's velocity
and rests. "Approaches again" is compared with `ComputeLargestApproachLeftAfterResponse`, never with 0.
`kSqueezedBetweenSolids` compares two solids and cannot arise between two entries of one mesh.
`kSecondReactingSolid` compares solids, so several hits on one reacting mesh are one reacting solid; every
change counts in its reaction (P5).

The surface list keeps module 1's `kMaxHitsPerStep = 4` entries. When it is full and the new hit is on a mesh
that already has an entry, the oldest entry of that mesh is replaced (a normal of a part the particle has left);
otherwise a full list is `kTooManyHits` as today. This cannot let a particle in: the sweep keeps it out whatever
the list holds; the list only shapes the velocity.

Hits on meshes are also counted, all meshes together. A curved concave surface made of small triangles gives
one hit per crease the path turns into. At hit `kMaxMeshHitsPerStep + 1` (the 17th) the particle **rests for
the remainder of the step**: its velocity for this step becomes the solid's surface velocity at that point, its
velocity for the next step stays as the last response made it. It stays at the rest gap, loses at most the
travel left, and this is not a close call: shortening the step for one particle crossing many small triangles
would punish the whole scene for a dense mesh. `SolidContactResult::hit_count` counts every hit (its comment
changes: at most `kMaxHitsPerStep + kMaxMeshHitsPerStep + 1`).

### 5.7 A mesh that turns

- *First hit:* module 1's stepping search (Task 9) unchanged, with the gap `d(t) - tol` from
  `ComputeDistanceForParticle` at the pose of time `t`: the mesh distance is a true 1-Lipschitz distance, which
  is all the bound `|u| + |omega| * radius` needs (sphere tracing, Hart 1996, checked; conservative advancement,
  Zhang et al. 2006, checked). No start cases, as for every turning solid: a particle resting on it is hit at 0.
- *Slide:* module 1 carries the rest of the step in the solid's axes along a straight line (P7, Task 13). For a
  mesh that line is swept: in own axes the mesh is at rest, so the remainder runs 5.4 (a near start: the search
  stops at `d <= 1.25 tol`), the move of 5.5 and, at each further hit, the removal of the approach to the new
  normal and to the one before it (both gone: the particle rests on the mesh); at most `kMaxMeshHitsPerStep`
  hits, then rest. The carry changes the position only; the velocities stay as the first response made them, as
  for a box. The end is placed with the end pose. P7's `kTurningSolidNearOtherSolid` test is unchanged.
- A mesh has no spin: `ComputeShapeTurningAngularVelocity` already returns the whole angular velocity for any
  shape but a sphere and a cylinder, and `kMaxTurnPerStep` applies to it.

### 5.8 Motion

First version: **solids that only move and turn**: one mesh, built once, a pose per key through the solids
module's `DrivenPath` and `SolidMotion` unchanged; fixed, driven and free (A83).

A mesh that changes shape (A84, later) needs vertex positions at both ends of the step, the tree's boxes resized
over both, the surface velocity from the nearest triangle's vertex velocities, and a conservative first hit on a
moving triangle (Provot 1997; Wang et al. 2021, checked: floating-point root finders miss hits). Room is left:
`DistanceToMesh` returns `triangle_index` and `own_nearest_point`; the tree's shape does not depend on
positions; `MeshShape` can gain two arrays.

### 5.9 Thin parts: a limit of the first version (A94)

**What works whatever the thickness.** A particle never crosses a mesh, however thin, down to walls of
`1024 tol` (about a micrometre): contact is exact on the triangles (5.4). A wall at least as thick as the
pressure solve needs holds liquid: the corner distances then see a whole layer of cells inside it.

**The limit.** Below that thickness the grid does not see two sides. Liquid on the two sides of the part shares
cells and faces: a thin-walled container does not hold its level (the solve lets the grid flow through a wall
the particles cannot cross, and the liquid packs against it), a thin blade does not push the liquid as a thick
one would, and when the corner samples happen to see liquid on both sides inside one cell the run stops with
pressure's `kCellSplitBySolid`. Snow, sand and springy solids on the two sides exchange momentum through the
shared faces. Nothing is built to prevent this: no sides on mesh parts, no closed faces, no growth for the
grid. Module 1's plate keeps its approved rules (its two sides, and its thickness minimum with liquid or gas).

**The threshold.** `kMinWallThicknessInCells = 2.5` smallest cell widths (declared by the step, which owns the
check). Reason: it is the plate's existing rule (solids 5.9, the rule that keeps `kCellSplitBySolid` away:
thickness at least `(|n.x| + |n.y| + |n.z| + max(|n.x|, |n.y|, |n.z|))` cell widths, so that some layer of cells
has all eight corners inside). That factor is 2 for a wall along the axes and at most `Sqrt(6) = 2.449` (at
`n = (2, 1, 1) / Sqrt(6)`; the solids design quotes the looser 2.74). A mesh wall has every direction and may
turn, so the largest value is used, rounded up to 2.5. The width is the scene's smallest cell width, the width
of cells at a solid (grid 5.9); where the artist's refinement volume makes cells at the part larger, the real
need grows with them, and the message says which width was used.

**The check and the finding.** The builder measures `thinnest_wall` and where (5.2 check 7). The step's
`CheckSceneCanRun` compares it, per mesh solid, with `2.5 * smallest_cell_width` and adds one finding per mesh
below it: reason `CapabilityReason::kMeshPartTooThinForTheGrid` (NEW; I/O code `mesh_part_too_thin_for_the_grid`),
`solid_id`, `magnitude` = `thinnest_wall`, `limit` = the threshold, both in metres, suggested change
`kMakeSolidThicker`. I/O's message, with the object's name and the point in the object's coordinates:
"<object> is <magnitude> m thick near (<x>, <y>, <z>), less than the <limit> m (2.5 cells of <width> m) the grid
needs to keep its two sides apart. Particles still cannot cross it. Where liquid meets both sides, or the part
is the wall of a container, the liquid is wrong there or the run stops (thin_solid_inside_one_cell). Make the
part thicker or use smaller cells."

**The level: a known limit (`CapabilityLevel::kExperimental`, I/O severity `has_known_limit`), at every
thickness below the threshold; never blocking.** Re-derived now that nothing compensates:

- The earlier two levels (a note below one cell, blocking below a quarter cell) described a mechanism that
  weakened as the wall thinned. With no mechanism nothing changes at one cell or at a quarter: a wall of 2 cells
  and one of a tenth of a cell fail in the same way. One level.
- Not a note: a note is for something with no consequence. Here the result can be wrong, and a run can stop
  part-way. "It runs; a stated limit applies" is what `has_known_limit` means.
- Not blocking: the check knows the mesh, not what the liquid will do. A statue's fingers, a boat's bow, a fin
  in air, a spoon beside the splash are thinner than 2.5 cells in most scenes and are harmless; refusing them
  refuses most real meshes, and the owner twice declined refusal (A89, A94). Nothing unsafe follows from
  running: particles never cross, and the two bad outcomes are named in the message beforehand.
- The price, plainly: a scene whose thin part does separate liquid starts, and then gives a container that
  loses its level or stops with `kCellSplitBySolid`. The plate differs and stays blocking: it is declared as a
  separator, has one thickness, and its check is exact.
- The floor below which a mesh is refused is contact's own: `kPartTooThin` under `1024 tol`.

**Later.** The owner's recorded preference for a later piece is closed faces by side (A91, with A85, A92, A93):
a grid face is closed where the line between its two cell centres crosses the wall, one pressure per cell, two
sides for the exchange. The evidence is the survey `literature-thin-walls.md` (Guendelman 2005 for the form;
no paper shows it with these transfers). It binds nothing here, and nothing in this design is shaped for it
beyond what the triangles already give.

## 6. CPU and GPU

**Shared:** `mesh_shape.h`, `mesh_shape_steps.h`, `mesh_contact.h` entirely. Only `+ - * /`, `math::Sqrt`,
`math::Abs`, `math::Min`, `math::Max` and comparisons, the operations `base/scalar_math.h` promises the same
bits for, so distance and first hit on a mesh that does not turn give the same bits on CPU and GPU. A turning
mesh reaches `math::Rotation` through the pose and agrees within module 1's tolerance (contact section 6).

**Order is fixed:** nearer box first, ties the first box; triangles in array order; ties the lower index; pieces
in the order face, edges, vertices. No sum is made.

**CPU:** the builder (one thread; same bytes on every run) and all queries. I/O's CPU arrays serve
`BuildStartCells`, `BuildStartParticles` and the render-output worker.

**GPU:** `MeshShapeBuffers` holds five arrays for all meshes end to end and the table, written once with
`WriteBuffer` at run start, never read back, never rewritten in the first version. The per-particle function
is module 1's `ApplyMoveParticleAt`; `MoveParticlesView` gains the `MeshShapes` field; no new kernel. Each
search keeps `kMaxMeshTreeDepth = 24` indexes (96 B) on the thread's stack. The device code is compiled, not
run, until the owner's hardware run; the evidence until then is the `cpu` tests that run the per-particle
function for every thread block and thread. A thread near a mesh runs longer than its neighbours: the only new
cost shape.

**Range of cell blocks, batches:** contact reads no cell. Meshes are whole-run data present for every batch.

**CPU versus GPU tolerance:** module 1's, unchanged.

## 7. Tests

`tol = kTestContactTolerance = 0x1p-30` unless stated. Meshes are generated in the test from integer formulas
(`mesh_test_cases.h`). Every expected number below was derived by hand and, where marked (script), recomputed
with a short script on 2026-11-15.

Builder and measures (`cpu`):

- One test per `MeshProblem`: a cube less one triangle (`kNotClosed`, magnitude 3); one triangle turned over;
  a sliver of height `2^-21` of its edge; a wedge of `2^-7` rad; two cubes through each other; a fin `512 tol`
  thick (`CheckMeshShape`, `kPartTooThin`); a triangle height `512 tol` (`CheckMeshShape`).
- Same input twice gives the same bytes; a cube given inside out is turned over.
- Tree: for 1, 4, 5, 12, 1280 and 20480 triangles the deepest end box is at `ComputeMeshTreeDepth` (0, 0, 1, 2,
  9, 13; script), every box contains its triangles, end boxes hold at most 4.
- `thinnest_wall`: a slab 0.02 thick: 0.02 within `0x1p-48`, the point on a large face; a cube of side 1: 1.0; a
  regular tetrahedron: +infinity; a 45 degree wedge: thin toward its edge; a fin `1e-4` thick: `1e-4` within `0x1p-60`.

Distance (`cpu`):

- *Mesh cube equals the box.* 12 triangles, half size 0.5, fixed, moved and turned poses. At 10,000 generated
  points: inside, and outside in a face region, the distance equals the box's within `0x1p-48`, same normal;
  outside an edge or corner it equals `Length(max(|p| - 0.5, 0))` within `0x1p-48` (the box reports its largest
  face gap there; the difference is stated, not hidden). The inside test agrees with the box at every point.
- *Icosphere converges to the sphere.* Radius 0.5, the icosahedron's edges halved 3, 4 and 5 times with the new
  vertices put on the sphere: 1280, 5120, 20480 triangles. Largest `|d_mesh - d_sphere|` along the directions of
  the triangle centres at radii 0.4, 0.5, 0.52, 0.6: at most `1.02 * longest_edge^2 / (6 r)`, measured
  `2.264e-3`, `5.67e-4`, `1.42e-4` m (script), each step at least 3.9 times smaller than the last.
- Degenerate normal on a vertex, an edge and a face point; a nearest-triangle tie keeps the lower index;
  `largest_distance` cuts the search (distance returned equal to it; one box visited, counted by the test).
- The search never keeps more than `ComputeMeshTreeDepth` indexes (a counter in the test's copy of the walk
  order; meshes of the tree test and the size test).

First hit (`cpu`):

- *Mesh cube against the box* (far starts), fixed and moving, with `start_elapsed > 0`: a face hit at the box's
  time within `0x1p-44` relative (`(0.5 - tol) / 2` for the box's Task 8 case); an edge hit from `{-1, -1, 0}`
  at `{1, 1, 0}` at `0.5 - tol / Sqrt(2)`; a vertex hit from `{-1, -1, -1}` at `{1, 1, 1}` at
  `0.5 - tol / Sqrt(3)` (script); normals near an edge or vertex within `1e-5`.
- *Start cases.* On the cube's top face from `{-0.4, 0, 0.5 + tol}`: sliding at `{5, 0, 0}` for 0.1 s across
  both diagonals is free; `{5, 0, -1}` is a hit at 0; `{0, 0, -0.75 tol}` for 1 s is a hit at `2/3` with
  distance `tol / 2`; a start at `0.4 tol` is a hit at 0; a start at `1.2 tol` sliding is free.
- *Shared edges and vertices never leak* (the Review Focus of the plan). Generated paths aimed at points of
  shared edges and at vertices of the cube and of the 80-triangle icosphere, the aim moved by up to 4 rounding
  units per coordinate, and paths tangent to a grown edge within 4 rounding units, for `r = tol` and
  `r = tol / 2`: 40,000 cases, each compared with exact rational arithmetic on the same doubles (the smallest
  distance between a segment and a triangle, in a test-side helper): the guarantee of 5.4, both directions.
- Same bits for distance and first hit under the CUDA compile of the shared headers (`cpu` label in the
  container) and, on hardware, `SharedFormulas, OnGpu_MatchCpu` gains mesh inputs.

`MoveParticle` (`cpu`; they need module 1's Tasks 10 to 14):

- Mesh cube face and friction cases equal to the box's within `2 tol` and `0x1p-44`; a slide over the top
  face's diagonals has `hit_count` 1.
- *Right-angle groove.* The prism with cross-section `(-1,-1), (1,-1), (1,1), (0,0), (-1,1)`: from
  `{0.1, 0, 0.5}` at `{0, 0, -2}`, friction 0, 1 s: `hit_fraction (0.4 - Sqrt(2) tol) / 2`, `hit_count` 2,
  velocities 0, end `{0, 0, Sqrt(2) tol}` within `2 tol`. The same in a moved and turned pose and with the
  groove moving. A 60 degree groove: stops, no error, end distance at least `tol / 2`.
- Start rules on a mesh, each with a moving mesh: shallow overlap moved out, deep overlap refused, a start
  within the tolerance moved silently; a start in a right-angled inside corner at `d = 0.2 tol` from both faces
  reaches `tol` from both.
- *Hit limit.* Inside a 64-sided tube of inner radius 0.5, sliding at 2 m/s: 0.2 s crosses 8 creases: no rest,
  speed between 0.955 and 0.969 of the start (`cos(360/64 degrees)` to the powers 9 and 7; script); 0.5 s
  crosses more than 16: `hit_count` 17, the particle rests, end distance in `[tol / 2, 2 tol]`.
- *Concave shapes hold resting particles.* A bowl (inside of a half sphere, radius 0.5, 5120 triangles, 0.05
  thick), the right-angle and 60 degree grooves, a torus (0.5, 0.15): one particle at rest at the lowest point,
  gravity added by the test, 10,000 steps of `1e-3` s, each end fed back with `start_overlap_tolerance = 0`: no
  error; `d` in `[tol / 2, 2 tol]` every step; position within `1e-9` m of the start after the last.
- *Groove slide.* The right-angle groove tilted 30 degrees, friction 0: speed along the groove after 1 s is
  `9.81 * sin(30 degrees) = 4.905` m/s within 1 %; `hit_count` at most 2 per step.
- *Thin fin.* A fin `1e-4` m thick: a particle crossing it at 1,000 m/s is stopped on the start's side. (That
  the finding is raised is tested in the step's plan, from `thinnest_wall`.)
- *A fast particle never passes through.* 10,000 generated cases on the bowl, the torus and the fin: speeds up
  to 1,000 m/s, any direction, friction 0 and 0.5, the solid moving. Every end has `d >= tol / 2`; the naive end
  (no sweep) is inside in at least one case (module 1's helper).
- *Turning mesh.* The mesh cube in Task 9's turning-box case: `elapsed` equals `0.3 + acos((0.5 + tol) / 0.6)`
  within `0x1p-30`. An L-shaped mesh turning at 1 rad/s carries a particle resting in its inside corner for
  1,000 steps of `1e-3`: no error, `d` in `[tol / 2, 2 tol]`, the particle's own-axes position within `1e-9` m.
- `SceneTurnedRigidly_ResultTurnsWithIt` for the mesh cases, relative `0x1p-40`.

Size (`cpu`; timings are to calibrate on the first run, known-open form, A38): an icosphere of 1,310,720
triangles: `BuildMeshShape` in at most 10 s on one thread; `BuildMeshShapeBuffers` hands out exactly
96,469,760 bytes (script: 655,362 vertices, 1,048,575 tree boxes, each buffer rounded up to 256 B);
1,000,000 particles within half a cell of it moved in at most 4 times the time of the same particles against
the analytic sphere (the trigger for the fallback of 5.1).

Reference and proofs: no Python piece and no Lean proof. The one reference is the test-side exact rational
helper above; it removes the one uncertainty an analytic test cannot: whether rounding lets a path through
between pieces.

Comparison with `main`: none (nothing exists).

GPU (label `gpu`): every mesh move case through `MoveParticles` on `kGpu` against `kCpu`: exact for kinds,
indexes and counts; same bytes for positions and velocities when no solid turns; module 1's tolerance otherwise;
`SameInputTwice_SameBytes`. Without a GPU: all shared-header tests in the CUDA configuration, and
`ApplyMoveParticleAt` over every thread block and thread with the mesh cases.

Memory: `BuildMeshShapeBuffers, IntoRunMemory_HandsOutExactlyTheEstimate`; one piece short is `kOutOfMemory`.

## 8. Method paper section

`== Mesh solids <sec-mesh-solids>`, after `<sec-round-solids>`, before `<sec-contact-response>`. Outline: why
the triangles are the solid; distance and the inside test `<eq:mesh-distance>`; the mesh grown by a radius, the
seven pieces of a grown triangle and their intervals `<eq:mesh-piece-interval>`, why no path passes between
them; far and near starts and the resting sweep; the repeated move to the rest gap; several hits on one mesh
and the rest rule; a turning mesh; thin parts as a stated limit (5.9). Figures: `<fig-mesh-distance>` (a concave
cut with the three regions of the nearest point and the outward direction at a face, an edge and a vertex);
`<fig-mesh-grown-triangle>` (a triangle from above and in side cut with its face, edge and vertex pieces and
their overlaps, one path through the face piece and one through a shared edge caught by the edge piece);
`<fig-mesh-resting-sweep>` (side cut: the surface, the rest gap at `tol`, the resting sweep's radius `tol / 2`,
a path sliding over a shared edge, and one stopped by the other side of a groove). Pseudocode:
`<proc-mesh-distance>`, `<proc-mesh-first-hit>`. Sources: Baerentzen and Aanaes 2005 (the inside test); Ericson
2005 (nearest point on a triangle); Hart 1996 (the stepping search, which fills contact section 8's "citation
needed"); Wang et al. 2021 (why hits are found by intervals on convex pieces and a conservative search, not by
a root finder). None is in the old reference list.

## 9. Vocabulary

| Old name on `main` | New name | Source |
|---|---|---|
| (none) | `SolidShape::kMesh`, `mesh solid`, `SolidFace::kMeshTriangle` | NEW |
| (none) | `MeshShape`, `MeshShapes`, `MeshShapeArrays`, `MeshShapeBuffers`, `mesh_index`, `kNoMesh` | NEW |
| (none) | `MeshTreeBox`, "tree of boxes", "end box" (papers: bounding volume hierarchy, node, leaf; never say: BVH, node, leaf) | NEW |
| (none) | `triangle`, `vertex`, `neighbours`, `vertex_normals` | NEW |
| (none) | `closed` (every edge shared by exactly two triangles), as the refinement zone uses it | DECIDED (A78 section) |
| (none) | `thinnest_wall`, `thinnest_wall_point`, "wall (of a mesh)": two faces back to back (never say: sheet, shell, thin part as a type) | NEW; `wall` is already the cylinder's word, the glossary row gains this second use |
| (none) | `MeshProblem`, `kMeshNotUsable` | NEW |
| (none) | "mesh grown by a radius", "piece" (face piece, edge piece, vertex piece) (papers: offset, Minkowski sum with a ball; never say: offset surface, capsule, slab) | NEW |
| (none) | "far start", "near start", "resting sweep" (never say: touching rule, which stays the convex shapes') | NEW |
| (none) | `MeshSweep`, `MeshFirstHit`, `DistanceToMesh`, `largest_distance` (not `search distance`, which is `ComputeSolidVelocitySample`'s; not `reach`, which is a solid's) | NEW |
| (none) | `kMaxMeshHitsPerStep`, `kMaxMoveOutCount`, `kMeshNearStartInTolerances`, `kMeshRestingSweepShare`, `kMeshPieceOverlapShare`, `kMaxMeshTreeDepth`, `ComputeMeshTreeDepth` | NEW |
| (none) | `kMeshPartTooThinForTheGrid`, `kMinWallThicknessInCells` (step) | NEW |

Kept from module 1 and used as defined there: reach, search distance, swept distance, gap, rest gap, sweep, hit
fraction, approach left after the response. Checked against `docs/glossary.md`: "distance to the solid,
negative inside", never "signed distance"; "solid" never "obstacle" or "body"; "solid that only moves and
turns" never "rigid body"; "check" never "validate".

## 10. Not ported

Nothing exists on `main`. Deliberately not built in the first version: a distance volume (5.1); meshes that
change shape (5.8); open surfaces (blocking finding, A87); contact between two solids; every thin-wall
mechanism: sides on mesh parts, closed faces, growth for the grid, cells below the scene's smallest (5.9, A94);
overlapping closed parts in one mesh; automatic repair of bad meshes.

## 11. Decisions

| Answer | Decision |
|---|---|
| A80 | Mesh solids are module 1b; the grid does not deform. |
| A81 | Triangles and a tree of boxes are the solid for every module; a volume is the measured fallback (5.1). |
| A82 | The contact part (sections 3 to 7) is built right after module 1's Task 18; the I/O, solids, step and add-on parts land with those modules. |
| A83 | A mesh solid may be free; I/O computes mass, centre of mass and inertia; the too-close test uses the bounding box. |
| A84 | Meshes that change shape come later; room is left (5.8). |
| A86 | Blender exports the render-level mesh, with a note when triangles are under an eighth of a cell. |
| A87 | An open mesh is a blocking finding; a repair button comes later. |
| A88 | "Mesh" is the default shape of a solid that is not exactly a box, sphere or cylinder. |
| A94 | The simple route: meshes with real thickness only; a thinner part is a finding naming it (5.9); no thin-wall mechanism. |
| A85, A89 to A93 | Recorded as the preference for a later piece (closed faces by side, A91), with `literature-thin-walls.md` as evidence. They bind nothing in the first version. |

**Questions for the owner:** none. One choice made here that the owner may want to see: the thin-part finding
is a known limit and never blocks (5.9), while the plate's thickness rule blocks.

**Left to the plans, not the owner:** the measured rounding of the piece intervals against the `tol / 8` of
5.4 (the exact-arithmetic test decides; if it fails the overlap share or the bound changes, not the form); the
cost of coverage and of the surface field with a large mesh (the trigger for the fallback); whether sixteen
mesh hits per step is the right limit; the two timings of the size test.
