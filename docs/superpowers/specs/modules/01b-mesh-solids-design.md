# Mesh solids module design

Status: updated 2026-11-15 with the owner's answers A81 to A88 (section 11 "Decisions"; thin parts: 5.11, A85).
Module 1b of the rebuild (owner answer A80). No implementation plan yet. One question remains for the owner
(section 11, Q9). The edits other designs need are listed in `changes-to-apply.md`, section "01b Mesh solids
(A80)". Literature claims are marked
(checked) when the citation was confirmed on the web on 2026-11-15 and (memory) when not.

## 1. Purpose

A real scene's solids are not boxes, spheres and cylinders: they are meshes made in Blender. This module lets a
solid be a closed surface of triangles, of any shape (a bowl, a pipe, a statue), so that particles stop at it and
slide along it exactly as they do on a box, and so that every other module can ask the same questions of it:
how far is this point from the solid, which way is out, how fast is the surface moving there.

**Owns:** the mesh shape (`SolidShape::kMesh`): its triangles in the solid's own axes, the tree of boxes that finds
the triangles near a point or a path, the distance and normal to it, the inside test, the first hit of a path on
it, the slide along it (concave corners included), which side of a thin part a point is on (5.11), the checks that
a mesh is usable, the builder that turns
vertices and triangles into the arrays the solver reads, and the buffers that hold them on the run's processor.

**Does not own:** reading a mesh file, the scene file fields, the files kept for resume, mass, centre of mass and
inertia of a mesh (I/O); exporting the mesh from Blender, quads and subdivision surfaces (add-on); how a solid
moves from step to step, the force on it, writing sides onto faces and particles (solids); blocked areas (pressure,
filled by solids); everything module 1 owns. A mesh that changes shape over time (a character) is not in the first version (section 5.8).

On record from the same conversation (A80): the grid does not follow surfaces. Cubes stay cubes; the grid knows a
mesh only through numbers per cell, face and corner (blocked areas, the distance at corners), and the particles
carry where the material is.

## 2. What exists today

- `main`: nothing. Solids are analytic shapes only. The solids design says so twice: section 1 "solid meshes (none:
  analytic shapes only, as on `main`)" and section 10 "Solid meshes, contact between two solids" (not ported). The
  add-on design section 10 leaves "evaluated mesh shapes" for later; a mesh object marked as a solid is exported
  as its bounding box (add-on 5.1 item 2: "The shape is declared, not guessed from the mesh").
- Rebuild, module 1 (built: Tasks 1 to 6, `native/fluidity/contact/solid_shapes.h`, `contact_tolerance.h`,
  `contact_error.h`): six shapes behind one `Solid` (200 bytes by its field layout: 1 + 3 pad + 4, pose 96, motion
  48, half size 24, six `bool` + 2 pad, 8, 1 + 7 pad). `CheckSolid`, `ComputeDistanceToSolid` and
  `ComputeContactScale` each switch on `SolidShape` with every case written out. `ComputeSolidVelocitySample` loops
  over all solids and takes a pointer and a count. Nothing in a `Solid` can refer to data outside it.
- The only mesh input planned anywhere is the closed-mesh refinement zone (`plan-decisions.md` "Refinement volume",
  I/O plan Tasks 29b and 29d, add-on `zone_mesh.py`): a binary PLY of triangles, a closed test (every edge shared
  by exactly two triangles), and an inside test per lattice row by ray parity. It gives no distance, no normal, no
  nearest point, no tree and nothing on the GPU. Reused here: the PLY layout and reader, the add-on's evaluated-mesh
  export with the fan rule for quads and n-gons, `count_open_edges`, the mesh files in the scene digest, the
  pattern of a listed `inputs/` array file that a resume reads instead of rebuilding.
- Module 1's arguments that rest on convexity, which a mesh breaks: the touching rule (P2, P16: "the true distance
  of a convex solid is convex along a line"); "each convex solid is hit at most once per sweep" (design 5.7); the
  closed-form hit times; `kMaxHitsPerStep = 4` ("three sides of a corner plus one"); the straight slide in a
  turning solid's axes (P7); one move along the normal reaches `d = tol` (5.5 steps 2 and 6).

Z-up: nothing specific. A mesh is stored in the solid's own axes; the pose places it.

## 3. Public interface

Folder `native/fluidity/contact/` (the shape is a contact shape; contact still includes only `base/`), namespace
`fluidity::contact`. All names NEW. *shared* = compiles for CPU and GPU.

### 3.1 Changes to module 1's headers (all appended; no earlier value or offset moves)

```cpp
// solid_shapes.h
enum class SolidShape : std::uint8_t { kFlatSolid, kPlate, kBox, kDomainBox, kSphere, kCylinder,
                                      kMesh };            // NEW, appended (value 6).
enum class SolidFace : std::uint8_t { /* ...kHighRim, */ kMeshTriangle };   // NEW, appended (value 9).
inline constexpr std::uint32_t kNoMesh = 0xFFFFFFFFu;     // "No mesh", as an index into the mesh shapes.
struct Solid {
  // ... the existing fields, unchanged, then:
  std::uint32_t mesh_index = kNoMesh;   // kMesh only: index into the run's mesh shapes. Checked figure:
                                        // sizeof(Solid) is 200 bytes with 7 trailing padding bytes
                                        // (reports_reaction at offset 192); the index sits at offset 196, so
                                        // sizeof(Solid) stays 200 (static_assert) and no memory estimate moves.
};
struct PlateSide {                      // Gains one field in its padding; sizeof stays 8 (static_assert).
  std::uint32_t plate_id = 0;           // solid_id of the plate, or of the mesh the thin part belongs to.
  SolidSide side = SolidSide::kNoPlate;
  std::uint16_t part = 0;               // NEW (A85). 0 for a plate. Thin part of a mesh, counted from 1 (5.11).
};
// NEW (A85). False only when both name the same plate_id and part and their sides differ (both not kNoPlate).
// For the one unbounded plate this equals today's equality wherever a face is outside the solid.
constexpr bool IsOnOppositeSides(PlateSide first, PlateSide second);
// kMesh: half_size holds, per own axis, the largest |coordinate| of any vertex (so reach = Length(half_size)
// bounds the surface from the origin, as for a box). ToSolid sets it from the shape.
Solid ToSolid(const MeshShape& shape, std::uint32_t mesh_index, SolidPose pose, SolidMotion motion,
              std::uint32_t solid_id);

// contact_error.h, appended after kOutOfMemory:
//   kMeshNotUsable   A mesh failed CheckMeshShape; ContactFailure::magnitude holds the offending length or
//                    count and the new field `mesh_problem` says why.
enum class MeshProblem : std::uint8_t {
  kNone, kEmpty, kTooManyTriangles, kNotClosed, kFacesTurnedBothWays, kTriangleTooFlat, kEdgeTooSharp,
  kPartTooThin, kCrossesItself,
};
// ContactFailure gains `MeshProblem mesh_problem = MeshProblem::kNone;` (in existing padding). ToMessage and
// ToName cover MeshProblem. ContactCloseCall is unchanged.
```

### 3.2 `contact/mesh_shape.h` (shared)

```cpp
struct MeshTreeBox {                  // 32 bytes. One box of the tree; papers: a bounding volume hierarchy node.
  float low[3], high[3];              // m, own axes. Rounded outward, so the box always contains its triangles.
  std::uint32_t first = 0;            // count == 0: index of the first of two inner boxes (the second is +1).
  std::uint32_t count = 0;            // count > 0: an end box holding triangles [first, first + count), <= 4.
};
struct MeshShape {                    // One mesh on one processor. Borrowed arrays; never allocates.
  const Vec3* vertices = nullptr;             // m, own axes. vertex_count.
  const std::uint32_t* triangles = nullptr;   // 3 vertex indexes per triangle, counter-clockwise seen from
                                              // outside. Ordered so each end box's triangles are contiguous.
  const std::uint32_t* neighbours = nullptr;  // 3 per triangle: the triangle across edge (0,1), (1,2), (2,0).
  const Vec3* vertex_normals = nullptr;       // Unit, own axes: the triangle normals around the vertex, each
                                              // weighted by its angle there (Baerentzen and Aanaes 2005).
  const std::uint16_t* sheet_sides = nullptr; // NEW (A85). Per triangle: 0, or 2 * part - 1 (the part's
                                              // kAlongNormal face) or 2 * part (its kAgainstNormal face). 5.11.
  const MeshTreeBox* tree = nullptr;          // tree_box_count. Box 0 is the root.
  std::uint32_t vertex_count = 0, triangle_count = 0, tree_box_count = 0;
  Vec3 half_size{};                   // m. Largest |coordinate| of a vertex per own axis.
  double thinnest_part = 0;           // m. Smallest thickness the builder measured (5.2 check 7).
  double thinnest_sheet = 0;          // m. NEW (A85). Smallest thickness between two faces back to back (5.11);
                                      // +infinity when the mesh has no thin part.
  double smallest_triangle_height = 0;// m.
};
struct MeshShapes { const MeshShape* meshes = nullptr; std::uint32_t mesh_count = 0; };   // Passed by value.
inline constexpr int kMaxMeshTreeDepth = 48;         // The builder guarantees it; the search keeps 48 indexes.
inline constexpr int kMaxHitsPerMeshPerStep = 16;    // 5.6.
inline constexpr int kMaxMoveOutCount = 4;           // 5.5.

struct DistanceToMesh {
  double distance = 0;                // m. Positive outside, negative inside. Exact to rounding (5.3).
  Vec3 own_normal_out_of_solid{};     // Unit, own axes.
  Vec3 own_nearest_point{};           // m, own axes, on the surface.
  std::uint32_t triangle_index = 0;   // The nearest triangle; ties keep the lower index.
};
// `largest_distance`: the search gives up beyond it and returns distance == largest_distance (sign from the
// tree's root box: positive), so far points cost one box test. Pass +infinity for the true value everywhere.
DistanceToMesh ComputeDistanceToMesh(const MeshShape& mesh, Vec3 own_point, double largest_distance);
struct MeshFirstHit { double elapsed = 0; bool is_hit = false; DistanceToMesh at_hit{}; };
// First time own_start + own_velocity * t, 0 <= t <= duration, comes within contact_tolerance of the mesh (5.4).
MeshFirstHit FindFirstHitOnMesh(const MeshShape& mesh, Vec3 own_start, Vec3 own_velocity, double duration,
                                double contact_tolerance);

// The mesh-aware forms of module 1's point functions. The forms without `meshes` stay and are these with
// MeshShapes{}; given a kMesh solid without its table they assert (debug) and treat it as absent.
DistanceToSolid ComputeDistanceToSolid(const Solid& solid, SolidPose pose, Vec3 point, MeshShapes meshes);
SolidVelocitySample ComputeSolidVelocitySample(const Solid* solids, std::uint32_t solid_count, double elapsed,
                                               Vec3 point, double search_distance, MeshShapes meshes);
ContactScale ComputeContactScale(const Solid* solids, std::uint32_t solid_count, double step_duration,
                                 double smallest_cell_width, MeshShapes meshes);
ContactError CheckSolid(const Solid& solid, double step_duration, MeshShapes meshes, ContactCloseCall* close_call);
// SolidContactInputs and MoveParticlesInputs each gain one field: `MeshShapes meshes{};` (on the processor of the
// call). MoveParticle, FindFirstHit and MoveParticles keep their signatures.
```

```cpp
// NEW (A85). The side of a thin part of a mesh the point is on (5.11): from the nearest triangle within
// largest_distance. {0, kNoPlate, 0} when that triangle belongs to no thin part, when none is that near, or when
// the point is inside the solid. For a plate: {solid_id, FindSolidSide(...), 0}.
PlateSide FindPlateSide(const Solid& solid, SolidPose pose, Vec3 point, double largest_distance, MeshShapes meshes);
```

`FindSolidSide` is unchanged (plates only). `SolidVelocitySample::side` is filled from `FindPlateSide` for a mesh
(same search, no second one). `ComputeSolidPose`, `ComputeSolidVelocity`, `ComputeContactResponse` are unchanged.

### 3.3 `contact/mesh_shape_builder.h` (CPU only), `mesh_shape_builder.cc`

```cpp
struct MeshShapeArrays {              // Owns one mesh's arrays in CPU memory. Movable.
  std::vector<Vec3> vertices; std::vector<std::uint32_t> triangles, neighbours;
  std::vector<Vec3> vertex_normals; std::vector<std::uint16_t> sheet_sides; std::vector<MeshTreeBox> tree;
  Vec3 half_size{}; double thinnest_part = 0; double thinnest_sheet = 0; double smallest_triangle_height = 0;
};
MeshShape ToMeshShape(const MeshShapeArrays& arrays);          // The borrowed form, CPU memory.
inline constexpr std::uint32_t kMaxMeshTriangles = 1u << 24;   // 16.8 million per mesh.
struct MeshCheckSettings { double contact_tolerance = 0; bool checks_self_crossing = true;
                           double sheet_thickness_limit = 0; };  // m. 0: label no thin part (5.11).
// Builds the arrays from vertices (m, own axes) and triangles, then runs the checks of 5.2. Same input, same
// bytes. Fails kMeshNotUsable (mesh_problem, magnitude), kBadInput (not finite, an index out of range).
std::expected<MeshShapeArrays, ContactFailure> BuildMeshShape(std::span<const Vec3> vertices,
    std::span<const std::array<std::uint32_t, 3>> triangles, const MeshCheckSettings& settings);
// The checks alone, for the step's capability check when the tolerance of the run is known.
ContactError CheckMeshShape(const MeshShape& mesh, double contact_tolerance, ContactFailure* failure);

class MeshShapeBuffers;   // The run's meshes on run_on(): one buffer per array kind, all meshes end to end, plus
                          // the MeshShape table. Members: run_on(), MeshShapes meshes() const.
std::expected<MeshShapeBuffers, ContactFailure> BuildMeshShapeBuffers(RunOn run_on,
    std::span<const MeshShapeArrays> meshes, RunMemory& run_memory);        // kOutOfMemory, kRunOnCallFailed
std::vector<MemoryEstimateEntry> BuildMeshShapeMemoryEstimate(std::uint64_t vertex_count,
    std::uint64_t triangle_count, std::uint64_t tree_box_count, std::uint32_t mesh_count);
```

Memory rows (all `kRunOn`, `kWholeRun`, given counts): "contact: mesh vertices" 24 B, "contact: mesh triangles"
12 B, "contact: mesh neighbours" 12 B, "contact: mesh vertex normals" 24 B, "contact: mesh sheet sides" 2 B,
"contact: mesh tree" 32 B, "contact: mesh table" `sizeof(MeshShape)`. With a closed mesh (vertices about half the
triangles) and at most 4 triangles per end box (tree boxes about half the triangles): **about 60 B per triangle**:
6 MB for 100,000 triangles, 60 MB for one million. I/O keeps the same arrays in CPU memory for the CPU-only consumers ("io: mesh solids", `kCpu`).

## 4. Consumes from other modules

**Foundation (EXISTS):** as contact section 4; in addition `math::Min`, `math::Max`, `Buffer`, `BuildBuffer`,
`WriteBuffer`, `RunMemory`, `MemoryEstimateEntry`.

**Contact (module 1; PROVIDED when its Tasks 8 to 16 are built as section 5.9 asks):** `Solid`, `SolidPose`,
`SolidMotion`, `DistanceToSolid`, `SolidVelocitySample`, `ContactScale`, `SolidContactInputs`, `FirstHit`,
`MoveParticle`, `FindFirstHit`, `MoveParticles`, `ContactBuffers`, the private dispatch points of
`solid_contact_steps.h` (5.9), the stepping search toward a turning solid, the response, the surface list.

**Nothing else.** I/O, solids, the step and the add-on consume this module (section 5.10).

## 5. Method

Notation: `tol` the contact tolerance, `d` the distance to the solid (negative inside), `p` a point in the solid's
own axes, `u` the particle's velocity for this step minus the solid's, in own axes.

### 5.1 Representation: the choice

| | (a) distance volume | (b) triangles and a tree (recommended) | (c) both |
|---|---|---|---|
| What it is | `d` sampled on a lattice in own axes, read by interpolation. Papers: signed distance field, level set; sparse form: OpenVDB (Museth 2013, checked), NanoVDB (Museth 2021, checked) | The triangles, exact nearest-point and path queries through a tree of boxes (Ericson 2005 5.1.5, memory; Karras 2012, checked) | Volume for grid-side numbers, triangles for particles |
| Error at the surface | `h^2 x curvature / 8` on smooth parts, of order `h` at edges (own derivation; no citable bound found). At `h = 2 mm`: micrometres to millimetres, 10^5 to 10^8 tolerances. The solid becomes "what the lattice says", not the mesh | Rounding only, about 100 `epsilon` x largest coordinate: the same order as module 1 and far below `tol` | Two answers to "where is the surface", differing by (a)'s error |
| Thin parts | A part thinner than about two samples vanishes or leaks. Production practice thickens it: Houdini advises volume plus surface collision because thin colliders leak; Bifrost asks for a thickness of one voxel or more (both checked) | Kept down to 1024 `tol` (about a micrometre) | Particles stop; the volume side still loses it |
| Memory, 2 m object | Dense: 32 MB at 1 cm, **4.0 GB** at 2 mm. Near the surface only (3 samples each side, 12 to 48 m2 of surface, real block filling): 6 to 30 MB at 1 cm, **150 to 700 MB** at 2 mm | **6 MB** per 100,000 triangles, **60 MB** per million, whatever the cell size | The sum |
| Cost per particle per step | 8 reads; far particles one read | Far particles (the usual case): one box test. Near: 20 to 60 box tests and 4 to 20 triangle tests (memory; to measure) | As (b) for particles |
| Same result on CPU and GPU | Same bits (reads, `+ - *`) | Same bits: only `+ - * /`, `Sqrt`, comparisons, a fixed visiting order | Same bits |
| Mesh that changes shape | A volume per key, seconds each at 2 mm, uploaded per key | New vertex positions and the boxes resized per step; the tree's shape is kept | Both |
| Build time | Distance at every kept sample: seconds to a minute | Sort and box build: well under 10 s for a million triangles on one CPU thread (to measure) | The sum |
| What replaces convexity | Nothing exact: published practice pushes particles out along the gradient after the fact (Bridson 2015, memory); misses and leaks with fast or thin solids are documented (Houdini docs, checked) | A triangle is convex, so module 1's arguments hold per triangle (5.4) | As (b) |
| "No particle ends a step inside" | Holds only for the lattice's solid, and only with a search that bounds the interpolant's slope (up to `Sqrt(3)`) | Holds for the mesh itself | Holds |

**Decision (A81): (b), the triangles are the solid, for every module.** Reasons, in order: (1) it is the only
choice that keeps module 1's tolerance and its safety property unchanged; the surveyed no-miss results in the
literature are all exact or conservative tests on the geometry (Wang et al. 2021, checked: floating-point root
finders miss; Brochu et al. 2012, checked), never push-out from a volume. (2) Three other modules need the true
distance, not an interpolated one: viscosity reads it into `1 / d` coefficients and tests `|d| <= 1e-9 w`
(viscosity plan Task 10); the grid's size rule and the start grid rely on `d` being 1-Lipschitz (grid 5.9, step
5.15); the open-to-sky flood needs a distance that never overestimates (solids 5.4 step 5). An interpolated
lattice guarantees none of these. (3) Memory is about a hundredth of a 2 mm volume and does not grow when the
artist asks for smaller cells. (4) One representation: one set of tests, no rule for which answer wins.

What is given up: the tree search is more code on the GPU than eight reads (a fixed 48-index list per thread, a
loop of variable length), and grid-side stages (coverage at every corner near the solid) pay tree searches where a
volume would pay reads. The owner's liking for inputs "passed as a volume, calculated elsewhere on CPU" (A78) is
kept in spirit, the arrays are built once on the CPU outside the solver and passed as data, but not in form.
(c) stays possible later behind `ComputeDistanceToMesh` without changing an interface, only for points farther
than `2 w` from the surface where no consumer needs exactness; it is not designed here. Decided: A81.

### 5.2 Building and checking a mesh (CPU, once per mesh)

Steps: 1 copy vertices; drop none, merge none (the exporter's job). 2 Checks 1 to 4. 3 Order triangles by the
Morton code of their centre and build the tree top-down by splitting the longest axis at the median, end boxes of
at most 4 triangles, depth at most `kMaxMeshTreeDepth` (a balanced median tree of 2^24 triangles has depth 22).
4 Neighbours across each edge. 5 Vertex normals. 6 Checks 5 to 8. 7 `half_size`.

Checks, in this order; the first failure is reported with the lowest triangle index:

1. `kEmpty` (no triangle), `kTooManyTriangles` (above `kMaxMeshTriangles`).
2. `kNotClosed`: every edge is shared by exactly two triangles (magnitude: open or over-shared edges).
3. `kFacesTurnedBothWays`: the two triangles of every edge run along it in opposite directions; then the summed
   volume (`Dot(a, Cross(b, c)) / 6` over triangles) is positive, else the builder turns every triangle over once.
4. `kTriangleTooFlat`: every triangle's smallest height is at least `1024 tol` and at least `2^-20` of its longest
   edge (below that the nearest-point arithmetic loses the digits the tolerance needs).
5. `kEdgeTooSharp`: the angle between the two faces of every edge, measured inside the solid, is at least
   `2^-6` rad (0.9 degrees). Sharper, the inside test's dot product at a distance `tol` falls toward rounding.
6. `kCrossesItself` (when `checks_self_crossing`): no two triangles that share no vertex intersect (tree against
   tree, exact predicates on the CPU). A self-crossing mesh has no inside.
7. `kPartTooThin`: `thinnest_part >= 1024 tol` (module 1's `kMinFeatureInTolerances`). `thinnest_part` is the
   smallest, over triangles, of the distance from the triangle's centre along its inward normal to the first
   triangle met (a ray through the tree). It is a measure, not a proof; the proof is the fin test of section 7.
8. Reported, never failing (I/O turns them into NOTE findings): `thinnest_part` and the largest and median edge
   length, for the comparisons with the cell width of 5.10.

### 5.3 Distance, normal and the inside test

`ComputeDistanceToMesh`: depth-first search of the tree from the root, nearer inner box first (equal: the first),
skipping a box whose distance to `p` is not below the best found (start: `largest_distance`). At an end box, the
nearest point of each triangle by the region form (Ericson 5.1.5): vertex, edge or face. Keep the smallest
squared distance; ties keep the lower triangle index. Then:

- Outward direction at the nearest point `c`: the triangle's normal when `c` is in its face; the sum of the two
  triangles' normals (through `neighbours`) on an edge; `vertex_normals` at a vertex. The point is inside exactly
  when `Dot(p - c, that direction) < 0` (Baerentzen and Aanaes 2005, checked: correct for a closed surface that
  does not cross itself, given the true nearest point).
- `distance = +-Length(p - c)`; `own_normal_out_of_solid = +-(p - c) / Length(p - c)`; when `p == c` as computed
  (degenerate normal), the direction above, normalised.

This is the true distance, like the sphere's and the cylinder's and unlike the box's: 1-Lipschitz, exact to
rounding. Only `+ - * /`, `Sqrt` and comparisons: same bits on both processors for a given pose.
`ComputeDistanceToSolid(..., meshes)` turns the world point into own axes with the pose, calls it with
`largest_distance = +infinity`, turns the normal back and reports face `kMeshTriangle`.
`ComputeSolidVelocitySample(..., meshes)` passes `largest_distance = max(search_distance, 0)` grown by the best
distance already found among earlier solids, so a mesh far from the point costs one box test.

### 5.4 First hit on a mesh that does not turn

In own axes the mesh is at rest and the path is the straight segment `p0 + u t`. A hit is `d == tol`, as for every
shape. Seen from outside, `d` is the smallest of the distances to the triangles, so the first hit on the mesh is
the earliest first hit on any triangle **grown by `tol`** (the triangle's points and everything within `tol` of
them). That grown triangle is convex and its surface has three kinds of pieces, each with a closed form module 1
already has:

- the two flat faces at `+-tol`: the flat solid's hit time (5.3 step 1), accepted when the hit point projects
  inside the triangle;
- a cylinder of radius `tol` around each edge: the cylinder wall's quadratic (P15), accepted between the edge's
  two ends;
- a sphere of radius `tol` at each vertex: the sphere's quadratic with its approach condition (5.7).

This is the classic test of a moving sphere against a triangle (Ericson 2005, 5.5.6, memory), with the sphere's
radius equal to the tolerance. Candidates come from the tree: every end box that the segment's own box, grown by
`tol`, overlaps, visited in tree order. The earliest accepted time wins; equal times keep the lower triangle
index. The reported distance, normal and nearest point are `ComputeDistanceToMesh` at the hit position (so they
are the mesh's, not one triangle's).

*Touching rule, per triangle (replaces P2 and P16).* A triangle is convex, so its distance is convex along a
line: `d_triangle(t) >= d_triangle(t0) + rate * (t - t0)`, with `rate = Dot(normal toward p at t0, u)`. Hence:
a triangle whose grown shape the path is in or enters at `t0` with `rate * (duration - t0) >= -tol` is **not hit**
in this sweep (the path cannot come nearer than `d_triangle(t0) - tol`); otherwise it is hit at `t0`. At the
start (`t0 = 0`, gap in `[-tol, 0]`) this is P2 word for word. Applied at any entry it also removes the false
hits a tessellated flat face would give: a particle sliding at `d = tol` over two triangles in one plane crosses
their shared edge with `rate` zero up to rounding. Since `d = min over triangles`, the mesh's distance along the
sweep is at least `d(t0) - tol` too; `MoveParticle` starts every sweep at `d = tol`, so the path never comes
nearer than 0 up to rounding, and the move-out of 5.5 restores `tol`. A start with a gap below `-tol` is a hit at
0 (defensive, as P2).

### 5.5 Start rules, settling and the end check

Module 1 moves a point with `d < tol` along the normal to `d = tol` once. In a concave corner of a mesh that
move can bring the point within `tol` of the neighbouring face. For a mesh the move is repeated: move along the
mesh normal by `tol - d`, recompute, at most `kMaxMoveOutCount = 4` times, stopping when `d >= tol`. For two
faces meeting at an inside angle `theta` each move leaves a shortfall of at most `cos(theta)` times the last, so
four moves reach `tol / 2` (the end check's bound) for every angle above about 33 degrees and reach it at once at
a right angle or wider. A sharper groove is `kTooCloseToCall/kEndsTooCloseToSolid` from the end check, as two
flat solids forming the same wedge are in module 1 today (its `AcuteWedge` test stops the particle instead; so
does 5.6's corner rule here before the end check is reached). Start classification (`kStartsInsideSolid`,
`kStartedInsideSolid`, the silent move) is otherwise module 1's, with the mesh's `d`.

### 5.6 Slide, concave corners, and what the hit limit means

After a hit the response uses the mesh normal at the hit (5.4) and the particle joins the surface list with
**that normal**; the re-sweep runs 5.4 again over the time left, against the same mesh among the others. A mesh
may therefore be hit several times in one step, each time with a new normal: a particle in a groove hits one
side, slides, hits the other. Module 1's rule for earlier surfaces applies between two normals of one mesh exactly
as between two solids: approach to an earlier surface is removed without friction; when the newest is then
approached again (a corner sharper than a right angle) the particle takes the solid's velocity and rests.
`kSqueezedBetweenSolids` cannot arise between two entries of the same solid (they do not close on each other): for
them the corner rule's stop is final. `kSecondReactingSolid` compares solids, so several hits on one reacting mesh
are one reacting solid; every change counts in its reaction (P5).

`kMaxHitsPerStep = 4` keeps its meaning **between solids** (hits on different solids, and the first hit on each).
Hits on one mesh are counted separately, up to `kMaxHitsPerMeshPerStep = 16`: a curved concave surface made of
small triangles gives one hit per crease the particle crosses, about `travel / triangle size` per step, and the
step's travel limit is half a cell. At the 17th hit on one mesh the particle **rests for the remainder of the
step**: its velocity for this step becomes the solid's surface velocity at that point, its velocity for the next
step is left as the last response made it. This is safe (it stays at `d = tol`), costs at most the travel left
(under half a cell, once), and is not a close call: shortening the step for a particle crossing many small
triangles would punish the whole scene for a dense mesh. `SolidContactResult::hit_count` saturates at 255.

### 5.7 A mesh that turns

- *First hit:* module 1's stepping search (5.3 step 2, Task 9) unchanged, with `g(t) = d(t) - tol` from
  `ComputeDistanceToMesh` at the pose of time `t`: the mesh distance is a true 1-Lipschitz distance, which is all
  the bound `|u| + |omega| * radius` needs (the idea is sphere tracing, Hart 1996, checked, on the path seen from
  the solid; conservative advancement, Zhang et al. 2006, checked). `largest_distance` is the bound times the time
  left, so a far mesh costs one box test per advance. No touching rule, as for every turning solid.
- *Slide:* module 1 carries the rest of the step in the solid's axes along a straight line (P7). In those axes a
  mesh is at rest and the line is straight, so the remainder is swept with 5.4, 5.5 and 5.6 in own axes (hits on
  the same mesh allowed), then placed with the end pose. P7's `kTurningSolidNearOtherSolid` test is unchanged.
- A mesh has no spin: every turn changes its shape. `kMaxTurnPerStep` applies to the whole turn.

### 5.8 Motion

First version: **solids that only move and turn**: one mesh, built once, a pose per key through the solids
module's `DrivenPath` and `SolidMotion` unchanged; fixed, driven and free (A83).

A mesh that changes shape (a character) later needs: vertex positions at the start and end of the step; the
tree's boxes resized over both; the surface velocity taken at the nearest point from its triangle's three vertex
velocities; a first hit of a point on a moving triangle, where published floating-point root finders miss hits
(Provot 1997; Wang et al. 2021, checked) and a conservative search is required. The interface leaves room without
redesign: `DistanceToMesh` already returns `triangle_index` and `own_nearest_point`; `MeshShape` gains
`const Vec3* vertices_at_step_end` and `const Vec3* vertex_velocities` (null today); the stepping search is the
conservative form with the vertices' largest speed added to its bound; the tree's shape (`first`, `count`) does
not depend on positions. `ComputeSolidVelocity(solid, pose, point)` is the one signature that would then be too
narrow for a mesh; consumers that want a surface velocity already go through `ComputeSolidVelocitySample`.

### 5.9 Where module 1 must leave room now

Module 1's remaining tasks need **no new feature and no public signature change**. They need three habits, so
that 1b adds a `case` and a field instead of rewriting them:

| Task | What to do differently |
|---|---|
| 7 response | Nothing. It works on a normal. |
| 8 first hit, solids that do not turn | (1) Every private helper in `solid_contact_steps.h` that evaluates a solid takes `const SolidContactInputs& inputs` and a `solid_index`, not a bare `const Solid&`: in particular `ComputeDistanceForParticle(inputs, solid_index, pose, point, plate_side)` (the plan's signature, with its trailing `PlateSide`, P3). 1b's table arrives as a field of `inputs` with no call site touched. (2) `FindFirstHit` loops over solids and calls one private `FindFirstHitOnSolid(inputs, solid_index, start, velocity, start_elapsed, duration, FirstHit*)` whose `switch (solid.shape)` writes every case out, with no `default` that computes (an unknown shape asserts and is `kBadInput`). (3) The touching rule lives inside each shape's branch, never in the loop. |
| 8a sphere and cylinder | Same three points; nothing else. |
| 9 turning solids | The stepping search gets `g(t)` only through `ComputeDistanceForParticle` and decides "turning" only through `ComputeShapeTurningAngularVelocity`. It must not assume the distance belongs to one face. |
| 10 `MoveParticle` | The surface list stores, per entry, the solid index **and the normal recorded at the hit**, and has `kMaxHitsPerStep` entries; nothing assumes a solid appears once. "A convex solid is hit once per sweep" stays a consequence of the touching rule, never a line of code that skips the solid just hit. |
| 11 start rules, and step 6 of 10 | "Move the point to `d = tol`" is one private helper, `MoveToRestGap(inputs, solid_index, pose, point)`, used by step 2 and step 6. For the six shapes it is one move along the normal. |
| 12 slide between surfaces | Nothing: it reads recorded normals (already so, amended A74). Keep `kSqueezedBetweenSolids` a comparison of two **solids**. |
| 12a round solids | The rim move stays inside the cylinder's branch. |
| 13 slide on a turning solid | The carry in the solid's axes is one private function taking `inputs` and `solid_index`, so 1b can sweep that remainder against a mesh. |
| 14 generated safety tests | Put the check "every end distance is at least `tol / 2`, and the naive control enters" in a helper of `contact_test_cases.h` that takes any list of solids; 1b runs it on meshes. |
| 15 `MoveParticles`, CPU | Nothing in signatures (`MoveParticlesInputs` gains a field later; mesh arrays are not in `ContactBuffers`). Add `static_assert(sizeof(Solid) == 200)` beside the estimate (the checked figure: 200 bytes, 7 trailing padding bytes, so a 4-byte index fits at offset 196). |
| 16 `MoveParticles`, GPU | The per-particle GPU function receives its scalar inputs as **one struct** (the device copy of the inputs), not a positional list, so a `MeshShapes` field is one line. Leave about 400 B of thread stack unassumed (the tree search's 48 indexes and the surface list). |
| 17, 18 | Nothing. Task 18's hand-over names the dispatch points above. |

Tasks 1 to 6 (built) need nothing now: 1b appends `kMesh`, `kMeshTriangle`, `mesh_index` and the mesh-aware
overloads; the existing `static_assert`s on enum values already force appending.

### 5.10 Cross-check of every other module

"Change": text or code of that module must change. "Verify": believed fine, to confirm when 1b's plan is written.
The edits themselves are in `changes-to-apply.md` "01b Mesh solids (A80)".

**02 Grid**

| Place | Status | What |
|---|---|---|
| 5.9 size rule, `CellSizeFacts::distance_to_solid` (3.8), plan Task 13a | No change | A formula on a number others fill. Relies on `d` being 1-Lipschitz: true for (b), not for a lattice. |
| `CornerValues` (3.6, 3.7) | No change | Holds any corner distance. |
| Tests | No change | |

**03 Particles**

| Place | Status | What |
|---|---|---|
| `PlateSide` fields (3.1, 3.5; 8 bytes, unchanged size) | No change | Carries the new `part`. |
| `FindNeighbourhood` mask (3.5, plan Task 9: "outside_solid and face_side == side"); merge "only when `same_side`" | Change | The test becomes `!IsOnOppositeSides` (5.11): a particle or face with no side exchanges with either side, as beyond a thin part's rim. Same result for the plate. |
| 5.6 "a child that lands inside a solid is handled by the next `MoveParticles`" | Verify | Holds: 5.5 keeps `kStartedInsideSolid`. A child deeper than `start_overlap_tolerance` inside a thin part is `kStartsInsideSolid` as today. |

**04 Transfer**

| Place | Status | What |
|---|---|---|
| Stages, `SolidsAtFaces`, 5.4 step 4 (unit normal) | No change | The mesh normal is unit by construction. |
| 5.1 `same_side` ("outside_solid && face side == particle side"), plan Task 6 `FindSameSideFaces`; 5.4 fill sources "side `s`" | Change | `!IsOnOppositeSides(face side, particle side)` (5.11). With it, particles on the two faces of a thin mesh part no longer exchange through shared grid faces: CPIC's rule. Test: two particle clouds either side of a mesh sheet a quarter cell thick, one at rest, one moving along it: the resting cloud's momentum stays exactly 0. |
| Decision 5, `kMaxSidesAtFace = 2`, `kTooManySidesAtFace` | Verify | A face near a rim can meet particles with side A, side B and none. Whether "none" needs a third stored value or joins both is for the transfer plan; near two thin parts closer than `2 * kSolidReachInCells` cells, solids reports the limit (08). |
| Plan Task 5 fixture `BuildSolidsAtFaces` | Change (small) | Calls the mesh-aware `ComputeSolidVelocitySample`; one test with a mesh cube equal to the box. |

**05 Pressure**

| Place | Status | What |
|---|---|---|
| `SolidCoverage`, open areas, volumes (5 steps 3, 4) | No change | Generic over corner distances ("any sampled field"). |
| `kCellSplitBySolid` (5 step 8, decision 7), `plan-decisions.md` Pressure lines 88-89 "a solid thinner than a cell that passes between corners is not seen. Solids' plate-thickness rule is what keeps that case out." | Change (text), no change (method) | Sides do not reach the solve: one pressure per cell, one velocity per face (5.11). The sentence becomes "Solids' thickness rule, for the plate and for every thin part of a mesh (`thinnest_sheet`), keeps that case out". Two cell equations per split cell stay not built (decision 7). |
| One distance per corner; sharp mesh edges inside a cell | Verify | Blocked area converges as for the box; add `ComputeSolidCoverage, MeshCube_EqualsBox` and an icosphere twin of `Sphere_BlockedAreaConverges` (solids' tests). |

**06 Viscosity**

| Place | Status | What |
|---|---|---|
| 3 `ViscosityInputs::solids`; 4; plan Task 10: `ComputeSolidVelocitySample(solids, count, elapsed, point, search_distance)` | Change | Takes `MeshShapes` (a field beside `solids`) and calls the mesh-aware form. |
| Plan Task 10 rule (a), `(u_P - v_solid) / distance`, on-surface test `1e-9 w` | No change with (b) | Needs the true distance; a reason for 5.1's choice. |
| "Inclined" test on the normal (plan 853-855); `has_inclined_solid` (solids 3.4) | Change | A mesh is inclined whatever its pose: solids' summary must say so. Until viscosity Task 21 lifts `kInclinedSolidNotSupported`, a viscous liquid with a mesh solid is excluded. |
| Memory | No change | Nothing scales with the mesh. |
| Tests | Add | Tilted plate (Task 21) as a mesh slab within the same 5 %. |

**07 Materials**

| Place | Status | What |
|---|---|---|
| All (reads `ParticleContact::kind`, interpolated coverage, friction) | No change | |
| `PlatesInScene`, `has_plate_side` (3) | Change (small) | True also when a mesh has a thin part, so snow and springy solids carry a side. |

**08 Solids** (the module most changed)

| Place | Status | What |
|---|---|---|
| 1 and 10: "solid meshes (none)" | Change | Removed; points here. |
| 3.2 `SolidSettings` (shape, `rest_pose`, `half_size`) | Change | Gains `std::uint32_t mesh_index = contact::kNoMesh`; `SolidsSettings` gains `contact::MeshShapes meshes` (on `run_on`). |
| 5.1 `BuildTry`, `BuildContactSolids` (switch on shape) | Change | Copies `mesh_index`; the `kMesh` case. |
| 5.4 `ComputeSolidsAtFaces`, `ComputeSolidCoverage` (distance at every corner place, capped at 2 cells), GPU halves (plan Task 18) | Change | Mesh-aware calls with `largest_distance` = the cap; kernels receive `MeshShapes`. Cost to measure (the hybrid is the fallback, A81). |
| 5.4 step 5 open-to-sky flood; 5.2 travel radius `Length(half_size)` | No change | True distance; `half_size` bounds the mesh from the origin. |
| `kSolidsTooClose` (plan Task 10: box corners and centre, sphere centre) | Change | A free mesh: sampled at the corners and centre of the mesh's bounding box (`half_size`), as for the cylinder; this reports too close earlier, never later. Sampling every vertex was considered and is too costly. |
| Free solid: "origin is the centre of mass", `inertia_in_own_axes` principal | Change in I/O | I/O moves the mesh to its centre of mass and principal axes at build (12 below). |
| 5.4 step 1 `ComputeSolidsAtFaces`: `side` from `FindSolidSide` for the scene's plate | Change | `side` from `FindPlateSide` of the nearest solid (plate or mesh), reach `kSolidReachInCells * w`. |
| 5.9 `ApplyPlateSides` (written once, never changed), `ApplyPlateRemoval` | Change | For a mesh the side is rewritten every step from `FindPlateSide` at the particle (5.11: contact keeps a particle from crossing, so it can only change through "no side", around the rim). The plate's once-only rule is unchanged. |
| 5.9 "Supported thickness", `ComputeMinPlateThickness`, `CheckSolidsCanRun` when `has_liquid_or_gas` | Change | The same bound applies to `thinnest_sheet` of every mesh: blocking finding `mesh_sheet_too_thin_for_liquid` (5.11). Without liquid or gas: no minimum. NOTE findings: two thin parts closer than `2 * kSolidReachInCells` cells; triangles much smaller than a cell (5.2 check 8). |
| `has_inclined_solid` | Change | True for any mesh. |
| `kMaxPlateCount = 1`, cutting by a driven plate | No change | Still one plate. A thin mesh part separates material the same way but is not a cutting tool in the first version (no `ComputeCutWork` for it). |
| Memory (plan Task 13 "solids: solids copy") | No change | `sizeof(Solid)` unchanged; mesh rows are contact's (3.3). |
| Tests | Add | Mesh twins of `Box_BlockedAreaConverges`, `Hydrostatic_PushIsWeightOfDisplacedLiquid`, a driven mesh paddle. |

**09 Gas**

| Place | Status | What |
|---|---|---|
| All (sees solids through `SolidCoverage`, `solid_volume_rate`, open-to-sky marks; markers through `MoveParticles`) | No change | Test fixtures use functions, not shapes. |

**10 Step**

| Place | Status | What |
|---|---|---|
| 3.2 `ReadyScene`; stage 16 `MoveParticlesInputs`; stage 11 `ViscosityInputs`; `FoamAndSprayInputs` | Change | The step owns one `contact::MeshShapeBuffers` for the run and fills the `meshes` fields. |
| 5.14 `CheckSceneCanRun`, stage 3 and 19 `CheckSolid` | Change | Mesh-aware `CheckSolid`, `ComputeContactScale`; `CheckMeshShape` once with the run's tolerance; new `CapabilityReason` for `kMeshNotUsable`. |
| 5.15 `BuildStartCells` (distance per candidate cell per solid, CPU; "four corners inside one solid") | Change (call), verify (rule) | Mesh-aware call. The four-corner test can be wrong for a non-convex solid (a face spanning a groove): add the face centre as a fifth point. |
| 5.9 `BuildRunPlan` | Change | Adds `BuildMeshShapeMemoryEstimate`. |
| 5.8 `kSolidMotion`, `kSolidTurn` | No change | Use `half_size` and angular speed. |
| Stages, order, commit and reject | No change | As A74: "no new stage or name". |

**11 Cell resize and batches**

| Place | Status | What |
|---|---|---|
| Size rule from `SolidCoverage` corners; refinement volume | No change | |
| `ResizeCells` step 6: `ComputeSolidsAtFaces` on the new grid, all blocks | Verify | Cost with a mesh (tree searches at corners near the solid only). |
| `BuildBatchPlan` whole-grid rows | Change | Mesh rows are whole-run, on the processor in every batch: counted like `kSolids` whole-grid rows; a mesh is never split per batch. |

**12 I/O and CLI**

| Place | Status | What |
|---|---|---|
| Scene file `solids[]` (plan Task 26; strict, unknown fields are errors) | Change | `shape` gains `mesh`; new field `mesh_file` (binary PLY, the zone format of Task 29d); scene schema version raised. |
| New `io/mesh_solid.h` (CPU only) | Change | `ReadMeshSolid` (PLY into vertices and triangles, scale applied), calls `contact::BuildMeshShape`, maps `MeshProblem` to findings with the field path; volume, centre of mass and inertia by the closed-form sums over triangles (Mirtich 1996, checked; Eberly 2002, checked), mesh re-expressed about the centre of mass and principal axes for a free solid. |
| Run folder, resume (5.3, 5.4; "resume never reads the scene file again") | Change | `inputs/mesh_solids` (FluidiTy array file with every built array), listed, read on resume, never rebuilt; mesh files in the fingerprint. Not in `ready_scene` (2 GB limit). |
| `BuildStartParticles` ("at least w/4 from every solid") | Change (call) | Mesh-aware distance. |
| Memory: `BuildIoMemoryEstimate`; budget text 5.10 | Change | "io: mesh solids" (`kCpu`); the planned-cost lines show mesh bytes. |
| Inertia closed forms (5.1) | Change | The mesh case above. |
| Reuse | - | PLY reader and layout, closed count, the `inputs/` pattern. The ray-parity raster is **not** reused: it answers inside or outside on the refinement lattice only. |
| Tests | Add | Read, build, write, resume without the PLY, a cube's inertia `m / 3 * (b^2 + c^2)`, an open mesh. |

**13 Surface, foam and spray**

| Place | Status | What |
|---|---|---|
| 5.3 surface field: `ComputeDistanceToSolid` at every lattice point, CPU, per frame; 5.5 foam; 5.6 foam volume | Change (call), verify (cost) | Mesh-aware calls with `largest_distance` = 4 spacings (the field's own cap), so only points near the mesh search the tree. CPU arrays from I/O. |
| `FrameInputs::solids`, `MotionSample::solids` | Change | Carry `MeshShapes` (CPU). |
| Solid motion writers | No change | Pose only. |

**14 Blender add-on**

| Place | Status | What |
|---|---|---|
| 3.2 `Shape`; 5.1 item 2 "declared, not guessed"; 10 "evaluated mesh shapes" not ported | Change | `Shape.MESH` for the `SOLID` role, the default for any object that is not exactly a box, sphere or cylinder (A88); the render-level evaluated mesh (A86). Export: the evaluated mesh (`evaluated_get(depsgraph).to_mesh()`, modifiers and subdivision applied; Blender API, checked), quads and n-gons by the fan rule, through `zone_mesh.py`'s writer, to `inputs/solids/solid_<id>.ply`; object scale applied to vertices. |
| Findings (3.4) | Change | `solid_mesh_not_closed` (blocking, with the open edge count), `solid_mesh_deforms` (blocking in the first version: a vertex moves in object coordinates between frames), `mesh_sheet_too_thin_for_liquid` (blocking, from the native check, A85) and `solid_mesh_triangles_small` (NOTE: triangles under an eighth of a cell, A86), `solid_mesh_too_many_triangles`. |
| Per-frame pose keys, `motion_not_keys` | No change | |
| Digest | Change | Covers the solid mesh files, as it covers zone meshes. |
| Tests | Add | `test_mesh_solid_exports_evaluated_triangles` (a subdivided cube: triangle count of the evaluated mesh), closed or not, deforms. |

**15 Sand**

| Place | Status | What |
|---|---|---|
| Hit split (reads `normal_out_of_solid`, pose, velocity), `solids_on_run_processor` | No change | |
| Sand beside a thin mesh part | Verify | Sand has no pressure solve, so sides separate it fully (5.11); one test in the sand plan. |
| 5.4 "hit point = start + velocity * duration" | Verify | With several hits on one mesh in a step, the record holds the first; as for several solids today. |

**Shared files.** `shared-interface.md` (b) and (c) Contact: the new names of 3.1 to 3.3. `tradeoff-particles.md`
1.3: no allowance for solid geometry exists; with (b) one million triangles are 60 MB of 13.05 GB (0.5 %), so
no verdict of 1.6 changes.

### 5.11 Thin parts of a mesh (A85)

The owner's answer: "I think that is covered in cpuc of mls mpm", that is CPIC, compatible particle-in-cell, of
Hu et al. 2018 (reference [3]). The paper could not be fetched on 2026-11-15 (the file is larger than the fetch
limit), so the description of CPIC below is **from memory** and must be checked against the paper when the plan
is written; the description of the rebuild is from its designs and code.

**What CPIC is (memory).** An explicit particle-grid method with no pressure solve. Each grid point near a thin
surface holds, per nearby solid, whether it is near (within the reach of the weights and with a nearest point on
the surface) and on which side (the sign of its offset from the nearest surface element along that element's
normal). Each particle gets the same from the grid points around it and keeps its side while it stays near.
A particle and a grid point are *compatible* unless they are both near the same solid on different sides.
Particle-to-grid and grid-to-particle exchange happens only between compatible pairs; for an incompatible grid
point the particle sees the solid's velocity there instead (with friction), and what it would have given goes to
the solid as an impulse. A particle found on the wrong side is pushed back by a penalty force, which is not a
guarantee. Beyond the edge of a thin surface no point is "near", so material flows round it.

**What the rebuild does for the plate, and how faithful it is.** Contact: `PlateSide {plate_id, side}` per
particle and `FindSolidSide` (the side of the one unbounded plate, from its analytic distance). Solids:
`ApplyPlateSides` writes each particle's side once; `ComputeSolidsAtFaces` gives each face centre its side from
the plate's pose at the start and end of the step. Transfer: a particle and a face exchange only when
`outside_solid && face side == particle side`, with weights renormalised over those faces (5.1); the fill reads
sources of the same side (5.4). That is CPIC's compatibility, with four simplifications: the side comes from the
analytic plate, not from a field on the grid; a dropped face is left out of the weights instead of being replaced
by the solid's velocity; the particle's side never changes; and a swept first hit replaces the penalty, so a
particle is never on the wrong side (`kOnWrongSideOfPlate` is an error, not a force). Pressure has no part in it:
a plate beside liquid or gas must be thick enough that a whole layer of cells lies inside it
(`ComputeMinPlateThickness`, 2 to 2.74 cell widths), so no cell holds liquid on both sides and
`kCellSplitBySolid` never fires for a plate. So today sides separate snow and springy solids at a plate of any
thickness, and liquid only at a thick one.

**The same mechanism on a mesh.**

1. *Which triangles form a thin part* (builder, once; needs `sheet_thickness_limit`, which I/O sets to
   `ComputeMinPlateThickness` for the smallest cell width, 2.74 cells at most). From each triangle's centre a ray
   goes inward along its normal (the ray of check 7). The triangle is a *sheet triangle* when the first triangle
   met is nearer than the limit and faces the other way (`Dot` of the two normals at most `-0.5`): two faces back
   to back. A wedge, a cone's tip or a box's edge is not a sheet (the normals are not opposed). Sheet triangles
   joined across edges whose two normals have `Dot` above `0.5` form a patch; a patch and the patch its rays
   meet are the two faces of one *thin part*, numbered from 1 in order of lowest triangle index; the face holding
   that triangle is `kAlongNormal`. `sheet_sides` holds the result; `thinnest_sheet` the smallest such distance.
   More than 32,767 parts is `kMeshNotUsable/kTooManyTriangles`.
2. *The side of a point* (`FindPlateSide`): the nearest triangle within `largest_distance` (the consumer's reach,
   `kSolidReachInCells * w`), read from the search `ComputeDistanceToMesh` already made; its `sheet_sides` entry
   gives `{solid_id, side, part}`. No side: farther than the reach, inside the solid, or nearest to a triangle of
   no thin part. This is CPIC's rule (the nearest surface element decides) without the grid field.
3. *Near the rim*, where the two faces meet, the nearest triangle is a rim triangle (no part) or the last triangle
   of either face. A point there has no side or either side, changing within about one thickness. This is why
   the exchange test must be `!IsOnOppositeSides` and not equality: "no side" exchanges with both, which is CPIC's
   "not near, so compatible", and lets material turn the corner.
4. *Near two thin parts* (a slot between two sheets): a point has one side, of the nearer part. Two points with
   sides of different parts count as not opposite, so exchange across the farther sheet is not stopped. CPIC
   keeps a side per nearby solid and does not have this limit; the rebuild keeps one `PlateSide` per particle
   (8 bytes, as for the plate). Reported by solids as a NOTE when two thin parts are closer than
   `2 * kSolidReachInCells` cells.
5. *Particles.* The side of a particle near a mesh is recomputed every step (stage `FindSolidSurfaces`), not
   kept: contact never lets it cross the sheet, so its side can only change by passing through "no side" round
   the rim. `kOnWrongSideOfPlate` stays the plate's.

**What this gives, and what it does not.**

- Materials with no pressure solve (snow, springy solids, sand): complete. Particles on the two faces of a mesh
  sheet of any thickness down to `1024 tol` neither cross (contact) nor exchange momentum (sides). This is
  CPIC's own scope.
- Liquid and gas, particle-to-face and face-to-particle exchange: also separated by sides.
- Liquid and gas, **the pressure solve: not helped at all.** CPIC has no pressure solve and gives nothing here.
  The solve has one pressure per cell and one velocity per face; a face cut by a sheet has liquid of both sides
  on it, and a cell cut by a sheet is one unknown. Two things then happen with a sheet thinner than the bound.
  Seen by the corner samples with liquid on both sides in one cell: `kCellSplitBySolid`, the run stops. Passing
  between corners: the blocked area is zero, the solve lets the grid flow through a surface the particles
  cannot cross, and **a thin-walled container does not hold its liquid's pressure** even with liquid on one side
  only. Sides cannot repair either: they would need two cell equations per split cell (pressure decision 7, not
  built) and two velocities per cut face.

**First version.** Built: 1 to 5 above (contact, now: 2 bytes per triangle, one ray per triangle at build,
nothing per query beyond the search already made); the exchange test in transfer and particles; solids' face and
particle sides. Recorded as a limit: in a scene with liquid or gas, a mesh whose `thinnest_sheet` is below
`ComputeMinPlateThickness` is a **blocking** finding, `mesh_sheet_too_thin_for_liquid`, with the measured
thickness, the bound and the part's position. Blocking, not a note, for three reasons: it is the rule the plate
already has (Q38); the alternative is a run that stops midway with `kCellSplitBySolid` or a container that
silently leaks pressure; and the artist has two cheap repairs the finding names (thicken the part to the bound,
or shrink the cells near solids). This replaces the earlier recommendation (a note under one cell, blocking
under a quarter): under the bound the solve is wrong at any thickness, so there is no thickness at which a note
is honest. What the earlier recommendation keeps: thin parts never block a scene of snow, sand or springy solids
only. Lifting the limit for liquid is Q9 (section 11).

## 6. CPU and GPU

**Shared:** `mesh_shape.h` entirely. The same code on both processors; only `+ - * /`, `Sqrt`, `Abs`, `Min`,
`Max` and comparisons, no function of the last-bits group, so distance and first hit on a mesh that does not turn
give the same bits on CPU and GPU. A turning mesh reaches `math::Rotation` through the pose and agrees within
module 1's tolerance (contact section 6).

**Search order is fixed:** nearer box first, ties the first box; triangles in array order; ties the lower index.
No sum is made, so there is no order of sums.

**CPU:** the builder (single thread; same bytes on every run) and all queries. I/O's CPU arrays serve
`BuildStartCells`, `BuildStartParticles` and the render-output worker.

**GPU:** `MeshShapeBuffers` holds five arrays for all meshes end to end and the table, written once with
`WriteBuffer` at run start (one planned copy), never read back, never rewritten in the first version. The
per-particle GPU function keeps a list of 48 tree indexes on its stack. Work is one thread per particle, as module
1; a thread near a mesh runs longer than its neighbours, which is the only new cost shape.

**Range of cell blocks, batches:** contact reads no cell. Meshes are whole-run data present for every batch.

**CPU versus GPU tolerance:** module 1's (contact section 6), unchanged.

## 7. Tests

`tol = kTestContactTolerance = 0x1p-30` unless stated. Meshes are generated in the test from integer formulas.

Analytic, distance (`cpu`, `no_gpu`):

- *Mesh cube equals the box.* 12 triangles, half size 0.5. At 10,000 generated points: inside, and outside in a
  face region, `ComputeDistanceToSolid` equals the box's within `0x1p-48`, same normal; outside an edge or corner
  it equals the Euclidean distance `Length(max(|p| - 0.5, 0))` within `0x1p-48` (the box reports its largest face
  gap there; the difference is stated, not hidden). Inside test agrees with the box at every point.
- *Icosphere converges to the sphere.* Radius 0.5, vertices on the sphere, 1280, 5120 and 20480 triangles. Largest
  `|d_mesh - d_sphere|` over 10,000 points within 0.1 of the surface: at most `1.5e-3`, `3.7e-4`, `9.5e-5` m (the
  triangle centre's sag, `edge^2 / (6 r)`), and each step at least 3.5 times smaller than the last (second order).
- Degenerate normal on a vertex, an edge and a face point; nearest-triangle tie keeps the lower index.
- Thin parts (A85): a slab 0.02 thick with limit 0.05: its two large faces are parts' sides `kAlongNormal` and
  `kAgainstNormal` of part 1, its rim has none, `thinnest_sheet == 0.02` within `0x1p-48`; a cube of side 1 and a
  cone: no thin part, `thinnest_sheet` infinite; `FindPlateSide` 0.01 off each face, beyond the rim (no side) and
  beyond the reach (no side); `IsOnOppositeSides` truth table (same part and different sides only).
- `BuildMeshShape`: one test per `MeshProblem` (a cube less one triangle: `kNotClosed`, magnitude 3; one triangle
  turned over; a sliver of height `2^-21` of its edge; two cubes through each other; a fin `512 tol` thick); same
  input twice gives the same bytes.

Analytic, first hit and move:

- Mesh cube against Task 8's box cases: face hits at the same `elapsed` within `0x1p-44` relative; an edge hit
  within `(Sqrt(2) - 1) tol / speed` (the box grows with sharp edges, the mesh with round ones).
- Flat face of 2 x 2 x 2 triangles, particle sliding at `d = tol` across every shared edge at 5 m/s: free (no
  hit), end distance in `[tol / 2, 2 tol]`.
- *Concave shapes hold resting particles.* For each of: a bowl (inside of a half sphere, radius 0.5, 5120
  triangles, 0.05 thick), a V groove of 90 and of 60 degrees, a torus (0.5, 0.15; particle in the hole's lowest
  point): one particle at rest at the lowest point, gravity added by the test, 10,000 steps of `1e-3` s, each end
  fed back with `start_overlap_tolerance = 0`: no error; `d` in `[tol / 2, 2 tol]` every step; position within
  `1e-9` m of the start after the last.
- Groove slide: frictionless 90 degree groove tilted 30 degrees; from rest the particle's speed along the groove
  after 1 s is `g sin(30) = 4.905` m/s within 1 %; `hit_count` at most 2 per step.
- Bowl slide: 2 m/s along the bowl's inside, 1,000 steps: no error, never rests by the 16-hit rule when the
  travel per step is under 8 triangle edges; a control with travel of 40 edges per step rests and stays at
  `d >= tol / 2`.
- *Thin fin.* A fin `1e-4` m thick (2 triangles per side and a rim), smallest cell 0.01: a particle crossing it
  at 10 m in one step is stopped (hit reported, end on the start's side).
- *A fast particle never passes through.* 10,000 generated cases (P11's generator) on the bowl, the torus and the
  fin: speeds up to 1,000 m/s, any direction, friction 0 and 0.5. Every end has `d >= tol / 2`; in the same test
  the naive end (no sweep) is inside in at least one case (the control, as module 1's Task 14).
- *Turning mesh.* The mesh cube in Task 9's turning-box case: `elapsed` equals `0.3 + acos((0.5 + tol) / 0.6)`
  within `0x1p-30`. An L-shaped mesh turning at 1 rad/s carries a particle resting in its inside corner for 1,000
  steps of `1e-3`: no error, `d` in `[tol / 2, 2 tol]`, the particle's own-axes position within `1e-9` m.
- `SceneTurnedRigidly_ResultTurnsWithIt` for the mesh cases, relative `0x1p-40`.

Size (`cpu`, and the same on the GPU run): an icosphere of 1,310,720 triangles: `BuildMeshShape` in at most 10 s
on one thread and 79 MB handed out (`ComputeRunMemoryBytes` of the estimate, exact); 1,000,000 particles within
half a cell of it moved in at most 4 times the time of the same particles against the analytic sphere. Both
timings are to calibrate on the first run (known-open test form, A38) and are the trigger for the hybrid
(A81) if missed.

Reference and proofs: none. Every expected value is a closed form or the analytic twin.

Comparison with `main`: none (nothing exists).

GPU (label `gpu`): every move case through `MoveParticles` on `kGpu` against `kCpu`: exact for kinds, indexes and
counts; same bytes for positions and velocities when no solid turns; module 1's tolerance otherwise;
`SameInputTwice_SameBytes`. Without a GPU (`no_gpu`): all shared-header tests in the CUDA configuration.

Memory: `BuildMeshShapeBuffers, IntoRunMemory_HandsOutExactlyTheEstimate`; one piece short is `kOutOfMemory`.

## 8. Method paper section

`== Mesh solids <sec-mesh-solids>`, after `<sec-round-solids>`, before `<sec-contact-response>`. Outline: why the
triangles are the solid (the tolerance); distance and the inside test `<eq:mesh-distance>`; the triangle grown by
the tolerance and its three kinds of pieces `<eq:mesh-hit-time>`; the touching rule per triangle with its
convexity argument; the repeated move out of a corner; several hits on one mesh and the rest rule; a turning
mesh; thin parts and their sides, with what the pressure solve still cannot see (5.11). Figures: `<fig-mesh-thin-part>`
(a sheet in cut: its two faces, the rim with no side, a face centre and a particle on opposite sides, and a cell the
sheet splits); `<fig-mesh-distance>` (a concave cut with the three regions of the nearest point and the outward
direction at a face, an edge and a vertex); `<fig-mesh-grown-triangle>` (side cut of one triangle grown by `tol`,
a path through the flat piece, one into an edge's cylinder, and a sliding path crossing a shared edge without a
hit); `<fig-mesh-groove>` (a particle in a groove: hit, slide, second hit, rest). Pseudocode:
`<proc-mesh-distance>`, `<proc-mesh-first-hit>`. Sources: Baerentzen and Aanaes 2005 (the inside test); Ericson
2005 (nearest point on a triangle, moving sphere against a triangle); Hart 1996 (the stepping search, which fills
contact section 8's "citation needed"); Wang et al. 2021 (why hits are found by closed forms on convex pieces and
a conservative search, not by a root finder on a moving triangle). None is in the old reference list.

## 9. Vocabulary

| Old name on `main` | New name | Source |
|---|---|---|
| (none) | `SolidShape::kMesh`, `mesh solid` | NEW |
| (none) | `MeshShape`, `MeshShapes`, `MeshShapeArrays`, `MeshShapeBuffers` | NEW |
| (none) | `mesh_index`, `kNoMesh` | NEW |
| (none) | `MeshTreeBox`, "tree of boxes", "end box" (papers: bounding volume hierarchy, node, leaf; `leaf` is banned) | NEW |
| (none) | `triangle`, `vertex`, `neighbours`, `vertex_normals` | NEW |
| (none) | `closed` (every edge shared by exactly two triangles), as the refinement zone uses it | DECIDED (A78 section) |
| (none) | `thinnest_part`, `MeshProblem`, `kMeshNotUsable` | NEW |
| (none) | "grown triangle" (papers: offset, Minkowski sum with a ball) | NEW |
| (none) | `kMaxHitsPerMeshPerStep`, `kMaxMoveOutCount`, `MoveToRestGap` (private, module 1) | NEW |
| (none) | `thin part`, `sheet triangle`, `sheet_sides`, `thinnest_sheet`, `PlateSide::part`, `FindPlateSide`, `IsOnOppositeSides` (papers: CPIC's colour or tag, and compatibility) | NEW (A85) |
| `same_side` (equality of two `PlateSide`s) | `same_side` now means "not on opposite sides of the same plate or thin part" (`!IsOnOppositeSides`); glossary row to amend | DECIDED word, meaning widened (A85) |
| (none) | `largest_distance` (not `search distance`, which is `ComputeSolidVelocitySample`'s; not `reach`, which is a solid's) | NEW |

Checked against `docs/glossary.md`: "distance to the solid, negative inside" is used, never the banned
"signed distance" (given once as the papers' word); "solid" never "obstacle" or "body"; "solid that only moves and
turns" never "rigid body"; "wall" is not used (a groove has sides); "near the surface only" never "band"; "check"
never "validate"; "saved state" never "checkpoint".

## 10. Not ported

Nothing exists on `main`. Deliberately not built in the first version: a distance volume (5.1); meshes that
change shape (5.8); open surfaces (a single sheet with no inside: blocking finding, A87); contact between two solids; a
plate side for thin mesh parts (cutting stays the plate's); automatic repair of bad meshes (the checks report,
the artist fixes).

## 11. Decisions

**Q1. Where 1b sits.** Decision: (a) the contact part (sections 3 to 7) right after module 1's Task 18, before the
grid; the I/O, solids, step and add-on parts land with those modules. Owner answer A82.

**Q2. Free mesh solids in the first version.** Decision: (b) yes; I/O computes mass, centre of mass and inertia
and re-centres the mesh; the too-close test uses the bounding box. Owner answer A83.

**Q3. The triangles are the solid.** Decision: (a) triangles and a tree of boxes for every module; the hybrid
with a volume stays the measured fallback (5.1). Owner answer A81.

**Q4. Meshes that change shape.** Decision: (a) later, as their own piece; the interface leaves room (5.8).
Owner answer A84.

**Q5. A mesh part thinner than one cell.** The owner pointed to CPIC (Hu 2018). Decision: sides on thin mesh
parts, as designed in 5.11, in the first version; they separate snow, sand and springy solids fully and the
particle-face exchange of liquid; the pressure solve is not helped, so a sheet below the plate's thickness bound
in a scene with liquid or gas is a blocking finding. Owner answer A85; the remaining choice is Q9.

**Q6. Which mesh Blender exports for a subdivision surface.** Decision: (a) the render-level evaluated mesh, with
a NOTE when triangles are smaller than an eighth of a cell. Owner answer A86.

**Q7. An open mesh.** Decision: blocking finding "not closed" first; a repair button in the add-on later. Owner
answer A87.

**Q8. Default shape of a mesh object marked Solid.** Decision: (b) "Mesh" for any object that is not exactly a
box, sphere or cylinder. Owner answer A88.

**Still for the owner**

**Q9. Thin-walled containers of liquid (a glass, a pipe, a hull) when the wall is thinner than about two cells.**
Sides cannot make the pressure solve see such a wall (5.11). Options: (a) first version as designed: blocking
finding, the artist thickens the wall or shrinks cells near solids. (b) Grow thin parts for the grid only: the
corner distances that feed blocked areas treat a sheet as if it were as thick as the bound, while particles still
stop at the true surface (the practice of production solvers: Bifrost asks for a collider thickness of a voxel or
more, checked). The liquid's grid boundary is then up to about a cell off the true wall near thin parts, and
particles in that zone need a rule; this is a piece of work in solids and pressure, after both exist. (c) Two
cell equations per split cell and two velocities per cut face: exact, and a change to grid, pressure, viscosity
and gas (pressure decision 7 set it aside). Recommendation: (a) now, (b) as the next piece once pressure and
solids are built, because thin-walled containers are among the most common real scenes; (c) only if (b) proves
visibly wrong. Changes: with (b) or (c) the blocking finding becomes a note or disappears.

**Not settled here (for the plan, not the owner):** the rounding bound of the nearest-point arithmetic and the
two thresholds of checks 4 and 5 (to derive and prove with a generated test); the cost of coverage and of the
surface field with a large mesh (to measure; trigger for the hybrid); whether sixteen hits per mesh per step is
the right limit; the exact GPU stack need; CPIC as published, read from the paper itself (5.11 is from memory);
how transfer stores a face's value when particles with side A, side B and no side all reach it (transfer
decision 5).
