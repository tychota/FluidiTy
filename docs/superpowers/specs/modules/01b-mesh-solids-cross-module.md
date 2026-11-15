# Mesh solids: what every other module changes

Companion of `01b-mesh-solids-design.md` ("01b" below), final for the simple route (A94). Status of every row:
**to apply**, when that module's design and plan are next touched (A82), except module 1 (contact), whose rows
are applied by the 01b plan itself. "No change" rows are kept so the check is on record. Nothing here serves
thin walls: no sides on mesh parts, no closed faces, no growth. The same list, as edits, is in
`changes-to-apply.md` "01b Mesh solids (A80)".

Names used: `contact::MeshShapes`, `MeshShape`, `MeshShapeArrays`, `MeshShapeBuffers`, `BuildMeshShape`,
`CheckMeshShape`, `BuildMeshShapeBuffers`, `BuildMeshShapeMemoryEstimate`, the mesh-aware
`ComputeDistanceToSolid`, `ComputeSolidVelocitySample`, `ComputeContactScale`, `CheckSolid` (01b 3.1 to 3.4).

## 01 Contact (applied by the 01b plan)

| Design section | Plan task | What |
|---|---|---|
| 3.1 | 01b plan Task 2 | `ContactError::kMeshNotUsable`, `MeshProblem`, `ContactFailure::mesh_problem`. |
| 3.2 | 01b plan Tasks 2, 7 | `SolidShape::kMesh`, `SolidFace::kMeshTriangle`, `kNoMesh`, `Solid::mesh_index`; `CheckSolid`'s `kMesh` case; "Requires: no solid is a `kMesh`" on the forms without `MeshShapes`. |
| 3.4, 3.5 | 01b plan Tasks 7, 15 | `MeshShapes meshes` in `SolidContactInputs` and `MoveParticlesInputs`; `SolidContactResult::hit_count`'s comment. |
| 1, 5.5, 5.7 | 01b plan Task 18 | "Owns" names the mesh shape; "each convex solid is hit at most once" gains "a mesh may be hit several times (01b 5.6)"; step 4 gains the move of a mesh hit to the rest gap. |
| Plan Tasks 9 to 16 | (their "Room for module 1b" notes) | Already written into the contact plan. One number there is superseded: Task 16's "48 indexes" is 24 (`kMaxMeshTreeDepth`). |

## 02 Grid

| Design section | Plan task | What |
|---|---|---|
| 5.9 size rule, `CellSizeFacts::distance_to_solid` (3.8) | Task 13a | No change. One sentence to add in 5.9: the rule relies on a 1-Lipschitz distance, which the mesh distance is (01b 5.3). |
| `CornerValues` (3.6, 3.7); tests | - | No change. |

## 03 Particles

| Design section | Plan task | What |
|---|---|---|
| 3.1, 3.5 `PlateSide`; `FindNeighbourhood`; merge by side | Task 9 | No change: sides stay the plate's. |
| 5.6 "a child that lands inside a solid is handled by the next `MoveParticles`" | - | No change; one sentence pointing at 01b 5.5 (the move is repeated in a concave corner). |

## 04 Transfer

| Design section | Plan task | What |
|---|---|---|
| Stages, `SolidsAtFaces`, 5.1 `same_side`, 5.4 | Task 6 | No change. |
| Test fixture `BuildSolidsAtFaces` | Task 5 | Calls the mesh-aware `ComputeSolidVelocitySample`; new test `BuildSolidsAtFaces, MeshCube_EqualsBox`. |

## 05 Pressure

| Design section | Plan task | What |
|---|---|---|
| `SolidCoverage`, open areas, volumes (5 steps 3, 4) | - | No change: generic over corner distances. |
| 5 step 8, decision 7 `kCellSplitBySolid`; `plan-decisions.md` "Pressure": "a solid thinner than a cell that passes between corners is not seen. Solids' plate-thickness rule is what keeps that case out." | (text only) | The sentence gains: "A mesh has no such rule in the first version (A94): a mesh part thinner than 2.5 smallest cells is reported before the run as a known limit (`mesh_part_too_thin_for_the_grid`, 01b 5.9) and may meet `kCellSplitBySolid` or pass unseen." Decision 7's comment "a plate thinner than a cell" becomes "a plate or a thin mesh part". |
| Tests | solids' tests | `ComputeSolidCoverage, MeshCube_EqualsBox` and an icosphere twin of `Sphere_BlockedAreaConverges`. |

## 06 Viscosity

| Design section | Plan task | What |
|---|---|---|
| 3 `ViscosityInputs::solids`; 4 | Task 10 | Gains `contact::MeshShapes meshes` beside `solids`; calls `ComputeSolidVelocitySample(..., meshes)`. |
| Rule (a), `(u_P - v_solid) / distance`, on-surface test `1e-9 w` | Task 10 | No change (needs the true distance: 01b 5.1). |
| 5.7 "inclined" solids; `has_inclined_solid` | Tasks 6, 21 | A mesh counts as inclined whatever its pose. Until Task 21 lifts `kInclinedSolidNotSupported`, a viscous liquid with a mesh solid is excluded. Test: the tilted plate of Task 21 as a mesh slab, same 5 %. |

## 07 Materials, 09 Gas, 15 Sand

No change. They see solids through `ParticleContact`, `SolidCoverage` and `MoveParticles`. Sand 5.4 ("hit point
= start + velocity * duration"): with several hits on one mesh in a step the record holds the first, as for
several solids today.

## 08 Solids

| Design section | Plan task | What |
|---|---|---|
| 1 and 10: "solid meshes (none)" | - | Removed; points to 01b. |
| 3.2 `SolidSettings`, `SolidsSettings` | Task 3 | `SolidSettings` gains `std::uint32_t mesh_index = contact::kNoMesh`; `SolidsSettings` gains `contact::MeshShapes meshes` (on `run_on`). |
| 5.1 `BuildTry`, `BuildContactSolids` | Tasks 10, 12 | The `kMesh` case; copies `mesh_index`. |
| 5.4 `ComputeSolidsAtFaces`, `ComputeSolidCoverage` and their GPU halves | Tasks 15, 16, 18 | Mesh-aware calls; coverage passes its cap (`kCoverageCapInCells` widths) as `largest_distance`; the GPU functions receive `MeshShapes`. Cost to measure. |
| 5.4 step 5 open-to-sky flood; 5.2 travel radius `Length(half_size)` | - | No change. |
| `kSolidsTooClose` | Task 10 | A free mesh is sampled at the corners and centre of the box of its `half_size`, as the cylinder: too close is reported earlier, never later (A83). |
| 3.4 `has_inclined_solid` | Task 9 | True for any mesh. |
| 5.9 plate sides, `ComputeMinPlateThickness`, `kMaxPlateCount`, cutting | - | No change: the plate keeps its approved rules (A94). A thin mesh part has no side and is not a cutting tool. |
| Tests | Tasks 16, 35, 36a | Mesh twins of `Box_BlockedAreaConverges` and `Hydrostatic_PushIsWeightOfDisplacedLiquid`; a driven mesh paddle (thicker than 2.5 cells). |

## 10 Step

| Design section | Plan task | What |
|---|---|---|
| 3.2 `ReadyScene`; stages 11, 16; `FoamAndSprayInputs` | Tasks 5, 19, 20, 25, 26 | The step owns one `contact::MeshShapeBuffers` for the run and fills the `meshes` fields. Scenes `BuildDrivenMeshScene`, `BuildFreeMeshScene`. |
| 3.4 `CapabilityReason` | Task 3 | NEW `kMeshNotUsable` (`kUnavailable`; from `CheckMeshShape`, `module_code` = `ToName(MeshProblem)`) and NEW `kMeshPartTooThinForTheGrid` (`kExperimental`, `kMakeSolidThicker`, `solid_id`, `magnitude` = `thinnest_wall`, `limit` = `kMinWallThicknessInCells * smallest_cell_width`, both m). `kCapabilityReasonCount` grows by 2. NEW `inline constexpr double kMinWallThicknessInCells = 2.5` (01b 5.9). `kThinSolidInsideOneCell`'s comment: "a plate or a mesh part". |
| 5.14 `CheckSceneCanRun`; stages 3 and 19 | Tasks 7, 26 | Mesh-aware `CheckSolid` and `ComputeContactScale`; `CheckMeshShape` once per mesh with the run's tolerance; the thin-part comparison once per mesh solid, whatever the materials. Tests: `CheckSceneCanRun, MeshFinThinnerThanTheGridNeeds_IsAKnownLimitNamingTheSolid` (fin `1e-4` m, cells 0.01 m: magnitude `1e-4`, limit 0.025, `can_run` true); `..., MeshSlabThreeCellsThick_HasNoFinding`. |
| 5.15 `BuildStartCells` | Task 8a | Mesh-aware call (CPU arrays). The "four corners inside one solid" test gains the face centre as a fifth point (a face can span a groove of a solid that is not convex). |
| 5.9 `BuildRunPlan` | Task 8 | Adds `BuildMeshShapeMemoryEstimate`. |
| Stages, order, commit and reject; 5.8 `kSolidMotion`, `kSolidTurn` | - | No change. |

## 11 Cell resize and batches

| Design section | Plan task | What |
|---|---|---|
| Size rule; refinement volume | - | No change. |
| 5.9 `BuildBatchPlan` whole-grid rows | Task 38 | Mesh rows are whole-run rows on the processor in every batch; a mesh is never split per batch. |
| `ResizeCells` step 6 (`ComputeSolidsAtFaces` on the new grid) | - | No change; cost with a mesh to measure. |

## 12 I/O and CLI

| Design section | Plan task | What |
|---|---|---|
| Scene file `solids[]` | Task 26 | `shape` gains `mesh`; new field `mesh_file` (binary PLY, the zone format); scene schema version raised. |
| New `io/mesh_solid.h` (CPU only) | new Task 26a, after Task 26 | `ReadMeshSolid`: PLY into vertices and triangles, scale applied; `contact::BuildMeshShape`; `MeshProblem` mapped to findings with the field path; volume, centre of mass and inertia by the closed-form sums over triangles (Mirtich 1996; Eberly 2002; both checked); for a free solid the mesh re-expressed about its centre of mass and principal axes, the shift kept so a finding's point can be given in the object's coordinates. |
| 3.7 findings, `ToMessage` | Tasks 4, 36 | The message of `mesh_part_too_thin_for_the_grid` (01b 5.9), with the object's name, the point and the cell width used. |
| 5.3, 5.4 run folder, resume | Task 26a, Task 28 | `inputs/mesh_solids` (a FluidiTy array file with every built array), listed, read on resume, never rebuilt; mesh files in the fingerprint. |
| `BuildStartParticles` | Task 15 | Mesh-aware distance. |
| `BuildIoMemoryEstimate`; 5.10 budget lines | Task 23a | "io: mesh solids" (`kCpu`); the lines show mesh bytes. |
| Tests | Task 26a | Read, build, write, resume without the PLY; a cube's inertia `m / 3 * (b^2 + c^2)`; an open mesh. |

## 13 Surface, foam and spray

| Design section | Plan task | What |
|---|---|---|
| `FrameInputs::solids`, `MotionSample::solids` | Tasks 23, 45 | Carry `contact::MeshShapes` (CPU arrays from I/O). |
| 5.3 surface field, 5.5 foam, 5.6 foam volume | Tasks 18, 22, 35, 37 | Mesh-aware distance with `largest_distance` = 4 spacings (the field's own cap). Tests `BuildSurfaceField, BesideAMeshSolid_SameHeight`, `MoveFoamParticles, StopsAtAMeshSolid`. |

## 14 Blender add-on

| Design section | Plan task | What |
|---|---|---|
| 3.2 `Shape`; 5.1 item 2; 10 | Tasks 5, 8, 8a, 27 | `Shape.MESH` for the `SOLID` role, the default for an object that is not exactly a box, sphere or cylinder (A88); the render-level evaluated mesh (A86), quads and n-gons by the fan rule, through `zone_mesh.py`'s writer, to `inputs/solids/solid_<id>.ply`, object scale applied. Section 10 drops "evaluated mesh shapes". |
| 3.4 findings | Tasks 11, 13 | Quick check: `solid_mesh_not_closed` (`BLOCKS_RUN`, with the open edge count, A87), `solid_mesh_deforms` (`BLOCKS_RUN`, A84), `solid_mesh_triangles_small` (`NOTE`, under an eighth of a cell, A86), `solid_mesh_too_many_triangles` (`BLOCKS_RUN`). Native: `mesh_part_too_thin_for_the_grid` (`HAS_KNOWN_LIMIT`; repair "Select <object>"), and the `MeshProblem` codes (`BLOCKS_RUN`). |
| Digest | Task 8 | Covers the solid mesh files. |
| Tests | Tasks 8, 13, 27 | `test_mesh_solid_exports_evaluated_triangles`; closed or not; deforms. |

## Shared files

`shared-interface.md` (b) and (c) Contact: the names of 01b 3.1 to 3.4. `tradeoff-particles.md` 1.3: one line,
mesh solids, 64 to 74 B per triangle on the processor (one million triangles: 0.07 of 13.05 GB; no verdict of
1.6 changes).
