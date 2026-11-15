# FluidiTy glossary

This file is the source of truth for the words of the code base. A domain word may appear in an identifier or
a comment only if it is listed here (`AGENTS.md`, naming rule 2). The rules for using these words are in
`AGENTS.md`.

How to read the tables:

- **Say** is the word or name to use. `...` stands for the rest of a name.
- **Plain meaning** explains it for a good C++ programmer who has never read a fluid simulation paper. Units
  come first where the word is a physical quantity.
- **Never say** lists the words that are banned for that concept, separated by commas. A word in parentheses
  limits the ban to one sense: `component (for this)` bans `component` only for this concept, and `row (alone)`
  bans `row` unless it is part of the allowed phrase. A word without parentheses is banned everywhere. The rules
  check prints a note, not a failure, for a word banned in one sense.
- A tool reads this file. Every vocabulary table keeps exactly the header `| Say | Plain meaning | Never say |`.

A new word is added here in the same commit that first uses it. Words that are still being settled are not
listed; they arrive with the module that settles them (`docs/design/programme.md`, section 9).

## Grid and geometry

| Say | Plain meaning | Never say |
| --- | --- | --- |
| `StaggeredGrid`, `grid` | Pressure at cell centers, each velocity part on the faces it crosses. The header comment gives "also called MAC grid, Harlow and Welch 1965" once. | MAC, mac_grid |
| `NearbyFaceWeights`, `FaceWeight` | The grid faces near one particle, each with the weight the particle gives it. | stencil, support (for the grid faces near a particle), node (for a grid point near a particle), sample (for this), slot, row (alone) |
| `weight_curve` | The function that turns a distance into a weight. | basis, kernel (for this), quadratic kernel |
| `index` | Position in any array or sequence. | slot, ordinal |
| `place`, `CellPlaceIndex`, `FacePlaceIndex`, `CornerPlaceIndex` | A position in the arrays of one grid version. It is valid for that version only: after the grid changes, the same cell may sit at another place. Use `id` for an identity that must survive the change. | slot |
| `id` | Stable identity that survives reordering. | |
| `cell_coord`, `CellCoord` | Integer (x, y, z) location of a cell. | CellIndex3D |
| `axis` | The x, y or z part of a vector or grid. | component (for this) |
| `cell_low`, `cell_high` | The two cells on the low and high side of a face, along that face's axis. | tail (for a cell on one side of a face), head (for a cell on one side of a face) |
| `side_left` (-X), `side_right` (+X), `side_front` (-Y), `side_back` (+Y), `side_bottom` (-Z), `side_top` (+Z) | Blender viewport names. Z is up. | low_x wall, floor (for a side: math::Floor keeps the word), upper (for a side of the domain), lower (for a side of the domain) |
| `into_...`, `out_of_...` | Direction of a flow or normal, always naming the thing: `flow_into_liquid`, `normal_out_of_solid`. | inward (without the thing), outward (without the thing) |
| `touching` | Shares a boundary with. | incident |
| `partly_filled_cell` | A cell crossed by a solid or by the liquid surface. | cut cell |
| `open_area` | m2. Area of a face not blocked by solid. | |
| `open_fraction` | No unit, 0 to 1. The open area as a fraction of the face area. | aperture |
| `liquid_covered_area` | m2. Area of a solid face that liquid touches. | wetted |
| `cell_eighth` | One of the 8 sub-boxes of a cell. | octant |
| `distance_to_surface` | m. Distance to the liquid surface, negative inside the liquid. | signed distance |
| `sample_weight` | Weight of one sample point in a sum that approximates an integral. | quadrature weight |

## Solids and the domain

| Say | Plain meaning | Never say |
| --- | --- | --- |
| `solid`, `Solid`, `SolidShape` | Anything the liquid cannot enter. `Solid` is what the contact code reads: a shape, its pose at the start of the step and its motion during the step. | wall (for a whole solid), obstacle, body (for a solid) |
| `DomainBox`, `domain box`, `domain_side_...`, `DomainSide`, `ClosedSides` | The box the liquid lives in, open at the top by default. It is fixed and aligned with the world axes. Its faces are domain sides; a closed side stops particles. | tank, wall (for a domain side) |
| `FlatSolid`, `flat solid` | Fills one side of a plane. | halfspace |
| `Plate`, `plate` | Thin solid with two faces, unbounded sideways. | slab, blade |
| `Box`, `box` | Solid with six flat faces. | |
| `Sphere`, `sphere` | Round solid: every point within `radius` of its centre. | |
| `Cylinder`, `cylinder` | Finite solid cylinder: a round `wall` around its axis and two flat `cap`s. | |
| `wall` (of a cylinder) | The round surface of a cylinder, around its axis. Face `kCurved`. Only for a cylinder: a whole solid or a domain side is never a wall. | |
| `cap` | One of the two flat ends of a cylinder. Faces `kLowZ` and `kHighZ`. | |
| `rim` | The circle where a cylinder's wall meets a cap. Faces `kLowRim` and `kHighRim`. | |
| `spin` | A turn about a sphere's centre or a cylinder's axis. It moves the surface but not the shape, so it does not count toward the turn limit of a step. | |
| `plate side`, `PlateSide`, `SolidSide` | One of the two sides of one plate, as a stable identity: the plate's id and the side. Particles on different plate sides never exchange values. | side_tag |
| `solid face`, `SolidFace` | One part of a solid's surface, named in the solid's own axes: `kLowX` to `kHighZ` for flat faces, `kCurved`, `kLowRim`, `kHighRim` for round ones. | |
| `pose`, `SolidPose` | Where a solid is: the origin of its own axes, m, and the rotation whose columns are those axes in the world. | |
| `motion`, `SolidMotion` | How a solid moves during one step: the linear velocity of its origin, m/s, and its angular velocity about the origin, rad/s. Constant during the step. | |
| `fixed`, `driven`, `free` | Motion: never moves; follows a given path; pushed by the liquid. Combine with a shape: `DrivenFlatSolid`, `DrivenPlate`, `FreeBox`. | prescribed, rigid body, floating (for motion) |
| `wave_paddle` (a variable, not a type) | The driven flat solid that makes waves. It is its own solid, not part of the domain box. | piston |
| `solid_velocity`, `SolidVelocitySample` | m/s. Velocity of a solid's surface. | wall_velocity, LiquidPrescribedWallSample3D |
| `same_side` | Both are on the same side of a plate, so they may exchange data. | compatible (for two things on the same side of a plate) |
| `elapsed` | Clock time: `work_elapsed_seconds`. In contact and solids, seconds since the start of the step: `ComputeSolidPose(solid, elapsed)`, `start_elapsed`. | wall (for time) |
| `close call`, `ContactCloseCall` | A step that contact cannot answer safely. The step is shortened and tried again. | near miss, edge case (for this) |

## Particles and the particle-grid exchange

| Say | Plain meaning | Never say |
| --- | --- | --- |
| `ParticlesToGrid` | Each particle adds its mass and momentum to the faces near it. | P2G, deposit (as the stage name) |
| `GridToParticles` | Each particle reads the grid velocity at its position. | G2P, gather (as the stage name) |
| `grid_velocity_at_particle` | m/s. Grid velocity interpolated at the particle. | PIC velocity |
| `PerParticleVelocity<T> { for_next_step; for_this_step; }` | One value for each of a particle's two velocities. `for_next_step` is kept on the particle for the next step. `for_this_step` moves the particle in this step. | stored (for a velocity kept on the particle), advecting, state (as a velocity field), travel (as a velocity field) |
| `own_motion` | m/s. Particle velocity minus grid velocity before the solve. | history residual, particle_minus_old_pic_velocity |
| `own_motion_kept` | No unit, 0 to 1. 0 means follow the grid fully. | energy_control, alpha |
| `own_motion_in_travel` | No unit, 0 to 1. Share of own motion also used to move the particle in this step. | separation_factor, beta |
| `MixParticleVelocity` | Combines grid velocity and own motion. | BlendHistory |
| `used_grid_only` | The particle followed the grid velocity only. | old_pic_fallback, new_pic_fallback |
| `fitted_velocity_change` | 1/s. Best fit of how grid velocity varies around the particle. Comment: "papers: APIC matrix C". | velocity slopes, affine |
| `exact_velocity_change` | 1/s. Derivative of the interpolated velocity. Comment: "papers: velocity gradient L". | velocity gradient |
| `carries_fitted_change` | Yes or no: the particle carries a fitted velocity change. | affine_enabled |
| `AddParticles`, `SplitParticle`, `MergeParticles`, `RemoveParticles`, `AdjustParticleCount` | Change the number of particles. The reason is an enum: `AddReason { kFromSource, kRefillRegion }`. | emit, birth, reseed, retire, resampling |
| `particle_group`, `GroupParticlesByBlock` | Particles that share one block of cells. | sort bin, bin, sort (for grouping) |
| `position_when_grouped` | m. Particle position at the moment the particles were grouped. | sorted_world_position |
| `cell_block` | A fixed chunk of cells. | |
| `write_area` | The cells a particle may write to. | tile |
| `border_cells` | One extra layer of cells around a region. | halo |

## Pressure solve

| Say | Plain meaning | Never say |
| --- | --- | --- |
| `SolvePressure` | Find the pressure that makes inflow equal outflow in every cell. Used in scenes with no viscous ingredient. | project (for the solve), projection (for the solve) |
| `SolveViscosityAndPressure` | Find velocity and pressure together, so that thick liquid resists shearing and keeps its volume in one solve. Used in scenes with a viscous ingredient, instead of `SolvePressure`. | ApplyViscosity |
| `flow_imbalance` | Inflow minus outflow of one cell. | divergence, defect |
| `cell_equation` | One cell's balance equation. | pressure row, row (alone) |
| `matrix row` | One horizontal line of a matrix. The only allowed use of the word row. | row (alone) |
| `cell_connection` | The term that couples two neighbouring cells. | link (for the term that couples two cells) |
| `added_volume_rate` | m3/s. Volume added to a cell per second. | source (for this) |
| `PressureEquations` | All cell equations of one solve. | PreparedMacPressureSystem3D, system (for a set of linear equations) |
| `squeeze_allowance` | How much a cell may be squeezed. | compliance |
| `pressure_offset` | The free constant in the pressure. | gauge |

## Mixed-size grid

Every grid is a mixed-size grid from its first commit; these words name its parts.

| Say | Plain meaning | Never say |
| --- | --- | --- |
| `mixed_size_grid` | A grid whose cells have different sizes. | graded, adaptive |
| `big_cell`, `small_cells` | A cell and the 8 cells it can be split into. | parent (for a big cell), children (for small cells), coarse (for the size of a big cell), fine (for size) |
| `sibling_cells` | The 8 small cells of one big cell. | family |
| `active_cell` | A cell that is not split. | leaf |
| `size_level` | 0 is the biggest. | level (alone) |
| `face_piece` | The part of a big face that touches one small cell. | subface |
| `size_step_rule` | Neighbours differ by at most one size level. | 2:1 closure |
| `SplitCell`, `MergeCells`, `ResizeCells` | Split one big cell into its 8 small cells; merge 8 sibling cells into one big cell; change cell sizes after a committed step. | refine, coarsen, adapt |
| `particle_region` | Liquid near the surface, carried by particles. | band, narrow band |
| `grid_only_region` | Liquid with no particles. | core (for liquid with no particles) |
| `region_transfer` | Moving liquid between the two kinds of region. | handoff, owner exchange |

## Step, time and outcome

| Say | Plain meaning | Never say |
| --- | --- | --- |
| `frame` | One output time (a Blender frame). | |
| `step` | One advance of the simulation by `step_duration`. | substep |
| `try` (in names: `TryStep`, `try_count`) | One run of a step. A rejected try is retried with a shorter duration. `try` alone is a C++ keyword; never use it as a bare identifier. | attempt, trial |
| `state_version` | Counter that grows at each commit. | epoch |
| `check` | Any test of whether a value or a state is allowed. | certify, qualify, admit, validate |
| `reject`, `RejectReason { kBeforeTry, kAfterTry, ... }` | Any negative outcome of a try. The reason says when and why. | refuse, decline |
| `<Module>Failure`, `<Module>Error` | What a function returns when it cannot do its job: the module's error enum value plus the context that explains it (which particle, which face, what size). A step try that does not pass is rejected, not failed. | |
| `commit` | The try passed and replaces the state. | accept (for a passing try replacing the state), publish, install (for a passing try replacing the state) |
| `snapshot` | Read-only copy of state at one time. | frozen, freeze |

### Step stages, in order

A step runs these stages in this order. Each name is an allowed verb (`AGENTS.md`, "Allowed verbs"). The step
module may add stages; each new stage name enters this list in the commit that adds it.

1. `AddNewParticles`
2. `ChooseActiveCells`
3. `ParticlesToGrid`
4. `SaveGridBeforeForces`
5. `ApplyForces`
6. `DriveWaves`
7. `CollideSolids`
8. `FindLiquidSurface`
9. `SolveViscosityAndPressure`: scenes with a viscous ingredient run this stage instead of stage 10.
10. `SolvePressure`: scenes with no viscous ingredient.
11. `SaveGridAfterPressure`
12. `GridToParticles`
13. `MoveParticles`
14. `MakeFoamAndSpray`
15. `CheckStep`
16. `CommitStep`
17. `ResizeCells`

## Memory, files and plumbing

| Say | Plain meaning | Never say |
| --- | --- | --- |
| `buffers` | Preallocated, reused every step. Each one is a `Buffer` (below) built once and kept. | workspace, arena, staging, bank, storage (for the preallocated reused buffers) |
| `saved_state` | On disk; lets a run continue. Written in FluidiTy's own array file. | checkpoint, store (for saving a run to disk) |
| `playback_cache` | Output files for playback: particles, surface, foam and spray, volumes and solid motion. Written in open formats (Alembic, USD, OpenVDB) so they read with the add-on disabled. | point cache, frame cache |
| `scene_file` | What the artist set up, as exported from Blender. | |
| `ready_scene` | Scene with defaults filled in and checked. | resolved scene |
| `file_list` | Which files belong to a run. | manifest |
| `BufferLoan` | A borrowed GPU buffer. | lease |
| `Buffer`, `BuildBuffer`, `WriteBuffer`, `ReadBuffer`, `CopyBuffer` | An array of fixed size on the processor the caller chose (`base/buffer.h`). `WriteBuffer` and `ReadBuffer` copy between CPU memory and a buffer, the only way values enter or leave the GPU; `CopyBuffer` copies between two buffers on one processor. The plural `buffers` (above) names the ones kept and reused every step. | |
| `RunMemory`, `BuildRunMemory` | One block of memory reserved on one processor for a run (`base/run_memory.h`), sized by the memory estimate. Buffers are built from it as pieces, left to right. It frees all its memory at once when destroyed, so it must outlive its pieces. A run has one per processor it uses. | pool, allocator, suballocation |
| `piece`, `kRunMemoryPieceAlignmentBytes`, `ComputeRunMemoryPieceBytes` | The bytes of one buffer inside a `RunMemory`'s block. A piece starts at a multiple of 256 bytes from the start of the block and takes its byte count rounded up to a multiple of 256. It is never freed alone. | chunk (for a piece) |
| `reserved bytes`, `reserved_bytes()` | Bytes. The size of a `RunMemory`'s block, asked once from the processor. | |
| `handed out`, `handed_out_bytes()`, `ComputeRunMemoryBytes`, `kRunMemoryTooSmall` | Bytes of a block already given to pieces, their rounding included. Bytes handed out are never given back, even when the piece's buffer is destroyed. `ComputeRunMemoryBytes` gives what building every buffer of an estimate hands out, so a block of that size is used to the byte. `kRunMemoryTooSmall` says a piece did not fit: usually the estimate missed an array; a limit raised during the run can also cause it. | |
| `slice`, `first`, `count` | `count` items of an array starting at index `first`. | |
| `math::TreeSum`, `ComputeTreeSum`, `ComputeTreeDot`, `ComputeLargestMagnitude` | A sum of many doubles in one fixed order (`base/tree_sum.h`, `base/ordered_sums.h`): the values are padded with zeros to the next power of two and added in a binary tree, so the result has the same bits on every run and on both processors. `ComputeLargestMagnitude` returns the largest absolute value, or NaN if any value is NaN. | |
| `SumBuffers`, `BuildSumBuffers`, `ComputeSumBuffersSize`, `SumBuffersSize` | The working buffers of the sums on one processor (`base/ordered_sums.h`): on the GPU the products of a dot product and the layer arrays of the tree, on the CPU nothing. Built once for up to `max_count` values and passed to every sum, so that no call allocates; their contents mean nothing between calls. `SumBuffersSize` is their exact size in bytes, for memory plans. | |
| `pair sum`, `layer`, `padding` | The parts of a tree sum. A pair sum is its left side plus its right side, where a side is a value or a pair sum of the layer below. A layer is the pair sums at one height of the tree. Padding is the zeros that fill the values up to a power of two; it is never kept in memory. | |
| `BuildStableOrder` | The order of items by an unsigned integer key, with ties kept in the order they came in (`base/stable_order.h`). Only integers take part, so the CPU and the GPU give the same bytes. | |
| `OrderBuffers`, `BuildOrderBuffers`, `ComputeOrderBuffersSize`, `OrderBuffersSize` | The working buffers of the stable order, the prefix sum and the grouping on one processor (`base/stable_order.h`): the second sequence of the passes and the digit counts on the CPU; the second sequence, the table of chunk counts and the chunk totals of the prefix sums on the GPU. Built once for up to `max_count` items and passed to every call, so that no call allocates; their contents mean nothing between calls. `OrderBuffersSize` is their exact size in bytes, for memory plans. | |
| `stable`, `key`, `digit`, `counting pass`, `chunk` | The parts of a stable order. Stable: items with equal keys keep the order they had before. A key is the unsigned integer an item is ordered by; a digit is one byte of it. A counting pass places every item by one digit: count each digit value, turn the counts into first places, put each item at the next place of its digit. A chunk is 1,024 consecutive places of a sequence, handled by one thread block on the GPU. Papers: least-significant-digit radix sort. | |
| `ComputeExclusivePrefixSum`, exclusive prefix sum | `sums[i]` is the sum of the values before index `i`; `sums[count]` is the total. Exact integer arithmetic. | |
| `BuildGroupsByDestination`, `destination`, `first_of_group` | Puts items into groups by the group each one names (its destination): the items of group 0, then group 1, ..., each group in ascending item index, and where each group starts. | |
| `CpuMemory::kLockedForCopies`, locked for copies | CPU memory that is never moved or swapped out, so the GPU copies it directly. Papers and the CUDA manual: page-locked or pinned memory. | pinned (as a name) |
| `OnStepDone` and similar | A callback is named after its event or question (`ChooseCellSize`). | Observer, Hook, Provider, Selector |
| no suffix | A type is named after its content. Whether it owns memory is said in its comment and by `std::span` fields. | View (as a suffix), Record (as a suffix), Layout (as a suffix), payload |
| `max_...` | A hard limit: the code refuses to go past it. | capacity, reserve (for a limit), budget (for a hard limit) |
| `budget` | A planned cost shown to the user, such as a memory budget or a time budget. The code reports it and does not enforce it. | |
| `MemoryEstimateEntry`, memory estimate | One line of the memory estimate (`base/memory_count.h`): a named array a module will allocate, what its item count scales with, the bytes of one item, how many copies exist, how long it is kept and where. The memory estimate is the list of these entries. | |
| `tolerance` | An allowed error. | margin, epsilon (for an allowed error), resolution (for an allowed error) |
| `blob`, `blob_id`, `BlobHistory` | One connected body of liquid, or one air pocket. | component (for this), lineage |
| `ingredient` | One substance in a mixture. | component (for this) |
| `thread block`, `kThreadBlockSize`, `kMaxThreadBlockCount`, `LaunchSize` | The threads of one GPU launch unit, which can share fast memory and wait for each other (`base/run_on_call.h`). `kThreadBlockSize` is the threads in each thread block of a simple GPU function, `kMaxThreadBlockCount` the most thread blocks it starts. Never `block` or `group` alone in that sense. CUDA: thread block. | |

## Materials

The code uses the plain name. The comment on the field gives the textbook word once, so papers stay searchable.

| Say | Plain meaning | Never say |
| --- | --- | --- |
| `MaterialKind::kLiquid`, `kGas`, `kSnow`, `kSpringySolid` | The kinds of material. A springy solid deforms under force and returns to its shape. Textbook: elastic solid. | |
| `material set` | Materials that share one grid velocity. Materials that do not mix are in separate sets; every liquid is in one set, and gas markers are in one set. | |
| `stiffness_against_shear` | Pa. Textbook: shear modulus, mu. | shear_modulus (as a name) |
| `stiffness_against_squeeze` | Pa. Textbook: Lame lambda. | lame_lambda (as a name) |
| `springy_stretch` | No unit. How far the material is stretched along each of its three main directions, in the part that springs back. Textbook: elastic stretches. | elastic_stretches (as a name), elastic (as a name) |
| `springy_deformation` | No unit, a 3 by 3 matrix. The part of the deformation that springs back. Textbook: elastic deformation gradient. | elastic_deformation (as a name), deformation_gradient (as a name) |
| `permanent_volume_change` | Natural log of volume, no unit. Volume change that does not spring back. Textbook: log plastic Jacobian. | log_plastic_jacobian (as a name), plastic (as a name) |
| `internal_force_per_area` | Pa. The force the material exerts on itself, per area. Textbook: Kirchhoff stress. | kirchhoff_stress (as a name), stress (as a name) |
| `deformation_rate`, `stretch`, `shear` | 1/s. How fast the material deforms. `stretch` is the rate along one axis; `shear` is the rate between two axes. Textbook: strain rate. | strain (as a name) |
| `rest_volume` | m3. Volume of a particle's material at rest. | reference_volume (as a name) |
| `volume_ratio` | No unit. Current volume divided by rest volume; 1 at rest. | |
| `friction_angle` | rad, from 0 to less than pi/2. The steepest slope grains hold among themselves. Textbook: Drucker-Prager friction angle. | |
| `cohesion` | Natural log of volume, no unit, zero or more. How far grains can be pulled apart, as growth in volume, before they separate. Textbook: cohesion. Used by the sand module. | |
| `MaterialResponse` | How a material answers deformation with internal force. Textbook: constitutive model. | constitutive (as a name) |
| `PressureLaw` | How pressure follows from density. Textbook: equation of state. | EOS |

## Never say, no replacement

These words name machinery that does not exist on this branch. They have no replacement.

| Say | Plain meaning | Never say |
| --- | --- | --- |
| (nothing) | Fixed-point sums are not used. Sums are plain doubles in a fixed order. | quantum, quanta, quantized |
