// FluidiTy: the method. Build from the repository root:
//   typst compile docs/method-paper/fluidity-method.typ out/paper/fluidity-method.pdf
//
// Typefaces. The owner's licensed MB Type library sets this paper; the fonts are never committed.
//   Text:     Equity OT A, a book face in the Ehrhardt tradition: compact, sturdy and even on the page.
//             Grade A suits PDF.
//   Headings, captions, tables, figure labels and procedures: Hermes Maia OT 3, a sans with flared strokes in
//             the Optima line, set at 0.88 of the old sans sizes because it runs wider.
//   Code:     Triplicate OT A with its code alternates (ss02).
//   Math:     New Computer Modern Math, bundled with Typst.
// Fallbacks, used by anyone without the licensed fonts: Libertinus Serif (bundled with Typst, a book face of
// similar colour and width), Libertinus Sans (free to download, a humanist sans close to the flared Hermes Maia)
// and DejaVu Sans Mono (bundled with Typst). Typst then warns about each missing MB name and the paper still
// compiles.
//
// Page: A4, 11 pt text on a 106 mm measure, about 70 characters per line.

#import "pseudocode.typ": procedure
#import "figures/axes.typ": axes-figure
#import "figures/buffer_crossings.typ": buffer-crossings-figure
#import "figures/buffer_layers.typ": buffer-layers-figure
#import "figures/one_operation_two_processors.typ": run-on-figure
#import "figures/run_memory.typ": run-memory-figure
#import "figures/tree_sum.typ": tree-sum-figure
#import "figures/stable_order.typ": stable-order-figure
#import "figures/contact_solid_shapes.typ": solid-shapes-figure
#import "figures/contact_tolerance.typ": contact-tolerance-figure
#import "figures/contact_sweep_hit_slide.typ": sweep-hit-slide-figure
#import "figures/contact_response.typ": contact-response-figure
#import "figures/contact_round_solids.typ": round-first-hit-figure, round-solids-figure
#import "figures/contact_move_off_near_surfaces.typ": move-off-near-surfaces-figure
#import "figures/grid_staggered.typ": staggered-grid-figure
#import "figures/grid_face_pieces.typ": face-pieces-figure
#import "figures/grid_nearby_faces.typ": nearby-faces-figure
#import "figures/grid_parallel_build.typ": parallel-build-figure

#let serif = ("Equity OT A", "Libertinus Serif")
#let sans = ("Hermes Maia OT 3", "Libertinus Sans")
#let mono = ("Triplicate OT A", "DejaVu Sans Mono")

#set document(title: [FluidiTy: the method])
#set page(
  paper: "a4",
  margin: (x: 52mm, top: 26mm, bottom: 34mm),
  numbering: "1",
  number-align: center + bottom,
)
#set text(font: serif, size: 11pt, lang: "en", costs: (widow: 100%, orphan: 100%))
#set par(justify: true, leading: 0.62em, spacing: 0.62em, first-line-indent: (amount: 1.2em, all: false))
#set block(spacing: 1.2em)

#set heading(numbering: "1.1")
#show heading: set text(font: sans, weight: "bold", number-type: "lining", hyphenate: false)
#show heading: set par(justify: false)
#show heading.where(level: 1): set text(size: 1.2em)
#show heading.where(level: 1): set block(above: 2em, below: 0.9em)
#show heading.where(level: 2): set text(size: 1em)
#show heading.where(level: 2): set block(above: 1.5em, below: 0.7em)

#set raw(theme: none)
#show raw: set text(font: mono, features: ("ss02",), size: 1em / 0.8 * 0.85, ligatures: false)
#show raw.where(block: true): set par(justify: false)
#show raw.where(block: true): set block(inset: (left: 1.2em))
#show terms: set par(justify: false)

#show table: set text(font: sans, size: 7.92pt, number-type: "lining", number-width: "tabular")
#set table(stroke: none, inset: (x: 0.6em, y: 0.4em))

#show figure.caption: set text(font: sans, size: 7.92pt, number-type: "lining")
#show figure.caption: it => block(width: 100%, align(left)[
  *#it.supplement #context it.counter.display(it.numbering)#it.separator*#it.body
])
#show figure.where(kind: "procedure"): set figure.caption(position: top)
#show figure.where(kind: table): set figure.caption(position: top)
#set figure(gap: 0.9em)

// Only an equation with a label gets a number, so a `Read:` line in the code can name it and the reader can find
// it. An unlabelled display equation stays unnumbered and does not advance the count.
#set math.equation(numbering: "(1)")
#show math.equation: it => {
  if it.block and not it.has("label") and it.numbering != none {
    counter(math.equation).update(count => count - 1)
    math.equation(it.body, block: true, numbering: none)
  } else {
    it
  }
}

// Title block.
#block(below: 1.6em)[
  #set par(justify: false, first-line-indent: 0pt)
  #text(font: sans, size: 19.4pt, weight: "bold")[FluidiTy: the method]
  #v(0.3em)
  #text(font: sans, size: 10.6pt)[How the simulation works, written for programmers]
]

This paper describes the method that the FluidiTy code implements. FluidiTy simulates liquids, gas, snow and
springy solids for Blender. The paper is written for a good C++ programmer who has never read a fluid simulation
paper, and it grows section by section: each module of the code adds its section in the same commit as its code,
and each code file names the section it follows on its `Read:` line. A section shows its idea in a drawn figure
and its main procedure as numbered steps, and the steps match the `// Step N:` comments in the code. Today the paper
holds what every module shares: the conventions, the notation, how one operation runs on either processor, how a
sum of many doubles is added in one fixed order, and how items are put in an order that never depends on timing.
It also holds the first module, contact with solids: how a particle moves during a step when solids are in its
way, and the second, the grid: where the simulation keeps its values, and which faces a particle reads and writes.

= Conventions <sec-conventions>

== Units

Every physical quantity is in SI units: metres, seconds, kilograms, and the units built from them, such as m/s for
a speed, Pa for a pressure and kg/m³ for a density. Angles are in radians. A name in the code says the kind of
quantity in words (`step_duration`, `open_area`), and the first thing its comment states is the unit.

== Axes and gravity

Positions use right-handed axes with Z up, as Blender does. Right-handed means: point the fingers of the right hand
along X and curl them toward Y; the thumb then points along Z. Gravity points along (0, 0, −1). The six sides of a
box carry Blender's viewport names: left (−X), right (+X), front (−Y), back (+Y), bottom (−Z) and top (+Z).

The proof of concept that came before this code, like many papers, describes scenes with Y up. A quarter turn
about X carries such a description to ours (@fig-axes). In matrix form, with matrix rows (1, 0, 0), (0, 0, −1) and
(0, 1, 0),

$ mat(x_"Z up"; y_"Z up"; z_"Z up") = mat(1, 0, 0; 0, 0, -1; 0, 1, 0) mat(x_"Y up"; y_"Y up"; z_"Y up"). $

The turn keeps X. It takes the old vertical, Y, to Z, and it takes the old Z, which pointed at the reader, to −Y.
Seen as a motion of the axes, it lifts the Z arrow from the reader until it points up, while the world and gravity
stay where they are.
Gravity (0, −1, 0) becomes (0, 0, −1). The matrix has determinant +1, so it keeps every length, every angle and the
order in which a triangle's corners go round. When an old result is compared with a new one, the old output is
turned with this matrix first.

Swapping Y and Z looks just as simple, but its determinant is −1. A swap is a mirror image, not a turn (papers call
it a reflection). It reverses the order of every triangle's corners, so a normal computed from a triangle's corners
to point out of a solid would point into it.

#figure(
  axes-figure(),
  placement: auto,
  caption: [A quarter turn about X carries the Y-up axes to the Z-up axes and leaves gravity pointing down the
    page. Unlike a swap of Y and Z, the turn keeps the order of a triangle's corners.],
) <fig-axes>

== Time labels

A step advances the simulation by Δt, the `step_duration`. Inside one step the grid velocity changes many times:
forces, waves, solids and the pressure solve all change it. Three moments are saved and named, and a saved copy of
a grid quantity ends its name with the moment it was saved:

/ `before_forces`: after the particles have written their mass and momentum to the grid, before gravity and
  the other forces act (the `SaveGridBeforeForces` stage).
/ `before_pressure`: after the forces and the solid collisions, just before the pressure solve.
/ `after_pressure`: after the pressure solve (the `SaveGridAfterPressure` stage). This is the velocity the
  particles read back.

Papers write these three moments as superscripts; @sec-notation lists them.

== Two velocities per particle <sec-particle-velocities>

A particle carries two velocities, held together in one value of the type `PerParticleVelocity<T>` from
`base/per_particle_velocity.h`. Its field `for_next_step` is kept on the particle for the next step, and its field
`for_this_step` moves the particle during this step. They differ, for example, when a particle meets a solid that
does not react: it slides along the surface for the rest of the step but keeps the velocity it had before the hit for
the next step.

== What "same GPU, same input, identical output" promises

The promise: the same build, run twice on the same GPU with the same input, writes the same bytes in every output
(papers call this determinism). The CPU path makes the same promise on one machine with one build. Three rules keep
it. Every sum is made of plain doubles added in a fixed order, never with floating-point atomics, whose order depends
on which thread arrives first. Reductions follow a fixed tree (@sec-fixed-order-sums). When several items fail,
the one reported is the first by stable id. Repeat-run tests hold the code to this promise: they demand identical
bytes.

What it does not promise: the CPU and the GPU do not give the same bits for a whole operation. They agree within a
tolerance that each module states, as an absolute floor plus a term relative to the size of the value. Nor are
results bit-identical across GPU models, compilers or machines. The reason is that floating-point addition depends
on order. In doubles, (0.1 + 0.2) + 0.3 gives 0.6000000000000001, while 0.1 + (0.2 + 0.3) gives 0.6. The CPU loop
and the GPU function add the same numbers in different orders, so their last bits can differ. The sums of
@sec-fixed-order-sums are the exception: they add the same pairs in the same order on both processors, so they give
the same bits, unless the result is NaN, whose bits IEEE 754 leaves open.

A single math function has no sum whose order could differ, so it is a narrower case than a whole operation, and
the math functions that the shared formulas call come in two groups. Floor, absolute value, minimum, maximum and
square root give the same bits on both processors, and so does a formula built from them with addition,
subtraction, multiplication and division, such as the length of a vector. IEEE 754 fixes their results exactly:
the square root is correctly rounded, and both processors follow that rule for doubles. A NaN result, such as the
square root of −1, is the exception, since IEEE 754 leaves the bits of a NaN open. Cube root, exponential,
logarithm, power, sine and cosine call each processor's own library, which IEEE 754 does not require to round
correctly, so their last bits can differ, and so can a rotation built from them. A test that compares the
processors demands equal bits for a formula of the first group and a tolerance for any formula that uses the
second.

= Notation <sec-notation>

The code uses words, never paper symbols. @tab-notation pairs each symbol that stands for a code value with its
code word; a module that brings such a symbol adds it here. A symbol used only inside one derivation, such as the
coefficients of a quadratic, has no code word and is defined where it is used.

#figure(
  align(left, table(
    columns: (auto, auto),
    align: (center, left),
    table.hline(stroke: 0.6pt),
    table.header([*Paper symbol*], [*Code word*]),
    table.hline(stroke: 0.4pt),
    [$x$, $y$, $z$], [`x`, `y`, `z`: coordinates along X, Y, Z, in metres],
    [$Delta t$], [`step_duration`, in seconds],
    [$q^n$], [`..._before_forces`, where $q$ is any grid quantity],
    [$q^*$], [`..._before_pressure`],
    [$q^(n+1)$], [`..._after_pressure`],
    table.hline(stroke: 0.4pt),
    [$T$], [`step_duration` in @sec-contact (the same value as $Delta t$), in seconds],
    [$"tol"$], [`contact_tolerance`, in metres],
    [$d$], [`distance` to a solid, in metres: positive outside, negative inside],
    [$g$], [gap, $d - "tol"$, in metres],
    [$mu$], [`friction`, no unit],
    [$bold(n)$], [`normal_out_of_solid`, a unit vector],
    [$bold(u)$], [`solid_velocity`, the velocity of a solid's surface, in m/s],
    [$a$], [`approach_speed`, in m/s: negative means approaching],
    [$bold(p)$], [a point in a solid's own axes, in metres],
    [$bold(o)$, $A$], [a solid's `origin` (metres, world) and `axes` (a rotation whose columns are its own axes)],
    [$bold(v)_"lin"$, $bold(omega)$], [a solid's `linear_velocity` (m/s) and `angular_velocity` (rad/s)],
    [$L$, $w_"min"$], [`largest_coordinate` and `smallest_cell_width`, in metres],
    [$t_0$], [`start_elapsed`, the part of the step already used, in seconds],
    table.hline(stroke: 0.4pt),
    [$l$], [`size_level` of a cell, 0 for the biggest cells (@sec-grid)],
    [$h_0$, $h_l$], [`biggest_cell_width` and the width of a cell of size level $l$, in metres],
    [$h$], [`cell_width`: the width of the active cell that holds a position, in metres],
    [$Q$, $Q'$], [the `weight_curve` and its slope (`WeightCurveValue`), no unit],
    [$bold(x)_p$, $bold(x)_f$], [a particle's `position` and a face piece's centre, in metres, world],
    [$bold(d)$], [a face's offset from the particle in cell widths, $(bold(x)_f - bold(x)_p) \/ h$; not the contact
      $d$],
    [$n_p$, $n_f$], [on one axis, the position's quarter of its cell (0 to 3) and a face centre's place, both in quarter
      widths from the cell's low corner, integers],
    [$"area"_f$], [the area of a face piece, in m²],
    [$w_f$, $bold(g)_f$], [the raw weight of a face (no unit) and its change with the position (1/m)],
    [$W$, $bold(G)$], [`weight_sum` (no unit) and `raw_weight_change_sum` (1/m)],
    [$hat(w)_f$], [`weight` of a `FaceWeight`, no unit; its change with the position is `weight_change`, 1/m],
    [$bold(m)$, $S$], [`mean_offset` (m) and `weighted_spread` (m²)],
    [$n$], [`active_count`: the number of active cells in the list a grid is built from, in @sec-parallel-build
      and @proc-count-grid],
    table.hline(stroke: 0.6pt),
  )),
  caption: [Paper symbols and their code words.],
) <tab-notation>

= One operation, two processors <sec-run-on>

Every operation can run on the CPU or on the GPU. In the operation `ParticlesToGrid`, for example, each particle
adds its mass and momentum to the grid faces near it. Callers see one public function for it, and two private files
implement it (@fig-run-on):

#block(inset: (left: 1.2em), grid(
  columns: (auto, 1fr),
  column-gutter: 1em,
  row-gutter: 0.62em,
  [`transfer/particles_to_grid.h`], [public: the function],
  [`transfer/particles_to_grid_cpu.cc`], [private: a CPU loop],
  [`transfer/particles_to_grid_gpu.cu`], [private: a GPU function],
))

The GPU function is run by many GPU threads at once. Callers never see which file runs.

One public function means a scene is written once, and a test runs the same case on both processors by changing
one value. The formulas both sides need, such as weight curves, contact response and material response, live in
shared headers. Each function there is marked `FLUIDITY_CPU_AND_GPU`, so that both the C++ compiler and the CUDA
compiler build it. A formula written twice drifts apart; written once, a fix reaches both processors. Loops,
memory layout and scheduling are not shared, because each processor wants its own.

#figure(
  run-on-figure(),
  placement: auto,
  caption: [The two implementations share their formulas, not their loops. Whole operations are compared within a
    tolerance; the shared formulas that use only correctly rounded operations are checked bit for bit
    (@sec-conventions).],
) <fig-run-on>

== The caller chooses where it runs

Where an operation runs is one value, declared in the shared header `base/run_on.h` so that any shared type can
carry it:

```cpp
enum class RunOn : std::uint8_t { kCpu, kGpu };
```

The caller chooses it when the simulation is built. Nothing ever changes it silently.
When the GPU was chosen and cannot run, the call fails; the code never moves the work to the CPU on its own. A
silent move would make a run far slower without saying why, and it would hide a broken GPU setup.

One function answers whether that choice can run here, before any work starts:

```cpp
std::expected<void, RunOnError> CheckCanRunOn(
    RunOn run_on);
```

It first settles two cases. Asked for the CPU, it succeeds. Asked for the GPU in a build that has no GPU code, it
fails with "GPU not built". Otherwise it runs the GPU math check of @proc-check-can-run-on: it does more than look
for a GPU, it evaluates the shared math on the GPU with fixed inputs and compares the results with the CPU's, bit
for bit. The check runs once per process, at the first call that asks for the GPU; every later call, from any thread,
returns the remembered answer. Many calls ask (every GPU buffer does, @sec-buffer), and the answer does not change
while the program runs. A GPU that fails the check therefore stays failed until the program restarts, and a GPU lost
after the check is reported by the next GPU call that fails.

The check can demand equal bits where whole operations cannot. It computes one value at a time, with no sum whose
order differs between the processors. It evaluates the shared vector and matrix helpers: a cross product, a dot
product, a fixed rotation matrix times a vector, a determinant, a length, a square root and a floor. These use only
operations that IEEE 754 defines as correctly rounded: addition, subtraction, multiplication, square root and floor.
On an NVIDIA GPU, double-precision arithmetic always keeps subnormal numbers, the numbers below the smallest normal
double, and always rounds division and square root correctly. Of the four device flags, only #box[`--fmad=false`]
changes double results: it forbids fused multiply-add, which rounds $a times b + c$ once instead of twice. The other
three, #box[`--ftz=false`], #box[`--prec-div=true`] and #box[`--prec-sqrt=true`], govern single precision; they are
set so that no float code escapes the rule. On the CPU, `-ffp-contract=off` and `/fp:precise` forbid fusing, and the
absence of fast-math keeps subnormal numbers. A fused multiply-add on either processor, flush-to-zero on the CPU (a
fast-math build, or a process that turns it on), or a broken driver therefore shows up as a differing bit.

#procedure(
  [`CheckCanRunOn(RunOn::kGpu)`, the GPU math check],
  <proc-check-can-run-on>,
  inputs: [None beyond the choice `RunOn::kGpu`, in a build that has GPU code.],
  outputs: [Success, or one of the three reasons below.],
  fails: [No GPU found; a GPU call failed; the shared math gave different numbers on the GPU.],
  ([Find a GPU], [Ask the CUDA runtime how many GPUs it sees. Fail with "no GPU found" if there is none or no
    driver answers. Fail with "a GPU call failed" if the runtime reports any other error.]),
  ([Build the fixed inputs], [Generate 32 inputs in the code from a small integer formula, so both processors start
    from the same doubles on every run. Several inputs are so small that their products are below the smallest
    normal double (papers call such numbers subnormal): flush-to-zero would replace them by zero. Several products
    are picked so that fused multiply-add would round their sum differently. None produces a NaN or an infinity,
    because the bit pattern of a NaN can differ between processors even when both are right.]),
  ([Compute on the GPU], [Copy the inputs to the GPU (a `Write`), evaluate the shared math there with one input
    per thread, and copy the results back (a `Read`). Fail with "a GPU call failed" if any call reports an
    error.]),
  ([Compute on the CPU], [Evaluate the same shared math on the same inputs, one after another.]),
  ([Compare bit for bit], [Compare the 64 bits of each GPU result with those of the CPU result, as integers.
    Compare bits, not values: the operator `==` treats +0 and −0 as equal. Fail with "different numbers" at the
    first result that differs.]),
  ([Report], [Succeed. The check changes nothing: it never switches the choice to the CPU. The answer, success or
    the reason for failure, is remembered for the rest of the process; later calls return it without running these
    steps again.]),
)

Module tests compare the two implementations of an operation as whole operations. They run the same cases on both
processors. Discrete outputs, such as ids, counts and which cells are active, must match exactly. Floating-point
outputs must agree within the module's tolerance. For one value computed as `cpu` on the CPU and `gpu` on the
GPU, the test demands

$ |"cpu" - "gpu"| <= "absolute floor" + "relative term" times |"cpu"|. $

A repeat-run test on one GPU demands identical bytes. Until #box[`docs/gpu-runs.md`] records a run on real hardware,
every GPU path is compiled, not run.

== Arrays on the chosen processor <sec-buffer>

Every module keeps its arrays (per cell, per face, per particle) in a buffer from `base/buffer.h`: an array of
fixed size built for one processor.

```cpp
std::expected<Buffer<double>, BufferError> buffer =
    BuildBuffer<double>(RunOn::kGpu, size);
```

Building a GPU buffer first asks `CheckCanRunOn`, whose answer is remembered after the first call (@sec-run-on), so
the math check does not run again for every buffer. A buffer never grows: a new size is a new buffer. Its memory
starts as zero bytes, is aligned to 64 bytes, and is freed when the buffer is destroyed, except for a buffer built
from the run's one block of memory, which frees nothing on its own (@sec-run-memory).

Values cross between the processors only through `WriteBuffer` (CPU memory to a buffer) and `ReadBuffer` (a buffer
to CPU memory) (@fig-buffer-crossings). Every copy to or from the GPU is therefore visible where it is called.
`CopyBuffer` copies between two buffers on the same processor. Two buffers on different processors are a bug in the
caller, not an error the call reports: same processor is a precondition, checked by an assertion in debug builds
only. Each copy also has a form that takes a slice: `count` items starting at index `first`. Sizes, slices and overlap
are checked
before any memory is touched, so a call stopped by one of these checks writes nothing. Two overlapping ranges, such
as two slices of one buffer or a span that points into the buffer it is copied to, are not copied, because a plain
byte copy between them would mix old and new values. A GPU call that fails during the copy is different: the values
of the target slice are then unspecified, a mix of old and new, while values outside the slice are unchanged.

A CPU buffer that is copied to the GPU every step can ask for memory locked for copies, which the GPU reads
directly. Without a usable GPU it gets ordinary memory, since nothing would gain from locking.

#figure(
  buffer-crossings-figure(),
  placement: auto,
  caption: [A buffer lives in the memory of one processor. Values enter or leave a buffer only through
    `WriteBuffer` and `ReadBuffer`, the only calls that cross between CPU and GPU memory. `CopyBuffer` copies
    between two buffers of the same processor; two buffers of different processors break its precondition.],
) <fig-buffer-crossings>

`Buffer` has two layers (@fig-buffer-layers). Its typed functions, templates in `base/buffer.h`, check sizes and
slices, the only checks that need the value type. They then call untyped byte functions (`base/buffer_bytes.h`),
which pick the processor and copy with `memcpy` on the CPU or with the CUDA runtime on the GPU.

#figure(
  buffer-layers-figure(),
  placement: auto,
  caption: [The two layers of `Buffer`. The byte functions exist once for every value type and are defined in one
    CPU source file, so the templates hold no conditional compilation and name no GPU call.],
) <fig-buffer-layers>

=== The memory estimate and the processor description <sec-memory-estimate>

Before a run builds its buffers, it compares the memory they need with the memory the chosen processor has. Each
module lists the arrays it will allocate in a memory estimate (`base/memory_count.h`). One entry of it gives the
bytes of one item, the number of copies, and which count of the scene, such as the particles or the cell blocks,
gives the number of items. `ComputeMemoryBytes` adds the entries as if every array were alive at once, so the sum is
a peak that never falls below the real need. Its products and sums stop at the largest 64-bit number instead of
wrapping around, so a scene too large to count is reported as too large, never as small. `ReadMemoryAvailable` reads
the total and the free memory of the chosen processor: a reading at one moment, not a promise, since other programs
take and give back memory. An entry also says where its memory lives: in the run's memory, always in CPU memory, or
held by the GPU's driver outside every block, as the stack the driver gives each thread a GPU function can run at
once. Contact's particle visit needs about 12 KB of stack per thread, so on a card that holds 55,296 threads at once
the driver holds about 0.7 GB; `BuildContactBuffers` asks for it when it builds the buffers, so a card too full says so
then, not at the first step. `ReadProcessorDescription` gives the run report the processor's name, the software that
drives it and the software the program was built with, as text the machine reports.

@tab-memory-estimate works a small estimate. An entry's bytes are its item count times its bytes per item times
its copies: 1,000 particles with a position of 24 bytes (three doubles), kept in two copies (the committed state and
the try), give 1,000 × 24 × 2 = 48,000 bytes.

#figure(
  align(left, table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, left, right, right, right, right),
    table.hline(stroke: 0.6pt),
    table.header([*Entry*], [*Counted per*], [*Items*], [*Bytes each*], [*Copies*], [*Bytes*]),
    table.hline(stroke: 0.4pt),
    [positions], [particle], [1,000], [24], [2], [48,000],
    [velocities (two per particle)], [particle], [1,000], [48], [1], [48,000],
    [working buffers], [given count], [256], [4], [1], [1,024],
    table.hline(stroke: 0.4pt),
    [*sum*], [], [], [], [], [*97,024*],
    table.hline(stroke: 0.6pt),
  )),
  caption: [A memory estimate of three entries for a run of 1,000 particles. Each entry gives items × bytes each ×
    copies; the sum is the peak with every array alive at once.],
) <tab-memory-estimate>

=== One block of memory per run <sec-run-memory>

A run does not ask the processor for each of its arrays one by one. It reserves one block of memory per processor
at the start, a `RunMemory` (`base/run_memory.h`), sized by the memory estimate, and builds every buffer it keeps
from that block. One reservation has three gains. Memory cannot fragment, because nothing is freed and asked for
again while the run goes on. On the GPU, NVIDIA documents that `cudaMalloc` makes the whole device wait, so one
allocation at the start replaces hundreds of waits. And the memory the run takes equals the plan: a report can
compare the bytes handed out with the bytes planned, and any difference is a bug of the plan. Here `byte_count`
comes from the memory estimate, as the end of this section shows:

```cpp
std::expected<RunMemory, BufferError> block =
    BuildRunMemory(RunOn::kGpu, byte_count);
std::expected<Buffer<double>, BufferError> buffer =
    BuildBuffer<double>(RunOn::kGpu, size, *block);
```

A buffer built from a block takes a piece of it: the bytes right after the pieces handed out before
(@fig-run-memory). Each piece starts at a multiple of 256 bytes from the start of the block, so a piece takes its
byte count rounded up to a multiple of 256:

$ "piece bytes" = 256 times ceil("byte count" / 256), $

and a byte count of 0 takes nothing. 256 bytes is the alignment the CUDA manual promises for `cudaMalloc`, so a
piece is aligned as a buffer of its own would be. The rule is the same on the CPU, so the bytes of a plan do not
depend on the processor. The block itself starts where its processor puts it: at a multiple of 256 bytes on the
GPU, and on the CPU at a multiple of 64 bytes, the promise of every CPU buffer. The block is filled with zero bytes
once, when it is reserved, and a piece is never handed out twice, so every buffer still starts as zero bytes.

Two rules keep this simple. A piece is never freed alone: destroying a buffer built from a block frees nothing,
and its bytes stay taken until the block goes. The block outlives its pieces: it frees all its memory at once when
it is destroyed, so a buffer must not be used after its block is gone. Each block counts its live pieces, and a
debug build stops the program if a block is destroyed while one is still alive. A release build goes on: the
values of those buffers are gone, but destroying them later is still safe, because the block leaves its small
count behind for them. Moving a buffer or a block moves no memory and keeps the pieces where they are.

A piece that does not fit in what is left of the block fails with `kRunMemoryTooSmall`: nothing is taken and
nothing is written, and the block is unchanged. It is an error, not a broken precondition, because the caller
cannot guarantee it: what fits depends on a plan made elsewhere, and limits raised while the run goes on can
cause it. The run reserves exactly what its estimate gives, so the error usually means the estimate missed an
array.

The bytes to reserve come from the same rounding, written once in `ComputeRunMemoryPieceBytes`. Over the entries
kept in one memory (`MemoryWhere`), `ComputeRunMemoryBytes` adds the copies times the piece bytes of one copy, since
every copy is its own buffer and its own piece. This is a contract for every module: each copy of an entry is
exactly one buffer of items × bytes each. A module that keeps 24 bytes per particle as three buffers of 8 bytes
writes three entries, since three pieces round three times: for 10 particles, 3 × 256 = 768 bytes, not 256. A GPU run
reserves one block on the GPU for the `kRunOn` entries and one on the CPU for the `kCpu` entries;
a CPU run reserves one block for both. Applied to @tab-memory-estimate, the positions take 2 × 24,064 = 48,128
bytes (24,000 rounded up), the velocities 48,128 (48,000 rounded up), and the working buffers 1,024, already a
multiple of 256: 97,280 bytes in all, 256 more than the sum without rounding. Building the four buffers from a
block of 97,280 bytes hands out exactly 97,280 bytes.

#figure(
  run-memory-figure(),
  placement: auto,
  caption: [A block of 2,048 bytes after three buffers of 10, 100 and 3 doubles were built from it. Each piece
    starts at a multiple of 256 bytes and takes its byte count rounded up to 256 (dashed: the rounding). The next
    piece starts at the next free offset, which equals the bytes handed out so far.],
) <fig-run-memory>

= Sums in a fixed order <sec-fixed-order-sums>

Every solver tests whether it has converged with a sum, and every conservation check adds up a total. A computer
rounds the result of each addition of two doubles, so the order of the additions changes the result: in
@sec-conventions, (0.1 + 0.2) + 0.3 and 0.1 + (0.2 + 0.3) differ in the last bit. A CPU loop adds from left to
right. GPU threads finish in an order that changes from run to run. Left alone, the two would give different
bits, and the GPU would not even agree with itself.

So every sum of many doubles goes through one fixed order, which depends only on the number of values and their
indexes. The values are padded with zeros up to the next power of two and added in a binary tree (@fig-tree-sum). In
ordinary tree words, each node is its left child plus its right child, and the values are at the bottom of the tree.
This paper and the code call a node a pair sum, and its two children its left side and right side. Each pair sum
adds its left side and its right side, left first, and rounds once; a side is a value or a pair sum of the layer
below. Each layer halves the number of sums, so seven values padded to eight take three layers. The pair sum at the
top covers every index and is the result.

#figure(
  tree-sum-figure(),
  placement: auto,
  caption: [The tree over seven values. Each box is labelled with the indexes it covers. Index 7 is padding, a zero
    that is never kept in memory, so the pair sum 6–7 is value 6 plus +0.0. On the GPU each row is one layer, all of
    whose pairs are added at the same time.],
) <fig-tree-sum>

The padding is never written to memory. A pair whose right side covers only padding adds +0.0 to its left side,
because a sum of zeros is +0.0. Adding +0.0 gives back every number unchanged, bit for bit, with one exception:
in IEEE arithmetic −0.0 + 0.0 is +0.0. (A NaN stays NaN, though its bits may change: a signalling NaN comes back
quiet.) So the sum of the single value −0.0 is −0.0, since one value needs no padding and no addition. The sum
of three values −0.0 is +0.0, because the third one is paired with padding.

IEEE 754 defines the rounded result of every addition exactly, and the tree fixes which pairs are added. The CPU and
the GPU therefore give the same bits for these sums, not only the same bits from run to run, unless the result is
NaN, whose bits IEEE 754 leaves open. The dot product, the sum of `first[i] * second[i]`, rounds each product once
before the tree adds it. A fused multiply-add would round a product and the next addition together, so the flags
#box[`-ffp-contract=off`] and, on MSVC, #box[`/fp:precise`] on the CPU and #box[`--fmad=false`] on the GPU forbid
it. The largest magnitude, the largest absolute value, needs no fixed order: picking the larger of two values rounds
nothing. It is NaN when any value is NaN, so a NaN in a solver can never hide behind a comparison.

The order changes the answer. Take the four values 1e16, 1, −1e16 and 1. Near 1e16 the doubles are 2 apart, so
1e16 + 1 lies halfway between two doubles and rounds to 1e16 (a tie goes to the double whose last bit is even).
From left to right, 1e16 + 1 gives 1e16, minus 1e16 gives 0, and plus 1 gives 1. The tree adds the pairs first:
#box[(1e16 + 1) + (−1e16 + 1)] is #box[1e16 + (−1e16)], which is 0. The exact sum is 2. Neither 1 nor 0 is "the
right answer". The point of the tree is that its order is fixed, not that it is more accurate. A tree is chosen
over a left-to-right loop because a GPU can add all the pairs of one layer at the same time, and because the bound
on the rounding error grows with the depth of the tree, about log#sub[2] of the count, rather than with the count.

#procedure(
  [`math::TreeSum(values, count)`, the sum in a fixed order],
  <proc-tree-sum>,
  inputs: [`count` doubles, `values[0]` to `values[count − 1]`.],
  outputs: [Their sum: the same bits on every run and on both processors, unless the result is NaN, whose bits
    IEEE 754 leaves open.],
  ([No values], [If `count` is 0, return +0.0.]),
  ([Add the pairs with real values on the right], [Think of the values as padded with zeros up to the next power of
    two. Every pair sum whose right side covers at least one real value is its left side plus its right side,
    left first, rounded once.]),
  ([Add the pairs with padding on the right], [Every pair sum whose right side covers only padding is its left
    side plus +0.0. This gives the bits that adding the padded zeros one by one would give.]),
  ([Return the top], [Return the pair sum that covers every index. With one value, that is the value itself, and
    no pair is added.]),
)

The steps fix which two sides make each pair sum, not when it is computed. The GPU computes one layer at a time, all
the pairs of a layer at once, and keeps the sums of a layer in temporary GPU memory. That memory belongs to a
`SumBuffers`, working buffers built once for up to a largest count and passed to every sum, so a solver that sums
every iteration allocates nothing. The CPU reads the values once
from left to right and computes each pair sum as soon as its two sides exist. It keeps the left sides that still
wait for their right side, at most one per layer. A count fits in 64 bits, so 64 waiting sums are enough, and the
CPU code needs no memory beyond them, and its sum buffers hold nothing. Both add the same pairs, so both give the
same bits. On the GPU, the dot product writes its products to the sum buffers first and then sums them with the same
steps;
the CPU computes each product as the tree reaches it.

= Ordering that never depends on timing <sec-stable-order>

The grid builder, the particle-to-grid transfer and the batches all put items in order before they add anything:
particles by the cell that holds them, cells by their id, items by the place they write to. Each item gets an
unsigned 64-bit key, and `BuildStableOrder` in `base/stable_order.h` returns the order of the items by ascending
key: `order[i]` is the index of the item at place `i`.

Many items share a key. Two particles in the same cell have the same cell key, and they write to the same faces.
The order in which their contributions are added decides the rounding, so it decides the bits of the face sums
(@sec-fixed-order-sums). A sort that may swap two items with equal keys would let that order change from run to
run, or between the CPU and the GPU. So the sort is stable: items with equal keys keep the order they came in,
which is index order, or the order of a given `previous_order`. Stability also makes two keys easy. Sort by the
minor key first, then sort by the major key with the first order as `previous_order`. Items with equal major keys
then keep their minor order.

The sort works on one digit of the key at a time, a digit being one byte, from the lowest byte up (papers call
this a least-significant-digit radix sort). Each digit takes one counting pass (@fig-stable-order). The pass
counts how many items have each of the 256 digit values. An exclusive prefix sum turns these counts into first
places: the first place of a value is the number of items with a smaller value. Then the pass walks the sequence
from front to back. Each item goes to the first place of its digit plus the number of earlier items with the same
digit. An earlier item therefore always lands before a later item with the same digit, so the pass is stable.
After the pass for the highest byte, the items are in order of their whole key, and ties are still in their
starting order.

#figure(
  stable-order-figure(),
  placement: auto,
  caption: [One counting pass over eight items. The boxes hold item indexes, and the digit of each item is above
    it. The counts per digit, added up from the left, give each digit's first place. Items 0, 3 and 6 share digit 2:
    they fill places 4, 5 and 6 in the order they came in.],
) <fig-stable-order>

#procedure(
  [`BuildStableOrder(keys, count, previous_order, order, buffers)`, the order by key that keeps ties],
  <proc-stable-order>,
  inputs: [`count` keys of 64 bits; optionally `previous_order`, a permutation of the indexes `0` to
    `count − 1`.],
  outputs: [`order`: the indexes by ascending key; equal keys in the order of `previous_order`, or of their
    index.],
  ([Start], [The sequence is `previous_order`, or the indexes `0`, `1`, ..., `count − 1`.]),
  ([Count the digits], [For the current byte of the key, starting with the lowest, count how many items of the
    sequence have each of the 256 digit values.]),
  ([Find the first places], [The exclusive prefix sum of the counts gives the first place of each digit value.]),
  ([Place the items in order], [Walk the sequence from front to back. An item with digit `d` goes to the first
    place of `d` plus the number of earlier items in the sequence with digit `d`. The result is the next
    sequence.]),
  ([Next byte], [Repeat steps 2 to 4 for each of the eight bytes. A byte that is the same for every item may be
    skipped: its pass would leave every item where it is.]),
  ([Return], [The last sequence is the order.]),
)

Only integers take part: counts, places and indexes, never a rounded value. Where an item lands is a formula of
the keys and of the starting sequence, not of which thread finished first. The CPU and the GPU compute the same
formula, so they give the same order, bit for bit, and so does every run. The CPU counts the digits of all eight
bytes in one read of the keys, since a count does not depend on the order of the sequence, and skips the bytes
that are equal for every item. The GPU runs all eight passes; skipping changes nothing, so the results agree.

On the GPU, the sequence is cut into chunks of 1,024 places, and one thread block of 256 threads (the threads of one
GPU launch unit, which can share fast memory and wait for each other) takes one chunk. Thread `d` of the thread block
reads the chunk's digits from front to back and counts the items with digit `d`. A table holds these counts, digit
by digit, and chunk by chunk within a digit. Its exclusive prefix sum gives, for each digit and chunk, the place where
that chunk's first item with that digit lands: all items with a smaller digit come before
it, then the items with the same digit in earlier chunks. Thread `d` then walks its chunk again and places its
items one after another from there. No two threads write the same place, no counter is shared, and no thread's place
depends on when another thread runs.

`ComputeExclusivePrefixSum` is the prefix sum both processors use, also on its own: `sums[i]` is the sum of the
values before index `i`, and `sums[count]` is the total. The arithmetic is exact. The total is computed first, in
64 bits, and a total above 2#super[32] − 1 is reported as an error before anything is written. On the GPU the
values are cut into chunks of 1,024. Each chunk's total goes to a layer above, whose chunks are added up the same
way, until one total is left. Then each layer is turned into running sums from the top down, and each chunk adds
its running sum to the sums of its own values. In-place use is safe, because a thread block reads all values of its
chunk before it writes any sum.

`BuildGroupsByDestination` puts items into groups: `item_order` lists the items of destination 0, then destination
1, and so on, and `first_of_group[d]` is where the group of destination `d` starts. Like the order and the prefix sum,
it takes an `OrderBuffers`, working buffers built once, and allocates nothing. It is one stable order with the
destination as the key, so each group lists its items in ascending index. The start of a group is the number of
items with a smaller destination, so an empty group starts where the next one does.

#procedure(
  [`BuildGroupsByDestination`, items grouped by destination],
  <proc-groups-by-destination>,
  inputs: [`item_count` destinations, each below `destination_count`.],
  outputs: [`item_order`: the items of destination 0, then 1, and so on, each group in ascending index;
    `first_of_group`: where each group starts, `destination_count + 1` entries.],
  fails: [a buffer is too small, or a destination is `destination_count` or more; nothing is written. Every buffer
    on one processor is a precondition, checked in debug builds only.],
  ([Check the buffers], [Each buffer is large enough.]),
  ([Take the order buffers], [The caller's `OrderBuffers`, built for at least `item_count` items.]),
  ([Check the destinations], [Every destination is below `destination_count`, before anything is written.]),
  ([Order the items], [@proc-stable-order with the destination as the key and no previous order.]),
  ([Find the group starts], [The start of group `d` is the number of items whose destination is below `d`. The
    CPU counts each destination and adds the counts up from the left; the GPU searches the destinations in group
    order for the first one not below `d`.]),
)

= Contact with solids <sec-contact>

A particle travels in a straight line during one step. When that line meets a solid, the particle must not go
in. Contact moves one particle for one step and stops it at solid surfaces: it finds the first time the line meets
a solid (the hit), removes the speed into the solid, takes away some sideways speed by friction, and spends the
rest of the step sliding along the surface (the slide). The straight path of one step is the sweep. When contact
cannot answer a step safely it fails with a close call. Some close calls are cured by a shorter step and some are not:
a particle squeezed between solids that already touch it is squeezed at every step length. The failure's
`can_shorter_step_cure` says which: true only for a close call met at a hit after the start of the step, which a step
that ends before the hit does not meet, or whose test reads the time left.

The same code moves particles of every material: liquids, snow, springy solids and gas markers. Only two values
differ between them: the friction number and the plate side a particle may carry. The code has no case for one
material. Contact does not decide how a solid moves from step to step, nor the force on it; it reports, for each
particle, what the solid did to it, and the solids module adds those reports up.

Contact works on both velocities of a particle (@sec-particle-velocities). The sweep follows `velocity.for_this_step`,
since that is the velocity that moves the particle. At a hit both velocities get the response, each with its own
approach speed. In this section, $T$ is the step duration (the same value as $Delta t$), $"tol"$ the contact tolerance,
$d$ the distance from a point to a solid and $g = d - "tol"$ the gap (@tab-notation).

Two functions do the work. `MoveParticle` moves one particle; it lives in a shared header, so the CPU loop and the GPU
thread run the same code (@sec-run-on). `MoveParticles` runs it for every particle of one material in one range of cell
blocks: a CPU loop, or one GPU thread per particle. Particles are independent, so contact adds no floating-point values
across particles, and nothing in it depends on the order in which threads finish. A call first checks itself once:
sizes, pointers, overlapping arrays, the scalars and every solid. Then it visits each particle once, moves it and writes
its outputs. At every hit on one surface, the speeds involved times the time left must stay within $2^40 "tol"$, the
bound that keeps a response's rounding below $"tol"\/64$; otherwise the move is too close to call
(`kMovesTooFastForStep`), and a shorter step cures it. Speeds beyond any scene, a sum of absolute axes above $2^480$ m/s
for a velocity or a solid's motion, are bad input. So are coordinates beyond the scale the tolerance was computed for:
on some axis, a particle's start plus its travel at `for_this_step` over the step, or a solid's origin over the step
plus its reach, above $2^40 "tol"$, the premise `MoveParticle`'s rounding bounds rest on. The outputs belong to the
step's try: after a failure their contents are unspecified, and the step discards the whole try. When several particles
fail, the one reported is the failing particle with the lowest id, then the lowest index. The CPU loop keeps it as it
goes. The GPU cannot: each thread writes its particle's failure beside the particle, and three short GPU functions
follow the visits: the smallest failing id, then the smallest index among the failing particles with that id, both
atomic integer minima, then that particle's failure. A flag, set by an atomic or, says whether any particle failed, so
no id stands for "none"; the counts are atomic integer sums. Integer minima, an or and integer sums end with the same
values in every order of the threads, so the report is the same on every run and in every launch size; no floating-point
value is summed. The CPU and the GPU give equal discrete results (kinds, solids, faces, counts, errors). With no turning
or spinning solid the floating-point results have the same bits: every formula uses only $+$, $-$, $times$, $\/$,
absolute values and square roots, each correctly rounded on both processors, and neither may fuse a product into an
addition (`-ffp-contract=off`, `/fp:precise`, `--fmad=false`, @sec-conventions). A turning or spinning solid is posed
through a rotation built from sines, whose last bits differ between the processors' libraries, so its results agree
within $f + 2^(-40) |a|$ of the CPU's value $a$, with the floor $f$ a quarter of the contact tolerance for positions,
$2^(-40)$ times the largest input speed for velocities, and $2^(-40)$ for hit fractions and normals.

Most particles are far from most solids, so the cost of a move is mostly its reads of the solids. The start's depth
check reads each solid once. A solid whose distance there is above the particle's reach $R$ (its travel at the start's
velocity plus one move off one near solid at the start and one in the settle), plus how far the solid's distance can
fall in the step, plus $4 "tol"$, cannot come within any window a phase reads, and the move leaves it out of its other
reads. The distance falls by at most $|bold(v)_"lin"| T$ plus the chord of the solid's turn, $min(|bold(omega)| T, 2)$,
times an arm: the half diagonal for a box or a cylinder, none for a sphere, the particle's distance to the origin for a
flat solid or a plate. A turning box's distance is its largest distance to a face plane, which a turn moves by the
particle's arm, not the box's; it is at least the true distance over $sqrt(3)$, so the box's window counts twice. The
particle's distance from the start plus the travel still to come is checked against $R$ before every sweep and before
the settle, and a move off the start longer than one move off one near solid reads them all as well; past either, every
solid is read again. In the sweeps the move allowance alone keeps the distance above the largest window; the $4 "tol"$
carries the bound at the settle and the end check, and the box's double count is needed only by the stepping search of a
turning box. Phase by phase, a solid left out is above every window a phase reads, with room for the rounding of a
distance ($"tol"\/128$) and of a closed form's hit (about $0.013 "tol"$, a first-order estimate under `MoveParticle`'s
premise that every coordinate is within $2^40 "tol"$). That the answers are the same bits as with every solid read is
measured, not proved: on 40,240 test scenes and 71.7 million generated ones, none differs. A particle far from 32 solids
costs one read of each instead of three.

== Solid shapes <sec-solid-shapes>

A solid is a shape with its own axes. Its pose is an origin $bold(o)$, in metres, and a rotation $A$ whose columns
are the solid's own x, y and z axes written in world coordinates. The pose is given at the start of the step. The
motion is constant during the step: a linear velocity $bold(v)_"lin"$ of the origin, in m/s, and an angular velocity
$bold(omega)$ about the origin, in rad/s, whose direction is the axis of the turn and whose length is its rate.
After `elapsed` seconds $t$,

$ bold(o)(t) = bold(o) + bold(v)_"lin" t, quad A(t) = "Rot"(bold(omega) t) A, $

where $"Rot"(bold(r))$ turns by $|bold(r)|$ radians about $bold(r)$ (`math::Rotation`). The velocity of the solid's
surface at a world point $bold(x)$, the solid velocity, is that of a point fixed to the solid:

$ bold(u) = bold(v)_"lin" + bold(omega) times (bold(x) - bold(o)(t)). $

For example, a solid with linear velocity (1, 0, 0) m/s and angular velocity (0, 0, 2) rad/s, origin at
(0, 0, 0), moves at (1, 0, 0) + (0, 0, 2) × (0, 3, 0) = (−5, 0, 0) m/s at the point (0, 3, 0).

Each solid also carries a stable nonzero `solid_id`, unique in the scene, a `friction_scale` that multiplies the
particle's friction at this solid (1 by default, 0 for a smooth solid), and `reports_reaction`, true for driven and
free solids, whose motion the liquid's push matters for. The solids arrive in an array in ascending `solid_id`,
and contact visits them in that order.

Six shapes exist (@fig-solid-shapes, @fig-round-solids). Each is described in its own axes, with $bold(p) = A^T
(bold(x) - bold(o))$ the point in those axes:

/ Flat solid: fills $p_z <= 0$. It is given by a point on its surface and its `normal_out_of_solid`. Its own z axis is
  the normal, and its own x axis is the cross product of the world axis on which the normal has its smallest part (the
  first in x, y, z order on a tie) with the normal, normalised; a plate builds its axes the same way. Its one face is
  `kHighZ`.
/ Plate: fills $|p_z| <= "half_thickness"$, unbounded sideways. Its `normal` points to its `kAlongNormal` side, and its
  `kHighZ` face is on that side; the `kLowZ` face looks to the `kAgainstNormal` side. A thin solid of finite size is a
  box, which has no sides in this sense.
/ Box: fills $|p_x| <= "half_size"_x$, $|p_y| <= "half_size"_y$ and $|p_z| <= "half_size"_z$. Its six faces are
  `kLowX`, `kHighX`, `kLowY`, `kHighY`, `kLowZ` and `kHighZ`, named in its own axes.
/ Sphere: given by its `center` and `radius`, it fills $|bold(p)| <= "radius"$; its origin is its centre
  $bold(c)$ (@sec-round-solids).
/ Cylinder: a finite solid cylinder given by its `center`, its `axis` (a unit vector), its `radius` and its
  `half_height`. Its origin is its centre and its own z axis is its `axis`, the other two built as a plate builds
  them from its normal. With $rho = sqrt(p_x^2 + p_y^2)$, the distance from that axis, it fills $rho <= "radius"$
  and $|p_z| <= "half_height"$. Its round surface is the wall, its two flat ends are the caps, and the circles where
  they meet are the rims (@sec-round-solids).
/ Domain box: the box the particles live in. It is fixed and aligned with the world axes; the particles are
  inside. Each closed side is an unbounded flat face whose normal points into the box. Its sides carry the Blender
  names of @sec-conventions: left (−X, `kLowX`), right (+X, `kHighX`), front (−Y, `kLowY`), back (+Y, `kHighY`),
  bottom (−Z, `kLowZ`) and top (+Z, `kHighZ`). By default every side is closed except the top. A particle that
  leaves through an open side is removed by the particles module, not by contact.

#figure(
  solid-shapes-figure(),
  placement: auto,
  caption: [The four shapes with flat faces, each cut by the x–z plane of its own axes. Every shape is described
    in its own axes, so one formula per shape serves any position and turn. The plate has two sides, and the
    domain box keeps its particles inside, with Blender's names on its sides. The sphere and the cylinder are drawn
    in @fig-round-solids.],
) <fig-solid-shapes>

The distance $d$ from a point to a solid is positive outside the solid and negative inside. Per shape with flat
faces, with $sigma = -1$ for a low side and $+1$ for a high side of the domain box:

$
  d = cases(
    p_z & "flat solid",
    |p_z| - "half_thickness" & "plate",
    max_(k in {x, y, z}) (|p_k| - "half_size"_k) & "box",
    min_("closed sides") ("half_size"_k - sigma p_k) & "domain box",
  )
$ <eq:solid-distance>

where a domain box's origin is its centre and its half size is half its extent. The sphere and the cylinder use their
true distance to the surface (@eq:sphere-distance, @eq:cylinder-distance). The normal, `normal_out_of_solid`, belongs to
the face that gives $d$, written back in world axes with $A$. A flat solid's normal is its own +z. A plate's is +z with
face `kHighZ` when $p_z >= 0$, else −z with `kLowZ`. A box's face is the one on the axis that gives the maximum: the low
face when $p_k <= 0$, else the high one. A domain box's face is the closed side that gives the minimum, with its normal
pointing into the box. The face order is `kLowX`, `kHighX`, `kLowY`, `kHighY`, `kLowZ`, `kHighZ`, `kCurved`, `kLowRim`,
`kHighRim`. When two faces of a box or a domain box give the same value, the first in that order is taken; so a box's
middle plane $p_k = 0$ gives its low face. Two other shapes settle their ties by their own rules instead: a plate and a
cylinder's caps take `kHighZ` on their middle plane $p_z = 0$ (the rule above and @sec-round-solids), and a tie inside a
cylinder goes to its wall.

A box's $d$ is its largest distance to a face plane, negative behind the plane, not the straight-line distance to its
surface. The two agree everywhere except outside an edge or a corner, where $d$ is smaller. For a box with half sizes
(1, 1, 1) at the origin, the point (3, 0, 0) is at $d = 2$ from face `kHighX` with normal (1, 0, 0); the point (0.9, 0,
0.5) is inside at $d = -0.1$, again from `kHighX`; the point (−2, −2, 0) is at $d = 1$ from two faces at once, and the
tie gives `kLowX`. For a domain box from (0, 0, 0) to (1, 1, 1) with an open top, the point (0.5, 0.5, 0.95) is at $d =
0.5$: four sides tie and left wins. With the top closed it would be at 0.05 from the top.

=== The two sides of a plate <sec-plate-sides>

A plate is thin, often thinner than a cell. Particles on its two sides must stay on their own side, even when the
grid around them cannot tell the sides apart; papers on moving thin solids keep the two sides apart in the same way
[3]. A point is on the `kAlongNormal` side when $p_z > "half_thickness"$, on the `kAgainstNormal` side when $p_z <
-"half_thickness"$, and on no side inside the thickness (`FindSolidSide`). A particle can be committed to one side
of one plate, `PlateSide { plate_id, side }`: an identity, not a number to compute with. Two particles may exchange
values through the grid only when their plate sides are equal (`same_side`).

For a particle committed to side $s_"plate"$ ($+1$ along the normal, $-1$ against it), contact treats that plate as
the half space on its side:

$ d = s_"plate" p_z - "half_thickness", $

with face `kHighZ` for `kAlongNormal` and `kLowZ` for `kAgainstNormal`. This one distance is used when the start is
classified, in the sweep, when the particle settles and in the end check. A committed particle that starts inside the
thickness is therefore pushed out to its own side, never through to the other one. For a plate of half thickness 0.01 m
and a `start_overlap_tolerance` of 0.05 m (the tests' values), a particle committed to `kAlongNormal` at $p_z = -0.001$
m is at $d = -0.011$ m and goes to $p_z = 0.01 + "tol"$; an uncommitted particle at the same place is nearer to `kLowZ`
and goes to $p_z = -(0.01 + "tol")$. A committed particle more than $"tol"$ beyond the other face, $s_"plate" p_z <
-("half_thickness" + "tol")$, is on the wrong side of its plate, an error. This is checked before the depth inside the
half space: a particle on the other side is not inside its half space but on the wrong side, and the error says so. A
`plate_id` that names no plate among the solids is a bad input.

=== Checks before the step <sec-checks-before-step>

`CheckSolid` checks one solid: finite values and a step duration $T > 0$; axes that form a rotation, each column a unit
vector up to a squared length within $2^(-40)$ of 1, each pair of columns at right angles up to a dot product of at most
$2^(-40)$ in absolute value, and a positive determinant, so the axes are not a mirror image; positive sizes, with a
sphere's three half sizes equal and a cylinder's first two equal (both are stored in `half_size`: the radius three times
for a sphere, the radius twice and the half height for a cylinder); a nonzero id; a `friction_scale` of zero or more; a
domain box that does not move, keeps exactly the world axes ($A = I$) and has at least one closed side; and a turn of at
most a quarter turn during the step, $|bold(omega)| T <= pi\/2$. A larger turn asks for a shorter step. For the sphere,
the cylinder, the flat solid and the plate only the turn that changes the shape counts (@sec-round-solids,
@sec-first-hit): a sphere never fails this check, a cylinder may spin about its own axis at any rate, and a flat solid
or a plate about its normal. The step calls it once per step for every solid; `MoveParticle` trusts that call and does
not repeat it.

=== The nearest solid surface <sec-nearest-solid-surface>

Other modules need the velocity of the solid surface nearest a point, for example where the grid meets a solid.
`ComputeSolidVelocitySample` gives it: it poses every solid `elapsed` seconds into the step, reads the distance $d$ and
the normal $bold(n)$ of @eq:solid-distance at the point, and keeps the smallest distance; equal distances keep the
solid that comes first in the array. A surface farther than the caller's search distance gives no solid. Otherwise the
sample holds the solid's index, $d$, $bold(n)$, the side of a plate the point is on, and the solid velocity $bold(u)$
at $bold(x) - d bold(n)$: the nearest point of the surface, except outside a box's edge or corner, where it is the
point on the face plane that gives $d$. For a box with half sizes (1, 1, 1) at the origin turning at (0, 0, 1) rad/s
and the point (3, 0, 0) with a search distance of 5 m: $d = 2$, $bold(n) = (1, 0, 0)$, the surface point (1, 0, 0) and
$bold(u) = (0, 1, 0)$ m/s; with a search distance of 1.5 m, no solid (the code's own example). Contact itself does
not use it: a moving particle reads every solid near it.

== The contact tolerance <sec-contact-tolerance>

A computer evaluates $d$ with rounding. A particle placed exactly on a surface, at $d = 0$, may later be computed
at $d = -10^(-17)$ m by another module with its own order of operations, and so be found inside the solid. The
proof of concept guarded against this with error bounds around every sign and by moving velocities by one
representable step. This code uses one length instead, the contact tolerance:

$ "tol" = 2^(-30) w_"min" + 2^(-40) L $ <eq:contact-tolerance>

where $L$, the `largest_coordinate`, is the largest $|x|$, $|y|$ or $|z|$ a solid origin takes during the step, plus
each solid's reach from its origin, and $w_"min"$, the `smallest_cell_width`, is the width of the smallest cell of the
grid (`ComputeContactScale`, `ComputeContactTolerance`). The reach is the length of the half size of a box, a sphere, a
cylinder or the domain box; the domain box lets the particles' extent count, since every particle is inside it.

Why $2^(-40)$: write $epsilon = 2^(-52)$ for the machine epsilon, the gap between 1 and the next double. Computing $d$
takes a subtraction of the origin, a rotation made of three-term dot products and the shape's own formula. Together they
misplace $d$ by at most about $32 epsilon L$. A hit is placed about as finely: to first order the closed forms of
@sec-round-solids put it within $14 epsilon M + 3 epsilon L$, $M$ the largest own coordinate along the path, at most
$"tol"\/64$ when every coordinate is within $2^40 "tol"$ (@eq:sphere-discriminant). That premise is `MoveParticle`'s
Requires: the check below tests thickness only, and only the $2^(-40)$ share of the tolerance, with the particles inside
$L$ as a domain box keeps them, gives it. The relative term $2^(-40)$ is $4096 epsilon = 128 times 32 epsilon$, so
$"tol"$ is at least 128 times that error. The term in the cell width keeps $"tol"$ positive for a scene at the origin.

A worked example, with numbers of this paper's own: in a scene whose largest coordinate is 10 m and whose
smallest cell is 0.01 m wide, $"tol" = 9.31 times 10^(-12) + 9.09 times 10^(-12) = 1.84 times 10^(-11)$ m. The
rounding of $d$ is at most about $7.1 times 10^(-14)$ m, 259 times smaller. Every feature must be at least $1024
"tol" = 1.88 times 10^(-8)$ m thick.

Three rules use the tolerance (@fig-contact-tolerance):

+ *The particle keeps a rest gap.* Every sweep runs against the solid grown by $"tol"$: a hit is the moment $d =
  "tol"$, and a particle in contact rests at $d = "tol"$, never at $d = 0$. Any later evaluation of $d$, by any
  module with its own rounding, is therefore positive. This is what that one-step move guaranteed.
+ *The end is checked.* After the move, every solid must be at $d >= "tol"\/2$ at the pose of the end of the
  step. Otherwise the call fails with "too close to call, ends too close to a solid", and the step is tried again
  shorter.
+ *The scale is checked.* `CheckContactTolerance` fails with "solid too thin" unless $1024 "tol"$ is at most the
  thinnest solid and at most the smallest cell width. The thinnest solid is the smallest of the plate
  thicknesses, the box and domain box widths (twice the smallest half size), the sphere diameters and the
  cylinder diameters and heights, or the cell width when there is none of these. Since $1024 times 2^(-40) =
  2^(-30)$, this fails only for a feature about $10^9$ times smaller than its distance from the origin. The
  tolerance is never enlarged to make contact pass: a scene it cannot protect fails the check.

For a scene with a plate of half thickness 0.002 m and a box of half sizes (1, 2, 3) at (10, 0, 0), the thinnest
solid is 0.004 m and the largest coordinate is $10 + |(1, 2, 3)| = 13.74$ m.

#figure(
  contact-tolerance-figure(),
  placement: auto,
  caption: [A zoom on a solid surface, not to scale. The rest gap at $d = "tol"$ lies far outside the band where
    rounding can misplace the surface, and the end check at $"tol"\/2$ still does. On the right, the thinnest
    plate the tolerance protects: 1024 tolerances thick.],
) <fig-contact-tolerance>

A second length is the user's: `start_overlap_tolerance`, 0 by default, says how deep a particle may start inside
a solid before that is an error.

== Finding the first hit <sec-first-hit>

`FindFirstHit` takes a straight path $bold(x)(t) = bold(x)_0 + bold(v) t$ for $0 <= t <= "duration"$, with $bold(v)$ the
particle's `velocity.for_this_step`, and finds the first time it comes within $"tol"$ of a solid: the first time the gap
$g = d - "tol"$ reaches zero. The solids are sampled $t_0 + t$ seconds into the step, where $t_0$ (`start_elapsed`) is
the part of the step already used. It returns the time, the solid's index in the array and the distance, normal and face
at the hit, or no solid when the path is free. @proc-find-first-hit lists the steps; this part explains them.

=== A solid that does not turn <sec-not-turning>

A box turns when its angular velocity is not exactly (0, 0, 0). A flat solid or a plate turns when the part of its
angular velocity across its normal, $bold(omega) - (bold(omega) dot bold(e)_z) bold(e)_z$ with $bold(e)_z$ the normal,
is longer than $2^(-46) |bold(omega)|$: both are unbounded sideways, so a spin about the normal leaves the shape as it
is, and a part across that short is the rounding a spin about a tilted normal leaves, not a turn. Such a part keeps the
normal within $2 times 2^(-46)$ of its start however far the solid spins, so the face moves by less than $"tol"\/8$ at
any point within $2 sqrt(3) L$ of its origin, $L$ the largest coordinate of the contact scale (`kNegligibleTurnShare`;
the proof is in the code). The spin still moves the surface, and friction feels it through the solid velocity. A sphere
never counts as turning, and a cylinder only when its axis tilts (@sec-round-solids). One that does not turn only
translates, so each of its face planes moves at $bold(v)_"lin"$, and the gap to a face changes linearly with time:

$
  g_"face" (t) = g_"face" (0) + "rate" times t, quad "rate" = bold(n) dot (bold(v) - bold(v)_"lin"), \
  t_"hit" = (g_"face" (0)) / (-"rate") quad "when rate" < 0
$ <eq:hit-time>

A flat solid has one face, and each closed side of a domain box is its own flat face: the path hits it at
$t_"hit"$ when $g(0) > 0$, $"rate" < 0$ and $t_"hit" <= "duration"$. For example (the example of the code's
documentation), a flat solid with surface at $x = 1$ and normal (−1, 0, 0), and a particle at $x = 0.9$ moving at
+1 m/s for $T = 0.2$ s: $g(0) = 0.1 - "tol"$ and $"rate" = -1$ m/s, so the hit is at $0.1 - "tol"$ s, the
fraction $(0.1 - "tol")\/0.2$ of the step, and the particle ends at $x = 1 - "tol"$ with speed 0 and an approach
speed of −1 m/s. With the tolerance $2^(-30)$ m that the tests use, the fraction is $0.5 - 5 "tol" approx
0.4999999953$.

A plate and a box are convex: the path is inside the grown solid exactly while it is behind every face plane at
once. Each face with $g_"face" (0) > 0$ and $"rate" < 0$ is entered at $g_"face" (0)\/(-"rate")$; the
entry time is the largest of these, or 0 when there is none. Each face with $g_"face" (0) <= 0$ and $"rate" > 0$ is
left at $-g_"face" (0)\/"rate"$; the exit time is the smallest of these. A face with $g_"face" (0) > 0$
and $"rate" >= 0$ is never reached, so there is no hit. Otherwise the path hits when entry $<=$ exit and entry
$<=$ duration. The face reported is the one entered last, the one that gives the entry time; equal entry times keep
the first face in face order. For a box of half size 0.5 m at the origin and a path from (−1, 0.1, 0.2) at
(2, 0, 0) m/s, face `kLowX` gives entry $(0.5 - "tol")\/2$ s, face `kHighX` gives exit $(1.5 + "tol")\/2$ s,
and the y and z faces are neither entered nor left: the hit is at $0.25 - "tol"\/2$ s on `kLowX`. A plate seen by a
particle committed to one of its sides is a single face (@sec-solid-shapes). A sphere and a cylinder have closed
forms of their own, in @sec-round-solids.

=== A particle resting on a surface <sec-touching-rule>

After every hit the particle sits on the surface at $d = "tol"$ and is swept again for the rest of the step. If that
surface counted as hit again at once, every slide would end after a few steps in "too many hits". So the surface a
particle rests on is treated apart, by the touching rule: when the start's gap $g(0)$ is in $[-"tol", 0]$ for the
surface that gives $d$, that surface is not hit when $"rate" times "duration" >= -"tol"$, and it is hit at $t = 0$
otherwise. The surface that gives $d$ is: for a plate or a box, the face whose plane gives the largest distance; for a
flat solid or a closed side of a domain box, its one face. The reason: the solid lies entirely behind that face plane
(true for a plane, a convex solid and a domain side alike), so the gap along the path is at least $g(0) + "rate" times t
>= g(0) - "tol"$. `MoveParticle` starts every sweep at $d = "tol"$ (a hit or the start move put it there, a rim hit
within $"tol"\/1024$ of it), so $g(0) = 0$ and the path never goes below $d = 0$; the part of the window below 0 only
absorbs rounding. A sphere and a cylinder follow the same rule at the nearest point of their surface
(@sec-round-solids). The touching rule belongs to the closed forms of a solid that does not turn; a turning solid has
none (next part). A start deeper than $"tol"$ inside the grown solid, $g(0) < -"tol"$, is hit at $t = 0$; `MoveParticle`
never sweeps from there. For a flat solid with surface at $z = 0$, a particle at $z = "tol"$ moving at $(5, 0,
-"tol"\/4)$ m/s for 1 s slides freely; at (5, 0, −1) m/s it hits at 0.

A convex solid also lies behind surfaces that do not give the distance, its *bounding surfaces*: each face plane of a
box but the one that gives the distance, a cylinder's wall taken as unbounded along its axis, and the plane of each cap.
A start *rests* on one of them (for a cylinder, beside its rim, where neither wall nor cap gives the distance) when its
gap $g_b$ to it is in $[-"tol"\/8, "tol"\/16]$ and the path sinks toward it by at most $"tol"\/64$ during the sweep,
$dot(g)_b times "duration" >= -"tol"\/64$; the sweep is then free of the whole solid. It is safe for the reason the
touching rule is: the solid's distance is at least $g_b + "tol"$ at every point, and along the path $g_b$ is at least
$g_b (0) + dot(g)_b t$ (a plane's gap is linear in time, the distance from an axis convex), so a resting sweep stays at
least $"tol" - "tol"\/8 - "tol"\/64 = 55\/64 "tol"$ from the solid; the settle lifts it back to the rest gap. Below,
$"tol"\/8$ holds a particle that the settle left within $"tol"\/128$ of the rest gap and a response brought up to
$"tol"\/16$ nearer; above, $"tol"\/16$ is the touched window; the sink, $"tol"\/64$, is what a response may leave. What
it gives up: a path the true distance would stop up to $9\/64 "tol"$ inside the rest gap is not hit, and a ledge lower
than $"tol"\/8$ between two solids is not seen. It lets a particle cross the seam of two flush solids. Two drums of one
radius stacked on one axis: a particle sliding down the upper drum's wall at the rest gap reaches the lower drum beside
its rim, rests on the lower wall's unbounded surface, and keeps $(0, 0, -1)$ m/s bit for bit; read by the true distance
alone it would graze the lower rim at exactly $d = "tol"$ and be turned by the rim's normal there. A puck set flush in a
floor: a particle sliding along the floor at the rest gap onto it rests on the plane of its top and keeps its velocity.
Two boxes stacked into one side, two boxes side by side in one floor and a box set flush in a floor are the same seam
with flat faces: as the face gap of the next box's coplanar face is rounding of either sign, without the rule the path
counted as behind that plane, crossed the next box's side plane and stopped dead; resting on it, it keeps its velocity
(the tests' numbers). A box's grown edges stay sharp: a path that clips an edge without coming within $"tol"$ of the box
in a straight line is still hit, with a face normal, and a step up between two such boxes of $"tol"\/8$ to $"tol"$ still
stops a particle; rounding them is a change of the box's distance, not of this rule. A known choice of the order below:
a particle on the last $"tol"\/8$ of a box's grown top, its centre at least $7\/8 "tol"$ beyond the real edge, rests on
the side plane and falls past the box, where the top held it before.

After the first response of a step every solid gets a stricter form of the rule. After a hit the particle stands at $d =
"tol"$ only up to rounding, within $"tol"\/64$ either side, and the response leaves an approach of a few rounding
units along every normal (@sec-contact-response). With the rule above such a start would count as outside, and the
closed form would hit the same solid again from two rounding-sized numbers. So `MoveParticle` passes the sweep whether a
response was made in the step and the largest bound $b$ of the approach the responses so far may have left, one number
per velocity, each computed from the velocity its response received. For every solid that does not turn, on every
surface of it, a start gap up to $"tol"\/16$ then counts as resting, and the solid is hit at $t = 0$ only when $"rate" <
-b$, a real approach; otherwise the path is free of it. Why every solid, with no list of the solids met: a sweep after a
response starts where that response was made, and a solid whose start gap is within the window was within $"tol" +
"tol"\/16$ there, so it was an active surface of that response (@sec-contact-response), which left its approach within
the response's bound, at most $b$. The touching rule reads that approach by the same expressions as the response, for
every shape: the normal of the solid's distance in world axes at the same point and pose (a sphere's and a cylinder's
too), the solid velocity $bold(u)$ at the particle, and the surface the sweep hit listed with the hit's own distance,
which is that same function there. So its rate is minus the response's own target, bit for bit, and no curvature enters:
the normal is read where the response read it, not where the particle slid to. A solid outside the window takes the
closed form. One exception, a hit and not a miss: a surface whose normal agrees with another's within $2^(-40)$ but not
bit for bit, in a change that uses a third direction, is missed by up to $2^(-40)$ of the change, which the sweep reads
as an approach and hits at 0. The bound is carried, not recomputed: its derivation (@sec-contact-response) proves it
from the velocity the response received, and a bound from the returned velocity can be far smaller than what is left,
for example after a slip just below the stick limit. What it costs: a free path may still approach at up to $b$, and $b$
grows with later responses. Each response $j$ requires or checks $b_j t_"left, j" <= "tol"\/64$ at its own time. Between
two responses the largest bound is one of an earlier response, whose time left was longer, so each stretch adds at most
$"tol"\/64$, and the at most four responses of a step add at most $"tol"\/16$. That bound, not $d >= 0$, is what is
proved: before the first response the plain rule lets a path from the rest gap sink by less than $"tol"$, so it ends at
$d >= 0$, on the surface at worst, and the touched stretches after a later response may then take it up to $"tol"\/16$
past that surface, which the settle lifts back. At the extreme of the requirement below, a non-committed plate thinner
than $"tol"\/8$ could have its middle passed, and one thinner than $"tol"\/16$ be passed through; neither has been
constructed nor proved impossible. With $b = 2^(-46) A$, $A$ the sum of the absolute axes of the speeds involved, that
is $A t_"left" <= 2^40 "tol"$, true whenever those speeds over one step stay within the largest coordinate. Nothing
skips a solid: a start farther than $"tol"\/16$ takes the closed form, a real approach is a hit, and a turning solid
keeps no touching rule. For the fixed flat solid with normal (0.6, 0, 0.8) through the origin and a particle from
(0.125, 0, 0.125) at (−2, 0, −2) m/s for 0.5 s (this paper's numbers), the hit position is $2.8 times 10^(-18)$ m
outside the rest gap and the response's approach is $-2^(-55)$ m/s: the solid is not hit again.

A sweep and a response must read a solid alike also where its bounding surfaces free the sweep. A point is *level with*
a bounding surface when its gap to it is in the window above, $[-"tol"\/8, "tol"\/16]$, the gap alone, whatever the
velocity; the particle passes a solid *at a bounding surface's level* when it is level with one that does not give the
solid's distance. Such a solid is no active surface of a response, at the start or at a hit on another solid, unless the
sweep hit it: its distance there is a rim whose normal leans across a seam, and the sweep is free of the solid while the
particle rests at that level. Not the bounding surface in its place: beside a lone solid it has nothing under it, and
would hold up a particle that should fall. And the sweep tries the bounding surfaces before the touching rule on the
distance, whatever the sign of the distance's gap, since after a response the touched rule would read an approach to
that rim at a start the bounding surface frees. A solid left out whose sweep does not rest meets the touching rule on
its distance: after a response it is hit at 0 for any approach beyond the bound $b$, one more hit, listed as the surface
hit; before any response the plain rule frees a path that sinks less than $"tol"$ into the distance's face, so such an
approach, at a rim's rest gap or a box's edge strip, gets no response and records no reaction in the step, a known
choice. At a start within rounding of the rest gap the sign of the computed gap decides: a distance a hair above $"tol"$
counts as outside, and the closed form hits it at once (the rim search within $"tol"\/1024$, at 0); at or below $"tol"$
the plain rule frees it. What is guaranteed either way is the position: from the rest gap a sink below $"tol"$ keeps $d
>= 0$ along the path (a sink of $0.999 "tol"$ ends $0.001 "tol"$ away), the path never enters the solid, and the settle
lifts the particle to $"tol"$, so the end check holds. Two drums of one radius stacked on one axis and a step that
starts $3\/16 "tol"$ above the seam at the wall's rest gap, $1.017 "tol"$ from the lower rim: the step keeps $(0, 0,
-1)$ m/s bit for bit, where a response on the rim turned it to $(0.181, 0, -0.966)$ m/s. A box set flush in a floor and
a step that starts $33\/32 "tol"$ beyond the seam at the floor's rest gap: the box's distance there is its side face's
plane continued above the top, $1.03 "tol"$ away while its edge is $1.43 "tol"$ away; the active surfaces are the
floor's alone, and the step keeps its velocity, where a response on that plane stopped it dead (the tests' numbers).

=== A turning solid <sec-turning-solid>

A turning solid's gap is not linear in time, and its faces do not keep their direction. Contact steps toward it
instead (citation needed). The gap cannot shrink faster than

$ "bound" = |bold(v) - bold(v)_"lin"| + |bold(omega)| times r_"max", $

where $r_"max"$ is the larger distance from the path to the solid's origin at the two ends of the sweep. In the
solid's own axes the particle moves at $bold(v) - bold(v)_"lin" - bold(omega) times (bold(x) - bold(o))$, whose
length is at most the bound, and a distance changes no faster than the point it is measured from. The particle's
distance to the moving origin is largest at one end of the sweep, because both move in straight lines. Starting at
$t = 0$, the first-order search repeats $t <- t + g(t)\/"bound"$. Each advance is safe: in that time the gap shrinks
by at most $g(t)$, so the path never passes through the solid between two samples. The proof of concept bracketed
the time by bisection over a stack of intervals; stepping forward needs neither. This first-order advance is the
floor of the search, and the whole of it for a shape whose distance is not convex.

Alone, the first-order advance is slow: it closes the gap by the share rate / bound of it, rate the speed at which
the gap really closes, so it needs about $ln(4 g\/"tol") times "bound"\/"rate"$ advances, and a face that turns onto
a particle at a small share of the bound runs out of them. When the solid's distance is convex in its own axes, the
search also uses how fast the gap is closing. That holds for every turning shape here: a flat solid's $p_z$, a
plate's $|p_z| - "half_thickness"$ or a committed plate's $s p_z - "half_thickness"$, a box's $max_k (|p_k| -
"half_size"_k)$ and a cylinder's true distance. At a sample $t$ with gap $g$, normal $bold(n)$ and the surface
velocity $bold(u)(bold(x)) = bold(v)_"lin" + bold(omega) times (bold(x) - bold(o))$, the gap rate is $dot(g) =
bold(n) dot (bold(v) - bold(u)(bold(x)))$, and the gap curvature bound

$ K = |bold(omega)| (2 |bold(v) - bold(v)_"lin"| + |bold(omega)| r_"max") $

bounds how fast that rate changes. Three facts make $g(t + s) >= g + dot(g) s - K s^2\/2$ for every $s$. First, the
distance is convex in the own point $bold(p)$, and the normal it reports, written in own axes, is a subgradient of
it, so $d(bold(p)(t + s)) >= d(bold(p)(t)) + bold(n)_"own" dot (bold(p)(t + s) - bold(p)(t))$: at a box's edge or
corner the reported face is one of the largest, on a box's middle plane $p_k = 0$ or a plate's middle plane the
reported side lies in $[-1, 1]$, the subgradients of $|dot|$ at 0, and at a cylinder's rim the normal is the gradient
of the true distance. Second, with $bold(omega)$ constant in the step, the own point's acceleration is $bold(p)'' =
A^T (-2 bold(omega) times (bold(v) - bold(v)_"lin") + bold(omega) times (bold(omega) times (bold(x) - bold(o))))$,
the terms papers call Coriolis and centripetal, of length at most $K$ while $|bold(x) - bold(o)| <= r_"max"$, which
holds over the whole sweep; no small-angle step is taken, so this holds up to a quarter turn and beyond. Third,
$bold(n)_"own" dot bold(p)' = dot(g)$. The second-order advance is the positive root of that lower model, written
so that nothing cancels:

$
  s = (2 g) / (sqrt(dot(g)^2 + 2 K g) - dot(g)) quad "when" dot(g) <= 0, quad
  s = (dot(g) + sqrt(dot(g)^2 + 2 K g)) / K quad "when" dot(g) > 0,
$ <eq:second-order-advance>

and $K = 0$ with $dot(g) >= 0$ is a gap that cannot shrink: the path is free. The search takes the larger of the two
advances; both are safe, so the larger is too. Both ends of $r_"max"$ count: a $K$ built from the start's distance
alone let an advance carry a particle 3.8 cm into a quarter-turn arm along which it moved outward, and one built from
the end's distance alone 1.8 mm into its end face. The domain box's distance, the smallest gap to its sides, is not
convex, and neither is a mesh's: one wall's model says nothing about the next, and a turning room searched with the
second-order advance passed a side on 27.5 % of measured paths. Such a shape keeps the first-order advance alone
(`HasConvexDistance`). The rounding margin is the first-order one: the distance is off by at most $"tol"\/128$, and
$dot(g)$, $K$ and the root carry relative rounding only, so the path stays farther than $"tol" - "tol"\/64$ before
the hit.

The search stops with a hit when $g(t) <= "tol"\/4$, and with no hit when $t >= "duration"$. More than 128 advances
(`kMaxAdvanceCount`) is "too close to call, hover near a turning solid", and the step is tried again shorter. With
the second-order advance this is rare: it remains for a path that hovers near the solid or grazes it, with almost no
motion along its normal (a graze is the same thing near the tangent point), where each advance is only about $sqrt(2
g\/K)$. A particle at rest $2^(-17)$ m above the top face of a box of half size 0.5 m turning at (0, 0, 1) rad/s, on its
turn axis, never gets closer, but needs 182 advances to reach the end of a 1 s sweep. On the generated paths of the
tests, 30,000 sweeps toward turning boxes, plates, flat solids, cylinders, a quarter-turn arm and a paddle, there was no
close call. A box that turns uses this search, and so do a flat solid or a plate that turns about an axis across its
normal and a cylinder whose axis tilts; a sphere, a cylinder that only spins about its own axis, and a flat solid or a
plate that only spins about its normal keep their closed forms (@sec-round-solids).

The touching rule does not apply to a turning solid. A particle resting on one, at $d = "tol"$, starts with $g(0)
= 0 <= "tol"\/4$ and is hit at $t = 0$ in every step; the response and the carry in the solid's axes
(@sec-contact-response) then take it with the solid for the rest of the step. This is intended: the face it rests
on is turning, so the plane argument of the touching rule does not hold, and carrying the particle in the solid's
axes is what keeps it on the face.

Nor does the argument that keeps a touched solid from being hit again. For a solid that does not turn, the sweep after a
response reads the rate the response targeted, bit for bit, at the same point and pose. Toward a turning solid the
surface velocity at the particle, $bold(v)_"lin" + bold(omega) times bold(r)$, changes as the particle slides, and the
normal turns, so a rate read where a sweep starts says nothing about the gap later; and the search hits the solid at 0
from any gap up to $"tol"\/4$, whatever the velocity. Swept again after a hit, a turning solid would be hit at 0 in
every sweep, and the fifth hit of the step would end in "too many hits". The carry in the solid's axes replaces that
argument: a hit on a turning solid ends the sweeps of the step, so the solid is never swept again from the moment it was
hit, and a hit at 0 just after a response, on a turning solid within $"tol" + "tol"\/16$, is the moment of that
response, which read the solid with the same distance at the same point and pose: no second response, one hit. In the
solid's own axes the solid stands still and its distance is convex, and the carry moves the particle on a straight line
there that does not approach along the normal the response read, so the distance stays at least its value at the hit for
the whole time left, exactly: no rate, no allowance and no curvature bound enter. What is not exact is rounding: the
turns into and out of those axes, below $"tol"\/64$, and the normal written in them, a few rounding units on a face;
beside a cylinder's rim the normal is read from a vector about $"tol"$ long, so it turns by the own point's rounding
over $"tol"$, about $10^(-7)$ rad at $"tol" = 2^(-30)$ m and coordinates near 1 m. Measured on 12,000 generated moves
next to turning boxes, flat solids, plates, cylinders and a paddle, the carry ended no nearer than $0.999998 "tol"$
before the settle (measured, not proved).

A worked example, from the tests: a box of half size 0.5 m at the origin turns at (0, 0, 1) rad/s, and a
particle rests at $0.6 (cos 0.3, sin 0.3, 0)$ m, at $d = 0.073$ m from face `kHighX`. In the box's axes the particle
goes round at radius 0.6 m, so the bound is 0.6 m/s and $K = 1 times (0 + 1 times 0.6) = 0.6 "m/s"^2$, and the face
reaches it when $0.6 cos(t - 0.3) = 0.5 + "tol"$, at $t = 0.3 + arccos((0.5 + "tol")\/0.6) approx 0.8857$ s. With
$"tol" = 2^(-30)$ m, the search takes 3 advances and stops at 0.88568554050660 s, $1.4 times 10^(-10)$ s before that
time, within the test's $2^(-30)$ (this paper's count, from the code's run). The first-order advance alone takes 30.

The solids that do not turn are searched first, then the turning solids, each only up to the earliest hit found so far:
a later hit could not win, and a turning search that reaches that time is free, since the path stayed at $d >= "tol"$ up
to it. So a close call beyond an earlier hit is never reached. Of two turning solids that fill the same space through
the step (same shape, size, pose and motion, bit for bit), only the first in array order is searched: the second, capped
at the first one's hit, would sample other times and stop elsewhere within the hit gap, so a copy would change the hit.
When several solids are hit, the earliest time wins, and equal times keep the lower solid index.

#procedure(
  [`FindFirstHit`, the first hit of a straight path],
  <proc-find-first-hit>,
  inputs: [The start $bold(x)_0$, `velocity_for_this_step` $bold(v)$, `start_elapsed` $t_0$ and `duration`, all
    finite; the solids, the contact tolerance ($>= 0$; 0 gives the plain surface) and the particle's plate side.
    The inputs are not checked here.],
  outputs: [`FirstHit`: the time since $t_0$, the index of the solid hit (`kNoSolid` when the path is free), and the
    distance, normal and face at the hit.],
  fails: [a search toward a turning solid needs more than 128 advances: "too close to call, hover near a
    turning solid".],
  ([Solid that does not turn], [First, for each solid in array order that does not turn (a box with angular velocity
    exactly zero, a flat solid or plate whose angular velocity has no part across its normal longer than $2^(-46)
    |bold(omega)|$, so one that at most spins about its normal, any sphere, a cylinder whose axis does not tilt),
    sampled at $t_0$: if the start's gap is in $[-"tol", 0]$ for the surface that gives $d$, the solid is hit at 0 when
    $"rate" times "duration" < -"tol"$ and not hit otherwise; if the gap is below $-"tol"$, it is hit at 0. Otherwise
    use the closed form of its shape: one face plane for a flat solid, a committed plate or each closed domain side;
    entry and exit times over the faces for a plate or a box; @proc-sphere-first-hit for a sphere;
    @proc-cylinder-first-hit for a cylinder. Keep the time if it is at most `duration`, with the face at the hit.]),
  ([Turning solid], [Then for each solid in array order that turns, but one that fills the same space as an earlier one,
    with `duration` capped at the earliest hit so far: from $t = 0$, repeat $t <- t + "advance"$ with $g$ measured at
    the pose of $t_0 + t$, the advance the larger of $g(t)\/"bound"$ and the positive root of $g + dot(g) s - K s^2\/2$
    (@eq:second-order-advance) when the distance is convex (every turning shape here), $g(t)\/"bound"$ otherwise. Stop
    with a hit at $t$ when $g(t) <= "tol"\/4$, with no hit when $t >= "duration"$, and fail after 128 advances. There is
    no touching rule here: a particle resting on the solid is hit at 0.]),
  ([Keep the earliest], [Return the hit with the smallest time; equal times keep the lower solid index, whatever the
    order of the search. With no hit, return `kNoSolid`.]),
)

== Round solids: sphere and cylinder <sec-round-solids>

A sphere and a finite cylinder are round, so the distance to them is not a face plane's and does not change linearly
along a path. Contact gives them their true distance to the surface, a closed form for the first hit of a path, and
three rules of their own: a rim search, a touching rule on a curved surface and a spin rule (@fig-round-solids).

=== Distance and normal

A sphere with centre $bold(c)$ and radius `radius` is at distance

$ d = |bold(x) - bold(c)| - "radius", quad bold(n) = (bold(x) - bold(c)) / (|bold(x) - bold(c)|) $ <eq:sphere-distance>

from a world point $bold(x)$, with the normal pointing from the centre to the point and face `kCurved`. At the
centre every direction gives the same distance; the code then takes the solid's own −x axis, the normal of `kLowX`,
the first face in face order: the same rule a box follows when faces tie. It is tested by exact equality,
$bold(p) dot bold(p) = 0$ as computed, so the CPU and the GPU take the same branch.

A cylinder is handled in its own axes. With $rho = sqrt(p_x^2 + p_y^2)$, $d_"wall"$ the signed distance to the
wall's surface (unbounded in height) and $d_"cap"$ the signed distance to the nearer cap's plane,

$
  d_"wall" = rho - "radius", quad d_"cap" = |p_z| - "half_height", \
  d = sqrt(max(d_"wall", 0)^2 + max(d_"cap", 0)^2) + min(max(d_"wall", d_"cap"), 0).
$ <eq:cylinder-distance>

The formula is written once for every region, so the CPU and the GPU compute it the same way; three branches only
choose the normal and the face. Write $bold(e)_rho = (p_x, p_y, 0)\/rho$, pointing away from the axis, and
$sigma_z = +1$ when $p_z >= 0$, else $-1$. In this order:

- *Rim*, when $d_"wall" > 0$ and $d_"cap" > 0$: the point is outside the wall and beyond a cap at once, $d =
  sqrt(d_"wall"^2 + d_"cap"^2)$ is its distance to the rim circle, the normal is $(d_"wall" bold(e)_rho + d_"cap"
  sigma_z bold(e)_z)\/d$, and the face is `kHighRim` when $sigma_z = +1$, else `kLowRim`.
- *Wall*, when $d_"wall" >= d_"cap"$: $d = d_"wall"$, normal $bold(e)_rho$, face `kCurved`.
- *Cap*, otherwise: $d = d_"cap"$, normal $sigma_z bold(e)_z$, face `kHighZ` or `kLowZ`.

Inside ($d_"wall", d_"cap" <= 0$) the nearer of wall and cap wins, and a tie goes to the wall. On a region boundary
both branches give the same $d$. On the axis, where $rho = 0$ as computed, $bold(e)_rho$ has no direction; in the
wall branch the code takes the own −x axis as for the sphere. The wall branch is reached on the axis only when
$|p_z| <= "half_height" - "radius"$, so only inside a cylinder at least as long as it is wide. For a cylinder of
radius 0.5 m and half height 0.25 m, the point (0.8, 0, 0.65) gives $d_"wall" = 0.3$ and $d_"cap" = 0.4$, so it
faces the rim at $d = 0.5$ with normal (0.6, 0, 0.8), face `kHighRim`; the point (0.1, 0, 0.2) gives $d_"wall" =
-0.4$ and $d_"cap" = -0.05$, so it is inside at $d = -0.05$ under the top cap, normal (0, 0, 1), face `kHighZ`.
These numbers are this paper's own.

#figure(
  round-solids-figure(),
  placement: auto,
  caption: [The sphere and a cut of the cylinder through its axis. The distance is the true distance to the
    surface; outside the cylinder it comes from the wall, a cap or a rim, by region. Where every direction is
    nearest, the sphere's centre and a point on the cylinder's axis, the normal is the solid's own −x axis.],
) <fig-round-solids>

=== First hit of a path <sec-round-first-hit>

The path meets the solid grown by $"tol"$. Here $bold(q)$ is the start relative to the solid's origin at time $t_0$
and $bold(w) = bold(v) - bold(v)_"lin"$ the velocity relative to the solid, both turned into the solid's own axes.

*Sphere.* The grown sphere has radius $R = "radius" + "tol"$, and the path meets it where $|bold(q) + bold(w) t| =
R$. With $alpha = bold(w) dot bold(w)$, $beta = bold(q) dot bold(w)$ and the gap at the start $g_0 = |bold(q)| -
"radius" - "tol"$,

$
  gamma = g_0 (|bold(q)| + R), quad t_"hit" = gamma / (-beta + sqrt(beta^2 - alpha gamma)).
$ <eq:sphere-hit-time>

$gamma$ equals $bold(q) dot bold(q) - R^2$, but computed from the gap its sign is exactly the sign of the gap that
the touching rule below reads, so the two can never disagree about a start on the grown sphere. The discriminant
$beta^2 - alpha gamma$ is computed in one of two forms, equal in exact arithmetic by Lagrange's identity $|bold(q)|^2
|bold(w)|^2 - (bold(q) dot bold(w))^2 = |bold(q) times bold(w)|^2$:

$
  Delta = beta^2 - alpha gamma quad "when" gamma <= R^2, quad
  Delta = alpha R^2 - |bold(q) times bold(w)|^2 quad "otherwise".
$ <eq:sphere-discriminant>

There is no hit when $alpha = 0$, when the path does not approach the centre ($beta >= 0$), or when the root does not
exist ($Delta < 0$); otherwise the hit is kept when $t_"hit" <= "duration"$, with $beta^2 - alpha gamma$ in
@eq:sphere-hit-time read as $Delta$. The root equals the textbook $(-beta - sqrt(Delta))\/alpha$, but it never divides
by a small $alpha$, and on a start near the grown sphere, $gamma$ small, it does not subtract two nearly equal numbers.
For a fixed sphere of radius 0.5 m at the origin and a particle from (−2, 0.3, 0) at (2, 0, 0) m/s (this paper's
numbers): $alpha = 4$, $beta = -4$, $gamma = 4.09 - R^2$, and $t_"hit" = 1 - sqrt(R^2 - 0.09)\/2$, which is $0.8 - 0.625
"tol"$ s. The hit point is (−0.4, 0.3, 0) less $1.25 "tol"$ along x, with normal (−0.8, 0.6, 0).

The discriminant needs the second form. Far from a small sphere, $beta^2$ and $alpha gamma$ are both about $|bold(q)|^2
|bold(w)|^2$ and each rounds by about $epsilon |bold(q)|^2 |bold(w)|^2$, while their difference, $|bold(w)|^2 (R^2 -
h^2)$ with $h$ the distance of the line from the centre, is at most $R^2 |bold(w)|^2$: the hit moved by about $epsilon
|bold(q)|^2 \/ R$. At the edge of the scale the tolerance protects, $|bold(q)| = 2^41.5 "tol"$ and a radius of 1000 tol,
the first form alone would stop a path through the centre 32,150 tol short of the sphere, let a path 1300 tol deep pass
through it with no hit, and hit a path 250 tol beside it (measured). In $alpha R^2 - |bold(q) times bold(w)|^2$ no term
is of size $|bold(q)|^2$: the cross product has length $h |bold(w)|$ and rounds by about $epsilon |bold(q)| |bold(w)|$
per axis, so $Delta$ rounds by about $5 epsilon alpha R |bold(q)|$. Near the sphere, $gamma <= R^2$, both forms have
terms of size $alpha R^2$, and $beta^2 - alpha gamma$ keeps the gap's digits for a start just outside the surface moving
nearly along it, where both its terms are small. For the computed coefficients, $alpha t^2 + 2 beta t + gamma$ at the
computed root is exactly $gamma (Delta' - Delta) \/ (-beta + sqrt(Delta'))^2$, at most $|Delta' - Delta| \/ alpha$, so
the discriminant's rounding moves the hit by at most about $3 epsilon |bold(q)|$ along the distance. Counted to first
order, with the rounding of the gap, of the other coefficients, of the own path and of the solid's pose at $t_0$ (about
$epsilon L$), the hit's true distance is within $14 epsilon |bold(q)| + 3 epsilon L$ of $"tol"$, and a path called free
comes no nearer than $"tol" - 7 epsilon |bold(q)| - epsilon L$. For a cylinder, whose turn into own axes rounds the
start and the velocity by about $2.6 epsilon$ each, read $|bold(q)|$ as the largest own coordinate along the path, not
$rho_0$. These are estimates summed term by term, not a proof. When every coordinate is within $2^40 "tol"$ of the
origin, as `MoveParticle` requires, $|bold(q)| <= 2 sqrt(3) times 2^40 "tol"$ and the total is about 0.013 tol, under
$"tol"\/64$. Measured against an independent double-double reference with radii up to the largest that premise admits,
about $2^39$ tol: at most 0.0024 tol over 8.7 million sphere and cylinder paths, and $3.1 epsilon |bold(q)|$ on fixed
spheres.

*Cylinder.* The cylinder is grown with sharp edges: radius $R = "radius" + "tol"$, half height $H =
"half_height" + "tol"$. The grown cylinder is the part shared by the grown wall, taken as unbounded along the axis,
and the space between the planes of the grown caps, so the path is inside it exactly while it is inside both. Each
of the two is entered and left at most once along a straight path, as a box's face plane is:

$
  "wall:" & quad [t_"wall", t'_"wall"] = [gamma_rho / (-beta_rho + sqrt(beta_rho^2 - alpha_rho gamma_rho)),
    (-beta_rho + sqrt(beta_rho^2 - alpha_rho gamma_rho)) / alpha_rho], \
  "caps:" & quad [t_"cap", t'_"cap"] = [(d_"cap" - "tol") / (-sigma_0 w_z), (|q_z| + H) / (-sigma_0 w_z)], \
  "hit:" & quad t_"hit" = max(t_"wall", t_"cap") quad "when" t_"hit" <= t'_"wall", t_"hit" <= t'_"cap"
    "and" t_"hit" <= "duration".
$ <eq:cylinder-hit-time>

The wall interval comes from the sphere's quadratic in the x and y parts only: $alpha_rho = w_x^2 + w_y^2$,
$beta_rho = q_x w_x + q_y w_y$ and $gamma_rho = (d_"wall" - "tol") (rho_0 + R)$, from the gap, with $rho_0 =
sqrt(q_x^2 + q_y^2)$ and $d_"wall" = rho_0 - "radius"$ as @eq:cylinder-distance computes it. When the start is beyond
the grown wall, $d_"wall" - "tol" > 0$, the interval is the one written above, its discriminant computed as
@eq:sphere-discriminant in the plane across the axis (the cross product's part along the axis, $q_x w_y - q_y w_x$), and
it is empty when the path does not move toward the axis ($alpha_rho = 0$ or $beta_rho >= 0$) or the root does not exist
(a discriminant below 0). Otherwise the path is inside the grown wall from $t = 0$ until the larger root, written
$-gamma_rho \/ (beta_rho + sqrt(beta_rho^2 - alpha_rho gamma_rho))$ when $beta_rho > 0$ so that nothing cancels, or for
ever when $alpha_rho = 0$. The caps' interval is the one written above when the start is beyond a grown cap, $d_"cap" -
"tol" > 0$ with $sigma_0$ the sign of $q_z$, and empty unless the path moves toward that cap ($sigma_0 w_z < 0$);
otherwise the path is between the grown caps from $t = 0$ until it reaches the grown cap it moves toward, or for ever
when $w_z = 0$. A start the distance puts beyond the grown wall or a cap is beyond it here, bit for bit, since both read
the same computed gaps. The hit is the later entry, provided it comes no later than either exit: the same test of entry
and exit as for a box. Both limits count, so a path that meets the sharp rim edge exactly is a hit. A path that goes
into the cylinder through the rim edge leaves both regions long after it enters them, so the rounding of the two entry
times can change which one is later but cannot make the path miss; only a path that grazes the edge sits on the limit,
where a hit and a miss are both safe. (Accepting each entry only where the path is inside the other region at that time
is the same in exact arithmetic, but there a path through the edge can have both entries rejected by rounding and pass
through the solid.) The face, normal and distance reported are those of @eq:cylinder-distance at the hit position.

On the wall or a cap proper that entry is the hit: there the sharp grown surface is the true one. Beside a rim, where
both $d_"wall"$ and $d_"cap"$ are positive at the entry, the sharp corner lies up to $sqrt(2) "tol"$ from the rim, so
the entry is where the *rim search* starts. Along the path the gap $g(t) = d(bold(q) + bold(w) t) - "tol"$, with $d$ the
true distance of @eq:cylinder-distance, is a convex function of time, since the distance to a convex solid is convex
along a line. At a sample $t$ the search reads $g$ and $dot(g) = bold(n) dot bold(w)$, $bold(n)$ the normal there, and
in this order: $g <= 0$ is a hit at $t$; $dot(g) >= 0$ is free, the gap never closes again; when $t + g\/(-dot(g))$ is
later than `duration` it is free, since the gap closing at its present rate, which only slows, does not reach 0 in the
sweep; $g <= "tol"\/1024$ is a hit at $t$; after 16 advances it is a hit where it stands, the safe side; otherwise $t$
advances by $g\/(-dot(g))$. Convexity gives $g(s) >= g + dot(g) (s - t) > 0$ for every $s$ before $t + g\/(-dot(g))$, so
no advance passes a hit, and a path that the sharp shape never contains is free, since the sharp shape contains the true
grown one. The advance is the second-order advance of a solid that does not turn, with a gap curvature bound of 0
(@sec-first-hit), written as the quotient: the square of a gap rate below $1.6 times 10^(-162)$ m/s would underflow to
an advance of 0. A rim hit is at $"tol" <= d <= "tol" + "tol"\/1024$ up to rounding; a path that passes the rim within
$"tol"\/1024$ of the rest gap is hit and loses at most $sqrt(2 times 2^(-10)) = 0.044$ of its speed. On a graze the gap
falls about fourfold per advance, so a path is decided within about 5 advances (2.0 on average over the model's paths).
No hit is ever rejected by a second test: the intervals decide that a path misses the sharp shape, and after the entry
the path is hit there or searched on, each advance proved free. `MoveParticle` uses the hit where the search stopped, as
every hit: nothing moves it. For a fixed cylinder of radius 0.5 m and half height 1 m and a particle from $(0.5 +
"tol"\/2, 0, 3)$ at (0, 0, −1) m/s (the tests' numbers), the path is inside the grown wall all along ($alpha_rho = 0$)
and enters between the grown caps at $2 - "tol"$ s, at $(0.5 + "tol"\/2, 0, 1 + "tol")$, where $d_"wall" = "tol"\/2$ and
$d_"cap" = "tol"$: beside the high rim, $sqrt(1.25) "tol"$ from it. Two advances bring the path to $(0.5 + "tol"\/2, 0,
1 + sqrt(0.75) "tol")$ at $2 - sqrt(0.75) "tol"$ s, $d = "tol"$ within $4.5 times 10^(-7) "tol"$, face `kHighRim`,
normal $(1\/2, 0, sqrt(3)\/2)$; the response leaves $(sqrt(3)\/4, 0, -1\/4)$ m/s.

The other examples of a cylinder of radius 0.5 m and half height 0.25 m at the origin (this paper's numbers): from
(0.2, 0, 1) at (0, 0, −2) m/s the top cap is hit at $(1 - 0.25 - "tol")\/2 = 0.375 - "tol"\/2$ s, at $(0.2, 0,
0.25 + "tol")$; from (−1.5, 0.3, 0) at (2, 0, 0.2) m/s the wall is hit at $0.55 - 0.625 "tol"$ s near (−0.4, 0.3,
0.11), with normal (−0.8, 0.6, 0). From the same start at (2, 0, 0.5) m/s the cylinder is not hit: the path leaves
the space between the grown caps at $0.5 + 2 "tol"$ s, before it enters the grown wall at about 0.55 s, when it is at
height 0.275 m. Seen as geometry, the path crosses the cap's height 0.25 m at $rho = 0.58$ m, outside the disc: it
passes over the rim.

#figure(
  round-first-hit-figure(),
  placement: auto,
  caption: [Paths against the grown shapes, the tolerance exaggerated. The sphere grows into a larger sphere; the
    cylinder grows with sharp edges. A path entering the sharp corner beside the rim is farther than $"tol"$ from
    it there; the rim search advances along the path, two advances here, to the red arc, where $d = "tol"$. The red
    arrow at the sphere's hit is the normal there.],
) <fig-round-first-hit>

=== Starts on or near a round solid <sec-round-starts>

The start's gap $g_0 = d - "tol"$ uses the true distance. A start deeper than $"tol"$ inside the grown solid, $g_0
< -"tol"$, is hit at $t = 0$, as for a flat face. With $g_0$ in $[-"tol", 0]$ the touching rule applies, with $"rate" =
bold(n) dot (bold(v) - bold(u))$, $bold(n)$ the normal at the start in world axes and $bold(u)$ the surface's velocity
there, the very numbers a response reads (a spin about the centre or the axis adds nothing along $bold(n)$ but rounding;
a cylinder's tilt within the bound below adds its true surface speed): the solid is not hit during this sweep when
$"rate" times "duration" >= -"tol"$, and it is hit at $t = 0$ otherwise. It is safe because the true signed distance to
a convex solid is a convex function of position, so along the straight path $d(t) >= d(0) + "rate" times t >= d(0) -
"tol"$. `MoveParticle` starts every sweep at $d = "tol"$ (a hit or the start move put it there, a rim hit within
$"tol"\/1024$ of it), so $d(t) >= "tol" - "tol" = 0$; the part of the window below 0 only absorbs rounding, and the
settle at the end of `MoveParticle` puts the particle back at $"tol"$. It is the same tolerance as for a flat face,
with no term for the curvature. A cylinder start outside the touching window and beside a rim is read against its
bounding surfaces first (@sec-first-hit): resting on the wall's unbounded surface or on the cap's plane, it is free.
Otherwise a start inside the grown sharp cylinder with a positive gap $g_0$ lies in a rim corner, both distances in
$(0, "tol"]$; its entry is 0, and the rim search decides. For a sphere of radius 0.5 m at the origin, a particle at
$(-(0.5 + "tol"), 0, 0)$ moving at (0, 1, 0) m/s, along the tangent, is free; moving at (1, 0, 0) m/s it is hit at 0
(the tests' numbers).

#procedure(
  [The first hit of a straight path on a sphere],
  <proc-sphere-first-hit>,
  inputs: [The start and the velocity, in world axes for the touching rule, and relative to the sphere in its own
    axes for the root; `radius`, the contact tolerance and `duration`.],
  outputs: [The hit time since the start of the sweep, or no hit. The normal and face come from
    @eq:sphere-distance at the hit position.],
  fails: [never.],
  ([Read the start], [Compute the distance and the normal in world axes and the gap $g_0$. Below $-"tol"$: hit at 0.
    In $[-"tol", 0]$: the touching rule with the rate from the surface's velocity, hit at 0 when $"rate" times
    "duration" < -"tol"$, no hit otherwise.]),
  ([Find the root], [Compute the three coefficients of @eq:sphere-hit-time, the constant from the gap, and the
    discriminant by @eq:sphere-discriminant. No hit when the velocity is zero, the path does not approach the centre
    or the root does not exist.]),
  ([Keep the hit], [Compute the root without cancellation; it is the hit when it is at most `duration`.]),
)

#procedure(
  [The first hit of a straight path on a cylinder],
  <proc-cylinder-first-hit>,
  inputs: [The start and the velocity relative to the cylinder, in its own axes; `radius`, `half_height`, the
    contact tolerance and `duration`.],
  outputs: [The hit time since the start of the sweep, or no hit. The normal and face come from
    @eq:cylinder-distance at the hit position, a rim included.],
  fails: [never.],
  ([Read the start], [Compute the true distance at the start, in world axes, and its gap $g_0$.]),
  ([Bounding surfaces, then the touching rule], [For a start beside a rim, read the wall, unbounded along the axis,
    and the cap's plane on the start's side: no hit when the start rests on either, a gap in $[-"tol"\/8,
    "tol"\/16]$ sinking at most $"tol"\/64$ during the sweep, whatever the distance's gap. Then $g_0$ below $-"tol"$:
    hit at 0; in $[-"tol", 0]$: the touching rule.]),
  ([Find the entry], [Compute the times the path is inside the grown wall, unbounded along the axis, with
    @eq:cylinder-hit-time, the constant from the gap: from the first root to the second when the start is beyond the
    grown wall ($d_"wall" - "tol" > 0$), empty when it does not approach the axis or the root does not exist; from 0
    to the larger root, or for ever, otherwise. Compute the times it is between the grown caps' planes: from
    $(d_"cap" - "tol") \/ (-sigma_0 w_z)$ to the far grown cap when the start is beyond a grown cap, empty when it
    does not move toward that cap; from 0 to the grown cap it moves toward, or for ever, otherwise. The entry is the
    later of the two entries, when neither interval is empty and it comes no later than either exit and no later than
    `duration`; no hit otherwise. A start in a rim corner has the entry 0.]),
  ([Keep the hit], [An entry on the wall or a cap proper, as @eq:cylinder-distance reads the point there, is the
    hit. Beside a rim, the rim search from the entry: advances of $g\/(-dot(g))$ on the true distance until $g <=
    "tol"\/1024$ (a hit), $dot(g) >= 0$ or the end of the sweep (no hit), at most 16.]),
)

=== Slide and spin <sec-slide-and-spin>

After a hit the response leaves a velocity along the tangent plane at the hit. The sweep goes on along that tangent
in a straight line. On a sphere or a cylinder wall the path leaves the surface, its distance growing as
$("speed" times "time")^2\/(2 "radius")$; on a cap it stays on the plane, as on a flat face. By the convexity above,
a straight path from the surface cannot enter the same solid again in this step, so a round solid is hit at most
once per sweep, and sliding round a sphere does not run into the limit of four hits. Gravity or the pressure brings
the particle back to the surface in later steps, and over many steps it follows the curve; a particle sliding from
the top of a frictionless sphere leaves it where $cos theta = 2\/3$, $theta$ measured from the top.

A turn about a sphere's centre, or about a cylinder's own axis, moves the surface but not the shape; so does a turn of a
flat solid or a plate about its normal, since both are unbounded sideways (@sec-first-hit). Such a solid is swept and
slid as one that does not turn, with the closed forms above, and the spin enters only through the solid velocity
$bold(u)$, which always uses the full angular velocity: in the response, and in the rate the touching rule reads, so the
two agree bit for bit (@sec-first-hit). The turn that changes the shape is none for a sphere, and for a cylinder the
part of $bold(omega)$ across its axis, $bold(omega)_"across" = bold(omega) - (bold(omega) dot bold(e)_z) bold(e)_z$,
with $bold(e)_z$ the cylinder's own z axis (its `axis`) at the start of the sweep. A cylinder is a turning solid only
when $|bold(omega)_"across"| T |"half_size"| > "tol"\/8$, when its shape moves by more than $"tol"\/8$ during the step;
it then uses the stepping search of @sec-first-hit and the carry in its own axes of @sec-contact-response, and the
quarter-turn limit applies to $|bold(omega)_"across"| T$. A spin given along the axis stays below that bound up to
rounding. Below the bound the closed form keeps the axis of the sweep's start, while $bold(u)$ keeps the tilt, and the
touching rule reads the surface's normal speed where the sweep starts. A particle that slides across the tilting surface
keeps that speed while the surface under its new position moves the other way, so the gap falls by up to $2
|bold(omega)_"across"| max|bold(r)| T < "tol"\/4$, since $max|bold(r)| <= |"half_size"|$; a sweep the closed form
decides keeps the $"tol"\/8$. So the true grown shape is entered by up to $"tol"\/4$ without a hit, outside the
$"tol"\/16$ of @sec-first-hit; no hit into the solid is missed, the particle ends at least $"tol" - "tol"\/16 -
"tol"\/4 = 0.6875 "tol"$ from it, above the end check's $"tol"\/2$, and the settle lifts it back (measured: 0.759 tol
before the settle, for a cylinder of radius 0.1 m and half height 2 m at 0.99 of the bound, the tests' numbers).
Measured from $bold(v)_"lin"$ alone, a cap sinking at $3 times 10^(-10)$ m/s under a particle it had just stopped looked
approached, beyond any rounding bound, and was hit again at 0 until "too many hits", at every step length. Why not treat
a spinning roller as turning: the carry on a turning solid moves the particle in the solid's axes, so it would drag a
particle round the roller even with friction 0. Through the solid velocity, the spin acts by friction only. For a
cylinder of radius 0.5 m at the origin spinning at (0, 0, 10) rad/s, a particle hitting the wall at (−0.5, 0, 0) with
velocity (1, 0, 0) m/s meets a surface moving at (0, −5, 0) m/s: the approach speed is −1 m/s, and the particle keeps
(0, 0, 0) m/s with friction 0 and takes (0, −0.5, 0) m/s with friction 0.5 (this paper's numbers, by
@eq:contact-response).

== Response and slide <sec-contact-response>

=== The response at one surface

`ComputeContactResponse` changes one velocity $bold(v)$ at a surface with normal $bold(n)$ and solid velocity
$bold(u)$. A solid blocks motion into it and rubs against motion along it (papers call this Coulomb friction). The
particle's speed toward the surface, measured from the moving solid, is the approach speed

$ a = bold(n) dot (bold(v) - bold(u)), $

negative when the particle approaches. Then

$
  bold(v)' = bold(v) - a bold(n), quad s_"side" = |bold(v)' - bold(u)|, \
  bold(v)'' = cases(
    bold(v) & "if" a >= 0,
    bold(v)' & "if" a < 0 "and" mu = 0,
    bold(u) & "if" a < 0 "and" mu >= s_"side"\/(-a),
    bold(v)' + (mu (-a)) / s_"side" (bold(u) - bold(v)') & "otherwise",
  )
$ <eq:contact-response>

A velocity that leaves or slides along the surface comes back unchanged, and a velocity with no sideways speed left
sticks. The friction $mu$ is the particle's `friction` times the solid's `friction_scale`. In words: the speed into the
solid is removed; then friction takes $mu$ times that speed off the sideways speed $s_"side"$, measured from the solid.
When that would take all of it or more, the particle sticks and takes the solid's velocity; otherwise it slips with less
sideways speed. This is the friction response of the snow paper by Stomakhin et al. [19], with the sideways speed
measured from the solid. The removal is done once, the approach recomputed from the new velocity, and the rest removed
again. The result is exact only up to rounding: its approach, computed again, is zero within a few rounding units of the
speeds involved and can be slightly negative. With machine epsilon $epsilon = 2^(-52)$, each operation rounds by at
most $epsilon\/2$ of its result, and a dot product with the unit normal by at most $2 epsilon$ times the length of
its other vector. Let $S = |bold(v)| + |bold(u)|$. After the first removal, $bold(v)' - bold(u)$ is the sideways part
of $bold(v) - bold(u)$, so $|bold(v)' - bold(u)| <= S$ and $|bold(v)'| <= 2 S$. The second removal recomputes the
approach to within $2 epsilon S$ and its subtraction moves the approach by at most $epsilon\/2 dot 2 S = epsilon S$,
so at most $3 epsilon S$ stays. A slip keeps $1 - mu (-a)\/s_"side" <= 1$ times that and rounds its blend by at most
$2 epsilon S$ more; a stick returns $bold(u)$, whose approach computes to exactly 0. Computing the approach of the
result adds at most $2 epsilon S$: in all at most $7 epsilon S$. The code bounds it by $2^(-46) A = 64 epsilon A$
(`ComputeLargestApproachLeftAfterResponse`), with $A$ the sum of the absolute axes of $bold(v)$ and $bold(u)$. $A >=
S$, and unlike a length it squares nothing, so the bound stays right for speeds whose squares underflow; it holds for
$A$ between $2^(-1000)$ and $2^500$ m/s. A later test of whether such a velocity still approaches compares with that
bound, not with 0. For a fixed flat solid with normal (0.6, 0, 0.8), a velocity (−2, 0, −2) m/s with friction 0 keeps
an approach of $-2^(-55)$ m/s, within the bound of $2^(-44) = 5.7 times 10^(-14)$ m/s; the same velocity times
$2^(-830)$ keeps $-2^(-885)$ m/s, within $2^(-874)$ m/s, where a bound built from lengths would be 0.

For a fixed flat solid with normal (0, 0, 1) and an incoming velocity of (3, 0, −1) m/s (@fig-contact-response, the
tests' numbers): $a = -1$ m/s, so $bold(v)' = (3, 0, 0)$ and the sideways speed is 3 m/s. With $mu = 0.5$,
friction takes 0.5 m/s and the particle slips on at (2.5, 0, 0) m/s. With $mu = 3$ or more, friction would take
all 3 m/s: the particle sticks at (0, 0, 0). A head-on velocity (0, 0, −2) m/s at a solid moving at (0, 0, 0.5)
m/s becomes (0, 0, 0.5) m/s.

#figure(
  contact-response-figure(),
  placement: auto,
  caption: [One velocity at a fixed flat solid: the approach part is removed, and friction takes friction × 1 m/s
    off the 3 m/s sideways part. With friction 0.5 the particle slips; with friction 3 nothing is left and it
    sticks.],
) <fig-contact-response>

Two choices in @eq:contact-response matter for the code. First, the approach is removed in world coordinates, as
$bold(v) - a bold(n)$, not from the velocity measured from the solid, $bold(v) - bold(u)$, with $bold(u)$ added back
afterwards. That round trip would erase a small sideways velocity next to a fast solid. With friction 0 at a flat solid
with normal (0, 0, 1) sliding sideways at $10^8$ m/s, a particle at $(10^(-9), 0, -1)$ m/s keeps its x part of $10^(-9)$
m/s in world coordinates, while the round trip computes $(10^(-9) - 10^8) + 10^8 = 0$, since doubles near $10^8$ are
$1.5 times 10^(-8)$ apart (this paper's example). Second, the stick test compares $mu$ with the quotient, $mu >=
s_"side"\/(-a)$, not the product with the sideways speed, $mu (-a) >= s_"side"$. The two forms round differently at the
limit: with sideways speed 3 m/s and approach −0.7 m/s, the friction $3\/0.7$ rounded to a double sticks by the
quotient, while the product rounds to $2.9999999999999996 < 3$ and would slip at $4.4 times 10^(-16)$ m/s. Overflow is
no reason for either form: an infinite product still compares correctly.

Each of the particle's two velocities gets the response with its own approach speed. A particle whose
`for_this_step` is (0, 0, −1) m/s and whose `for_next_step` is (1, 0, −3) m/s meets a fixed flat solid with
friction 0.5: `for_next_step` has approach −3 m/s and sideways speed 1 m/s, and $0.5 >= 1\/3$, so it sticks at
zero; `for_this_step` has approach −1 m/s and no sideways speed, so it too becomes zero. The record keeps both
approach speeds, −3 and −1 m/s.

=== Moving one particle <sec-move-particle>

`MoveParticle` moves one particle for one step in five steps (@proc-move-particle): check the inputs, classify the
start, find the first hit, apply the joint response, and settle and check the end. Without any hit the end position is
$bold(x)_0 + bold(v) T$, one multiplication and one addition per axis, and both velocities come back unchanged.

The start is classified against every solid at the start of the step, in three parts. First the checks that fail,
every solid in array order at the start as given, and the first solid that fails decides. For a committed plate,
being more than $"tol"$ beyond its other face is checked first. Then a start deeper than `start_overlap_tolerance`
inside a solid is an error. The depth is read before any move, so a move cannot carry a start that is too deep out
of the solid and hide it.

Then the start moves once off every surface closer than $"tol"$, of every solid together, before the sweep and
whatever its velocity, to a point at $d >= "tol"$ from each, so a particle placed next to solids, by a source or by
a change of cell size, never fails the end check for that reason. With one near solid the move is along its normal
to $d = "tol"$; in a corner of the domain box each closed side closer than $"tol"$ is a surface of its own, as in the
sweep. With several, one move per solid in turn would not do: in a wedge sharper than a right angle each solid's move
pushes the start toward the other (@fig-move-off-near-surfaces, left), and a start resting on both faces failed at
every step length. The move is the smallest change $bold(w)$ from the start $bold(x)_0$ with $d_i + bold(n)_i dot
bold(w) >= "tol"$ for each near surface $i$, its *shortfall* $"tol" - d_i$ met along its normal. That problem is
convex, so its answer meets at most three surfaces as equalities: the solver tries each surface alone, each pair
and each triple, solves their equations by Cramer's rule, and keeps the shortest change that meets every surface.
Two flat faces meeting at a wedge angle $a$ (180 degrees minus the angle between their normals) give a move of
$"tol" \/ sin(a\/2)$ along the bisector from a start on both: 2 tol at 60 degrees.

A round surface's normal turns as the point moves, so the move is solved again from each iterate, $bold(x)_(k+1) =
bold(x)_0 + bold(w)$ with the distances and normals read at $bold(x)_k$: $d_i (bold(x)_k) + bold(n)_i dot (bold(x)_0
+ bold(w) - bold(x)_k) >= "tol"$. Every shape's distance is convex (@sec-round-solids), so $d_i (bold(y)) >= d_i
(bold(x)) + bold(n)_i (bold(x)) dot (bold(y) - bold(x))$: each iterate the solver finds is at the rest gap of every
surface it read, up to the solver's slack, and since it then meets its own equations, the next one is no farther from
$bold(x)_0$. This holds from the first iterate that was not shortened (below): a deep start can end beyond the rest
point, still at the rest gap of both surfaces. The slack includes the rounding of composing $bold(w)$, which grows with
the multipliers, so no iterate is trusted: the last one is kept only after its true distances are read. Under a ball of
radius $R$ resting on a floor, a *tangent contact*, the free space is a *cusp* whose angle shrinks to 0 at the contact.
The first iterate from deep in the cusp overshoots out along the floor, where the two tangent planes meet; the next ones
come back to the rest point $2 sqrt(R "tol")$ from the contact, both distances exactly $"tol"$
(@fig-move-off-near-surfaces, right): 5 iterates at $R \/ "tol" = 2^29$, within $10^(-4)$ of it after the largest
number, 8, at $R \/ "tol" = 2^40$. The iterates stop when one moves less than $"tol" \/ 64$ and is within $"tol" \/
1024$ of the rest gap of every solid; a surface that an iterate comes within $"tol"$ of joins the next one. The segment
from $bold(x)_0$ to each iterate is swept against every other surface, so a long move cannot pass through a thin solid:
a move that would cross one stops where it first comes within $"tol"$ of it, and that surface joins the next iterate,
linearised there. Surfaces with the same normal share one equation, the most demanding. The particle is displaced
without a matching velocity, so each change is shortened to at most $2^22 "tol" = h \/ 256 + L \/ 2^18$ (a 256th of a
cell plus 4 micrometres in a one-metre scene), which covers the rest point of every tangent contact, $2 sqrt(R "tol") <=
2^21 "tol"$. The last iterate is kept when every solid is at least $"tol" - "tol" \/ 128$ from it by its true distances.
A start with no such point within that length is the scene error "starts in narrow gap": two solids facing each other
less than $2 "tol"$ apart, a start within rounding of the contact point of a ball on a floor, a pocket too small for a
point at $"tol"$ from each side, or a start deep at the apex of a flat wedge sharper than about $2^(-21)$ rad. The grown
solids overlap there, so a sweep stops a moving particle before it, and a shorter step does not help. The same error
names a start near surfaces facing more than four different ways, which one move cannot read at once; surfaces with the
same normal, such as a ground lying on the domain's bottom or floors stacked closer than $"tol"$, share one target, the
most demanding, up to 16 surfaces in all. A known limit stands beside the start near the contact point: a solid within
$"ulp"(R) \/ (2 sqrt("tol" \/ R))$ of a cusp's rest point (0.0014 tol for a ball of 0.5 m, 128 tol for 1 km), where the
ball's rounded distance moves the linearised vertex past the wall, so no subset of the targets has an answer although
the rest point fits. The failure names the near solid latest in array order.

#figure(
  move-off-near-surfaces-figure(),
  placement: top,
  caption: [The start's move off several near surfaces. Left: at the apex of a 60 degree wedge, moving up off the
    floor takes the start into the other face, and moving off that face takes it back to 0.25 tol from the floor
    (faint); one move reaches the point at $"tol"$ from both. Right: the cusp under a ball of radius 12 tol resting
    on the floor, to scale. Each iterate is measured from the start $x_0$; the first overshoots along the floor,
    and the next come back to the rest point.],
) <fig-move-off-near-surfaces>

Then the start has a response when it was inside a solid as given, $d < 0$, or when either velocity approaches an
active surface at the moved start (@sec-contact-response); "approaches" there means an approach speed below 0, or
below $2^(-46) |bold(omega) times bold(r)|_1$, the rounding of a spinning solid's surface velocity: a floor spinning
about its normal at 100 m/s at the particle has no velocity along its normal, but computed it has about $10^(-14)$
m/s, which would be a hit. The response is one joint response over the active surfaces read at the moved start, not
at the start as given: the sweep that follows reads the normals there, and the response must leave no approach along
those. It is one hit, however many surfaces; the kind is "started inside solid" when the start was inside a solid,
else "hit". Otherwise the move is silent: velocities, kind, hit count and record are unchanged, and a touching
particle that leaves is free. A surface near the start as given but farther than $"tol" + "tol"\/16$ after the move
takes no part: of two parallel floors $"tol"\/4$ apart, only the upper one, which the particle rests on, responds and
is charged.

The record is that of the first surface, in array order of the solids and then face order, among the active surfaces
the start was inside or a velocity approaches, with hit fraction 0. A flat solid with surface at $z = 0$ rising at
2 m/s, under a particle at rest at $z = "tol"$ for 0.1 s: the particle approaches the solid at −2 m/s, so it is a hit
at fraction 0, both velocities become (0, 0, 2) m/s, and the particle ends carried at $z = 0.2 + "tol"$. The same
particle at $(0, 0, "tol"\/4)$ moving at (1, 0, 2) m/s, as fast as the floor along its normal (an approach speed of
exactly 0, which is no approach), is moved to $z = "tol"$ silently and ends at $(0.1, 0, 0.2 + "tol")$.

No hit is moved before its response: a rim hit is where the rim search found the true distance within
$"tol"\/1024$ of $"tol"$ (@sec-round-solids).

At each hit, at the start or along the path, both velocities get one *joint response* over the *active surfaces*: every
surface of every solid that the particle is within $"tol" + "tol"\/16$ of, read at its position then (not the normal
recorded at a hit, stale once the particle has slid along a ball; not a surface farther away, such as a ceiling hit 0.1
m above a floor touched earlier; not a solid passed at a bounding surface's level, @sec-first-hit), and the surface hit,
with the hit's own normal. A surface the same bit for bit as one listed, in normal, solid velocity, friction and
`reports_reaction`, is dropped: an exact copy adds nothing, so 17 or 40 coincident floors take one place. The rest are
sorted by value (normal, then solid velocity, then friction, then a reacting solid first, then solid index), so every
sum and every tie below reads them in an order their values decide, and the velocities and the reaction are the same
bits for every order of the solids array. The move off near surfaces at the start and the settle are not sorted: they
read the solids in array order and count an exact copy as a solid of its own, so a reordered array or a copy can change
the position by a unit in the last place (9 and 2 of 26,216 moves near round solids, at most $4.4 times 10^(-16)$ m), a
known limit; sorting their targets would change the rounding of every move whose array order is not the sorted one. A
coincident surface that is not an exact copy (rougher, smoother, or reacting) stays; with a moving solid a smoother one
changes the result by rounding only, at most $2.2 times 10^(-14)$ of the speed (measured). Surfaces whose normals agree
within $2^(-40)$ on every axis share one *direction*, with the normal of the first.

One active surface: each velocity gets the response above, nothing else. Several: for a velocity $bold(v)$, each
direction's target is the largest $-bold(n)_i dot (bold(v) - bold(u)_i)$ of its surfaces, $bold(u)_i$ the solid's
velocity at the particle. When no target is above the velocity's bound so far, $bold(v)$ keeps its bits. Otherwise the
solver of the start's move gives the smallest change $bold(w) = sum_k lambda_k bold(n)_k$ that meets every target,
over all directions (up to eight): the smallest change meets at most three as equalities, so the subsets of at most
three are tried, 92 at most. $lambda_k >= 0$ is the *push* of direction $k$, and $bold(v)_0 = bold(v) + bold(w)$ the
frictionless velocity. Two fixed surfaces keep the part along their common line, so a particle in a V-groove slides
on along it, and one pushed into a cusp or an acute wedge stops; three stop it; moving solids give the velocity that
approaches none, which can be faster than either: a floor rising at $V$ under a fixed face at 60 degrees carries the
particle out along the apex at $(V \/ tan 60 degree, 0, V)$. In a wedge of 120 degrees, friction 0, $(-1, 1, -1)$
gives $(0, 1, 0)$; responses one surface after the other would give $(-0.25, 1, 0.433)$ or $(0.183, 1, 0)$ by which
was listed first.

Each direction acts through its *pushing surface*: among its surfaces whose target is within $2^(-40)$ of the
speeds of the direction's (they push), the one with the largest friction $mu$, then one that reports its reaction,
then the first in sorted order. So a coincident copy as rough or smoother changes nothing, and a rougher one acts:
a ground with twice the friction lying on a floor acts for it. Its *friction limit* is $L_k = mu_k lambda_k$, the
most sideways speed its friction can take in this response. Friction then gives a change $bold(F)$:

- one used direction with friction: the response above with its own normal, solid velocity and friction, so a hit on
  one surface beside others that are not approached is the single-surface response, bit for bit (without friction it
  is $bold(v)_0$, the same up to rounding);
- several, with one solid velocity $bold(u)$ (fixed solids, or the faces of one moving solid): the surfaces act as
  one. With $bold(s) = bold(v)_0 - bold(u)$ and $L = sum_k L_k$, the velocity is $bold(u)$ when $L >= |bold(s)|$
  (stick), else $bold(v)_0 + (L \/ |bold(s)|) (bold(u) - bold(v)_0)$ (slip). This is the exact answer of friction
  with these pushes, and a wedge amplifies friction, as it should: at the rest point of a 30 degree wedge, $(-1, 1,
  -1)$ makes pushes 2.732 and 2, so friction 0.5 stops the particle;
- several, moving differently: the mean, over every order of the used directions (two or six), of their own frictions
  applied one after the other from $bold(v)_0$, each taking $min(L_k, "sideways speed")$ off the sideways velocity
  relative to its own solid at that moment. The mean does not depend on any order; when the solid velocities differ only
  by rounding it agrees with the case above to $1.7 times 10^(-14)$ of the speed. It approximates the exact answer of
  the same friction model: median 0.046, p90 0.19, p99 0.37 and at worst 0.66 of the largest speed over 2,418 draws with
  friction up to 2 and solids up to 1 m/s apart. Adding each surface's own friction from the same $bold(v)_0$ instead
  reverses the sliding when two solid velocities are one rounding unit apart. A known limit of the approximation: each
  step stops at sticking to its own solid at that moment, but a later step of the other surface moves the velocity on
  and the mean keeps both, so a surface's own friction can push the particle past sticking to its solid, reversing the
  sliding relative to it: measured in 1,331 of 24,401 generated responses that used the mean (5.5 %), by up to 0.22 and
  at p99 0.072 of the summed speeds. The exact answer for solids that move differently is left to the owner.

The result is $bold(v)_0 + bold(F)$ when it approaches no active direction beyond its bound. Otherwise, and always when
the solids move differently, a second frictionless change is made from $bold(v) + bold(F)$, not from $bold(v)_0 +
bold(F)$: the pushes are found afresh, so friction lifts a particle off a surface only when it overcomes what pressed it
there. Beside a floor, an upright rising at 2 m/s and a particle pushed into both at 1 m/s each: friction 0.5 leaves (0,
0, 0), friction 1.5 leaves (0, 0, 0.5); from $bold(v)_0 + bold(F)$ friction 0.5 would lift it at 0.5 m/s. Where normals
depend on each other (three faces sharing one line, coincident faces) the smallest change has several sets of pushes,
and the friction, which reads them, would jump with the rounding that picked one (up to 0.70 of the speed for a
$10^(-13)$ change of the velocity). So among the changes no longer than the shortest plus its bound the solver keeps the
one with the largest $sum_k mu_k lambda_k$, the largest value of a linear function over the admissible pushes, which
varies continuously with the velocity. And the solver's "no change" reads the bound so far, not its own: a real approach
of $10^(-13)$ m/s beside a floor sliding at 100 m/s is removed.

The solve rounds: one pair leaves about $2^(-52) \/ (1 + c)$ of its targets, with $c$ the dot of the two normals, and
up to four more solves on the residual bring it within the bound, so a pair is used for a velocity only when $1 + c
>= 2^(-40)$, a wedge of at least $1.35 times 10^(-6)$ rad. Every cusp a particle rests in is wider: at the rest point
its angle is about $2 sqrt("tol" \/ R) >= 2^(-19)$. Below the limit, a *sliver* (a plank lying on a floor with a
rounding-level tilt), or solids closing on the particle, give no change, and the particle sticks to the active
surface it approaches fastest: it takes that solid's velocity, and a surface the new velocity still approaches beyond
the bound is "too close to call, squeezed between solids": a ceiling coming down on a floor. A frictionless velocity
faster than 64 times the largest speed involved (both velocities before the response and the active solids'
velocities) is a squeeze too: a wedge closing at speed $V$ pushes the particle out at $V \/ sin a$, unbounded as it
flattens, so one face closing on a fixed one squeezes below $"asin"(1\/64) = 0.8953$ degrees and two faces closing
symmetrically below $2 "asin"(1\/64) = 1.7906$ degrees, both closed forms, since the cap reads the speeds before any
friction. The geometry and the speeds decide, so at a hit at the start of the step a shorter step does not help,
and the step module decides what a lasting squeeze does; at a later hit a step that ends before it does, until the
solids reach the particle at the start of a later step. A squeeze names the solid the failing velocity approaches
fastest. The response's bound,
$2^(-46) (|bold(v)|_1 + |bold(w)|_1 + sum_i |bold(u)_i|_1)$ for each change made, raises the move's bound of that
velocity; with several active surfaces, a bound whose product with the time left exceeds $"tol" \/ 64$ is a squeeze
too, too fast for the step, which a shorter step cures.

Every change of `velocity.for_next_step` made against a solid with `reports_reaction` is added to the particle's
reaction, with the position of the first such change: with one active surface the whole change; with several, each
used direction's push $lambda_k bold(n)_k$ and its share of $bold(F)$ to its pushing surface's solid (after a second
change, its pushes and the first's friction shares), and a stick's whole change to the solid stuck to. A
frictionless surface pushes only along its normal, so each solid receives its own push. Only `for_next_step` counts,
because it is the velocity the particle keeps; `for_this_step` only moves it during this step. The solids module
turns these changes, times each particle's mass, into the force on the solid. That force then comes from the very
velocity changes that stopped the particles, so the same geometry decides both, as Batty, Bertails and Bridson argue
for coupling liquids and solids [21]. A particle pushes at most one reacting solid per step; a second, different one
is "too close to call, second reacting solid", and within one response the charges go in ascending solid index, so it
names the higher. A particle in the crease of two reacting solids that pushes into both fails at every step length.
A change of exactly zero adds nothing, so it neither fills the record nor conflicts: a solid's count of hits counts
only the particles whose `for_next_step` changed at it.

On a solid that does not turn, the sweep starts again from the hit position with the new `for_this_step` for the time
left. A fifth hit in one step (`kMaxHitsPerStep` is 4) is "too close to call, too many hits", and so is a hit with
more than eight directions or 16 surfaces active; the start alone, one hit, never is. Where the rule is not
continuous: at the edge of the window, a surface $"tol" + "tol"\/16$ away is met together with the others or an
instant later, which no rule removes but which rounding cannot cross, since a particle at rest sits $"tol"\/16$ inside
it. Known limits: a fast particle falling into the neck of two touching balls meets one ball, then the other, as
separate hits, and zig-zags; the step needed shrinks in proportion to $sqrt(R "tol") \/ "speed"$ (a 0.1 s step at
2 m/s needed up to 1/256 of it in the tests); four moving faces within a few $"tol"$ of each other but not at one
rest point are met one after the other, and 13 of 283,929 such corners fail at every step length.

After a hit on a turning solid the particle is carried in the solid's own axes for the rest of the step (the carry), as
the proof of concept did. A sphere, a cylinder that only spins about its own axis, and a flat solid or a plate that only
spins about its normal, are not turning solids here (@sec-round-solids, @sec-first-hit). After the response, the
relative `for_this_step` at the hit, $bold(v) - bold(u)(bold(x))$ with $bold(u)$ the surface velocity at the hit
position and pose, is turned into those axes with the solid's axes at the middle of the time left, $bold(v)_"local" =
A_"mid"^T (bold(v) - bold(u)(bold(x)))$; its part into the face, along the normal of the hit written in own axes, is set
to zero; the particle advances in those axes for the time left $t_"left"$, $bold(p) + bold(v)_"local" t_"left"$, a
straight line there; and it is placed with the solid's pose at the end of the step. Both velocities stay as the response
made them. Why the middle axes: the solid carries the surface's motion at the particle exactly (the own point stays),
and the relative motion runs along a line in world axes while the solid turns under it. Turned with the axes of the hit,
a frictionless particle at rest on a turntable drifted out by $r a^2\/2$ a step, $a = |bold(omega)| t_"left"$ the turn
(6.4 % of $r$ over 50 steps of 0.1 s at 0.5 rad/s); taking the world line for the whole velocity moved that drift to a
stuck particle, and added energy. With the middle axes, the midpoint rule of the turn, a stuck particle keeps its own
point exactly, and a frictionless one at rest on a face turning about its normal ends at radius $r (1 + a^4\/48 +
dots)$, $r a^3\/24$ behind its start: third order a step, though not always negligible, +2 % of $r$ a step at $a = 1$
rad, near the largest turn a step allows. The path's mean velocity agrees with the velocity the response returned to
first order in $a$. Turned with the middle axes, a relative velocity tangent at the hit may point into the face by up to
$sin(a\/2)$ of itself when the face turns toward that line; the clamp keeps that part out. When the face turns away from
the line, that part points out and is kept, a known choice: the particle ends up to $sin(a\/2) |bold(v) - bold(u)|
t_"left"$ off the face, as a free flight in the solid's axes, while the velocity the response returned has no part along
the normal (a top face tilting at 0.5 rad/s under a sliding particle: up to 41,617 $"tol"$ at $T = 1\/120$ s, $5.3 times
10^(-3)$ m at 0.1 s). It does not build up from step to step, since the next sweep and gravity bring the particle back,
and the path converges at first order in the step. A particle that leaves the face is carried too, a known choice:
the stepping search hits it at 0 whatever its velocity, its record has kind "hit" with a positive approach speed, and
the rest of its free flight is a straight line in the solid's axes, bent in world axes (leaving at 3 m/s a face
turning at 2 rad/s, 0.2 m from the axis: +2.6 m/s recorded, and the end $3 times 10^(-6)$ m from the straight flight
after 1 ms). The proof of @sec-first-hit is unchanged: the carry is still a
straight line in own axes with no part along the hit's normal into the solid. The cost is one more pose. The hit loop
ends there; why no later sweep may meet the solid again is in @sec-first-hit. Before that, every other solid $j$ is
tested: the call is "too close to call, turning solid near another solid", naming the turning solid, when

$ d_j - "tol" <= t_"left" (|bold(v)_"local"| + |bold(v)_"lin"| + |bold(omega)| r + |bold(v)_("lin", j)| +
|bold(omega)_j| r_j), $

with $d_j$ the distance to solid $j$ at the hit, $bold(v)_"lin"$, $bold(omega)$ the turning solid's motion,
$bold(v)_("lin", j)$, $bold(omega)_j$ solid $j$'s, and $r$, $r_j$ the distances from the hit position to each solid's
origin. The right side bounds how far the particle and solid $j$ can close in $t_"left"$. The carry puts the particle at
$bold(x)(s) = bold(o) + bold(v)_"lin" s + R(s) A (bold(p) + bold(v)_"local" s)$, $R(s)$ the turn by $bold(omega) s$, so
$bold(x)(s) - bold(x)(0) = bold(v)_"lin" s + (R(s) - I) A bold(p) + R(s) A bold(v)_"local" s$; a turn by an angle $a$
moves a vector by at most $2 sin(a\/2) <= a$ times its length, so the particle moves by at most $s (|bold(v)_"lin"| +
|bold(omega)| r + |bold(v)_"local"|)$. In solid $j$'s axes the same argument adds $s (|bold(v)_("lin", j)| +
|bold(omega)_j| r_j)$, and a distance changes no faster than the point it is read at. The bound is on the displacement,
not on the length of the curved path, so no term in $t_"left"^2$ is needed. A solid that fills the same space as the
turning solid through the step is left out (the carry keeps the particle out of the one, so out of the other), and with
no time left nothing moves and nothing is tested. A box of half size 0.5 m turning at (0, 0, 1) rad/s on a fixed floor
that touches its bottom, and a particle at rest on its side 0.05 m above the floor: hit at 0, $|bold(v)_"local"| =
(0.5 + "tol") cos(t_"left"\/2)$ and $r = 0.6727$ m, so the test fails for $t_"left" >= 0.0426$ s: a step of 0.1 s is too
close to call, one of 0.04 s carries. Known limit: two flush boxes turning together cannot be crossed at their seam,
since a turning solid has no bounding surface and the carry sweeps nothing; within $"tol"$ of both, every step length is
too close to call. So is a particle resting on the floor against a paddle that turns through it, or in the crease of a
paddle and its shaft. The failure says so: `can_shorter_step_cure` is false when the carry starts at the start of the
step and the other solid is within the window $"tol" + "tol"\/16$, and true otherwise, since the bound shrinks with
$t_"left"$.

At the end the particle moves once more off every surface it is closer to than $"tol"$, at the end pose, as at the
start: every solid, touched or not, since a move off the touched ones alone could push it into another (an upright
beside a ball), and since a sweep may have brought it within $"tol"$ of a solid it never touched. When there is no such
point the particle stays. Then the end check of @sec-contact-tolerance runs against every solid. On any failure the
result is left unchanged, and the failure context names the solid by its stable `solid_id`, never by its index in the
array, with a length: the depth $-d$ for a start too deep inside, the distance $-(s_"plate" p_z + "half_thickness")$
beyond a committed plate, the end distance $d$ for an end too close, and 0 otherwise. `MoveParticles` adds the
particle's id.

@fig-sweep-hit-slide draws one step (this paper's numbers). A flat solid with normal (0, 0, 1) rises at 0.5 m/s,
friction 0.25, $T = 1$ s. The particle starts 1.2 m above the surface with `for_this_step` (2.5, 0, −1.5) m/s and
`for_next_step` (1.5, 0, −2.5) m/s (the y parts are 0 and left out of the figure). The gap $1.2 - "tol"$ closes at 2
m/s, so the hit is at $0.6 - "tol"\/2$ s, at $x approx 1.5$ m. `for_this_step` approaches at −2 m/s with sideways speed
2.5 m/s: friction takes 0.5 m/s and leaves (2, 0, 0.5) m/s. `for_next_step` approaches at −3 m/s with sideways speed 1.5
m/s: friction takes 0.75 m/s and leaves (0.75, 0, 0.5) m/s. The sweep starts again with (2, 0, 0.5) m/s, which moves
exactly as fast as the surface along the normal: the resting surface is not hit again, and the particle slides for 0.4 s
to $x = 2.3$ m, $"tol"$ above the surface at its end height of 0.5 m. If the solid reports its reaction, the change of
`for_next_step` is (−0.75, 0, 3) m/s.

#figure(
  sweep-hit-slide-figure(),
  placement: top,
  caption: [One step against a rising flat solid. The sweep stops one tolerance above the surface as it is at the
    moment of the hit; each velocity gets its own response; the slide rides up with the surface. The tolerance is
    drawn as 0.12 m so it shows, which moves the drawn hit to 0.54 s.],
) <fig-sweep-hit-slide>

#procedure(
  [`MoveParticle`, one particle for one step],
  <proc-move-particle>,
  breakable: true,
  inputs: [`start_position`, both velocities, the solids in ascending `solid_id`, `step_duration`, `friction`,
    `start_overlap_tolerance`, `contact_tolerance` and the particle's `plate_side`.],
  outputs: [The end position, both velocities after the response, the kind, the record of the first hit, the hit
    count and the reaction.],
  fails: [a bad input; a start too deep inside a solid, on the wrong side of its plate, or in a narrow gap; too close
    to call, for one of the seven reasons named in the text. The result is then unchanged, and the failure names the
    solid by its `solid_id` (never its index) with a length.],
  ([Check inputs], [Every value finite, `step_duration` $> 0$, `friction` $>= 0$, `contact_tolerance` $> 0$,
    `start_overlap_tolerance` $>= 0$, solids in strictly ascending `solid_id`, and, for a committed particle, a
    `plate_side` that names a plate among the solids. Otherwise fail with "bad input".]),
  ([Classify the start], [For each solid in array order, at the start as given: fail when the particle is more than
    $"tol"$ beyond the other face of its committed plate, then when $d < -$`start_overlap_tolerance`. Move the start
    once to the rest gap of every surface closer than $"tol"$, of every solid (each closed side of a domain box on its
    own): the smallest change meeting every shortfall, solved again from each iterate, at most $2^22 "tol"$; fail
    with "starts in narrow gap" when there is none. When the start as given was inside a solid, or either velocity
    approaches an active surface at the moved start (below 0, or the rounding of a spinning surface), apply one joint
    response at time 0 as in step 4, with the record of the first such surface in array order; the kind is "started
    inside solid" when the start was inside.]),
  ([Find the first hit], [@proc-find-first-hit from the current position along `for_this_step` for the time left,
    with the solids sampled at the time already used and, after the first response of the step, the stricter
    touching rule for every solid. With no hit, advance by `for_this_step` times the time left and go to step 5.]),
  ([Joint response], [Fail on a fifth hit. Move to the hit, nothing more. Read the active surfaces: those of every solid
    within $"tol" + "tol"\/16$ of the particle, except a solid passed at a bounding surface's level, normals read
    afresh, and the surface hit with its own normal; drop exact copies, sort by value, group into directions; fail with
    "too many hits" past eight directions or 16 surfaces. Fill the record if it is empty and count one hit. For each
    velocity: with one active surface, the response with that solid's velocity and `friction` times `friction_scale`;
    with several, when it approaches one beyond its bound, the smallest frictionless change, then the friction from each
    used direction's push (the single-surface response for one, the stick or slip toward one solid velocity, or the mean
    over the orders), then, when the result approaches an active surface or the solids move differently, a second
    frictionless change from the velocity plus the friction. With no change, stick to the surface approached fastest.
    Fail with "squeezed" when a stick still approaches a surface, a frictionless velocity is faster than 64 times the
    largest speed involved, or, with several surfaces, the bound times the time left exceeds $"tol"\/64$. Charge each
    push and friction share of `for_next_step` to its solid, in ascending solid index, when it reports its reaction;
    fail on a second reacting solid. Raise each velocity's approach bound. A solid that does not turn: go to step 3 for
    the time left. A turning solid (never a sphere, nor a cylinder that only spins about its axis, nor a flat solid or
    plate that only spins about its normal): skip the response when the hit is at 0 just after a response and within
    $"tol" + "tol"\/16$ (the same moment, read already); fail if another solid $j$ is within the closing bound of the
    text, carry the rest of the step in its axes, and go to step 5.]),
  ([Settle and check the end], [At the end pose, move the particle once to the rest gap of every surface of every
    solid closer than $"tol"$, as at the start; when there is no such point, leave it. Fail when any solid is at
    $d < "tol"\/2$; otherwise write the result.]),
)

= The grid <sec-grid>

The grid is the fixed scaffolding the simulation computes on. Space is cut into cubic cells: the pressure lives at
cell centres, and each part of the velocity lives on the cell faces it crosses. Cells may have different sizes, each
size half the one before, and only the parts of space that hold material are kept, in cell blocks of 4 × 4 × 4
cells. The grid also answers the one geometric question a particle asks it: which faces are near this position, and
how much weight does each get?

The grid never looks at physics. It owns no physical field: velocity, mass, pressure and the open area of a face
belong to the modules that compute them, which keep them in value buffers laid out by the grid (@sec-cell-blocks).
Which faces a particle may use is the caller's choice, passed as a mask (@sec-nearby-faces). Every material uses
the same faces and the same weights: liquids, snow, springy solids and gas markers, as Stomakhin et al. [18] put
varied materials on one staggered grid. Materials that do not mix are separate material sets, each with its own
values on the same faces.

== Where values live

Each cell carries one pressure, at its centre. Each face carries one part of the velocity: the part along the axis
the face is normal to. A face normal to X carries the x part, a face normal to Y the y part and a face normal to Z
the z part (@fig-staggered-grid). Papers call this a staggered grid, or the marker-and-cell grid of Harlow and Welch
(1965). The flow through a face, which the pressure solve balances, is then read where it is kept, with no average
of neighbouring values, and the pressure difference of two cells acts on the face between them.

Every face has one fixed orientation: its normal is +axis. The cell on its minus side is its `cell_low`, the cell on
its plus side its `cell_high`. A value kept on a face is the velocity part along +axis, so a positive value is
flow out of `cell_low` into `cell_high`, whatever the code that reads it.

#figure(
  staggered-grid-figure(),
  placement: auto,
  caption: [Pressure at the cell centre, each velocity part on the faces normal to its axis. In the slice (X to the
    right, Z up) every face carries its value along +axis, so the cell on its low side and the cell on its high
    side are fixed by the face alone.],
) <fig-staggered-grid>

== Cells of mixed sizes <sec-cell-sizes>

A cell has a size level $l$: 0 for the biggest cells, then 1, 2 and so on, at most 16 size levels in all
(`kMaxSizeLevels`). Its width is

$ h_l = h_0 / 2^l, $

where $h_0$ is the `biggest_cell_width`. A grid's frame (`GridFrame`) gives $h_0$, the world position of the low corner
of the cell with coordinate (0, 0, 0), and how many size levels the grid uses, from 1 to 16; the last of them holds the
smallest cells. The origin is finite, $h_0$ is finite and positive, and the smallest width $w_"min" = h_0 \/ 2^(L - 1)$,
for $L$ size levels, is a normal double, at least $2^(-1022)$: every halving down to it is then exact. A cell is named
by its size level and its integer coordinate (`CellCoord`), counted in cells of its own size level: the cell of size
level $l$ and coordinate $(i, j, k)$ covers $i h_l <= x < (i + 1) h_l$ from the frame's origin, and the same along Y and
Z. Each coordinate lies between $-(2^18 - 1)$ and $2^18 - 1$ (`kMaxCoord`), at every size level; a cell beyond is
`kCoordOutOfRange`. Cells of different size levels are aligned by construction: the cell $(i, j, k)$ of size level $l$
covers exactly the eight cells $(2i + a, 2j + b, 2k + c)$ of size level $l + 1$, with $a$, $b$ and $c$ each 0 or 1. Its
cell eighth, $a + 2b + 4c$, says which of the eight a small cell is. Wang, Sun and Zhu [6] solve the pressure on a grid
of this shape, which papers call an octree.

A grid is described by one list: its active cells, the cells that exist and are not split, in any order. They carry
the values. A big cell is either active or split into all eight of its small cells, its sibling cells, each of them
active or split in turn; a lone small cell without its siblings is `kSiblingCellMissing`. A cell listed twice, or a
cell listed with a bigger cell that holds it, is `kCellsOverlap`. The active cells therefore fill the space they
cover once, with no gap inside a split cell.

*Splitting near the coordinate limits.* A split cell needs its eight small cells in range, so a cell can be split only
when each of its coordinates lies between $-(2^17 - 1)$ and $2^17 - 1$, at every size level. Two consequences follow.
A cell of size level 1 or more at $-$`kMaxCoord` lies in a big cell at $-2^17$, whose small cells reach $-2^18$: no
valid list holds it, and the check reports `kSiblingCellMissing` without naming that small cell, since no id can. And a
cell of size level $l$ at $c$ holds cells of size level $m$ only when one of its cells of size level #box[$m - 1$], from
#box[$c dot 2^(m - 1 - l)$] to #box[$(c + 1) dot 2^(m - 1 - l) - 1$], can be split. For example, a cell of size level 0
holds cells
of size level 15 only between $-8$ and $7$ on each axis, and at $-8$ not at its low corner, whose cell of size level 14
lies at $-2^17$. A scene that needs small cells farther out moves the frame's origin or uses larger biggest cells.

*The size-step rule.* Two active cells that touch, by a face, an edge or a corner, differ by at most one size level.
A grid that breaks it is `kSizeStepRuleBroken`, which names both cells. One size level is the rule for two reasons.
The side of a big cell then meets at most four smaller cells, its quarters, so a face is one face piece or four
(@sec-face-pieces). And the faces near a particle come from at most three size levels, and their number has a proven
ceiling, so the list of one particle lives in a fixed array on either processor (@sec-nearby-faces). A grid that
needs a faster change of size closes it with cells of the size levels between.

Which active cell holds a position is one formula. Per axis, the coordinate in the smallest cells is
$floor((x - "origin") \/ w_"min")$, and the search walks down from the cell block of size level 0 through the
split cells to the active one (`FindCellAt`). A cell owns its low sides and not its high sides. A position that is
not finite, or whose coordinate lies outside the range a size-level-0 id can name, is in no cell, before any integer
conversion. Since this is the only definition of "inside", a particle one rounding step from a cell side is placed
in the same cell by every module.

The difference and the quotient are each rounded once, to within $2^(-53)$ of their value relatively, so the coordinate
never decreases as $x$ grows, and a position farther than about $2^(-52) |x - "origin"|$ from every side lies in its
exact cell. A nearer one may land in the cell on either side, and even the side as computed may land below it: the error
grows with the distance from the origin, not with $|x|$. Where $c w_"min"$ and the side $"origin" + c w_"min"$ are both
doubles, as with origin 0 and $h_0 = 1$ m, a position on the side gives exactly $c$ and lies in the cell above; with
origin 0 and $h_0$ a power of two nothing else rounds. In every frame a negative quotient of at most $2^(-1075)$ in
size, a position within $2^(-1075) w_"min"$ below the origin, rounds to $-0$ and lies in cell 0, not $-1$. No second
test recovers the exact side. There the side is no double, so nothing exact exists to compare with, and a second test
would be a second definition of "inside", which the window quarters of @sec-nearby-faces, taken from the same quotient,
would not share. A search that starts from a block near the position, the particle's block at the step before, finds the
same cell: it climbs to that block's block of size level 0, makes at most two moves through the neighbour tables
(@sec-cell-blocks) and otherwise searches, so it only saves the binary search.

== Face pieces <sec-face-pieces>

Between cells of two sizes, a side is not one face. The +X side of a big cell beside four small cells is four face
pieces, each the whole −X side of one small cell and one quarter of the big side (@fig-face-pieces). The grid keeps
face pieces, never whole sides, with three rules:

+ *One identity.* A face piece is named by its `FacePieceId`: its axis, its size level and the coordinate of the cell
  place on its high side at that size level. The id does not depend on which other cells exist.
+ *One area.* A face piece is kept at the size level of its smaller cell and has that cell's face area, $h_l^2$.
+ *Kept once.* Every face piece exists exactly once, in exactly one cell block. A value on it therefore has exactly
  one writer, and the flow through a big side is the sum over its face pieces (`FindFacePieces` lists the up to 24
  face pieces of a cell's six sides).

Each cell block (@sec-cell-blocks) has, for each axis, five planes of 4 × 4 face places at its own size level: plane
0 is the low side of its first layer of cells along that axis, plane 4 the high side of its last. A face place is a
face piece when the two cell places beside it are:

/ active and active: `kBetweenCells`, two cells of one size;
/ active and inside a bigger cell: `kBetweenCells`, the small cell's whole side and one quarter of the big cell's
  side;
/ active and absent: `kGridEdgeLow` when the active cell is on the high side, `kGridEdgeHigh` when it is on the low
  side. What lies beyond is not the grid's to say: the pressure solve and the solids decide what such a face means.

Every other face place is `kNotAPiece`. A place between an active cell and a split one is not a face piece: the four
face pieces of that side live one size level down, at the size level of the small cells. So a face piece is always
kept at the size level of its smaller cell.

Plane 4 of a cell block and plane 0 of the block of the same size level on its high side are the same places in space.
When that block exists, the face piece is kept there, in its plane 0, and plane 4 of the first block is `kNotAPiece`.
Plane 4 holds a face piece only where no block of the same size level lies on the high side. This rule makes "kept
once" hold across blocks.

*Geometry of a face piece.* A face piece of size level $l$ along an axis, whose high cell place has the coordinate
$c$, lies on the plane $c h_l$ from the origin along that axis; its centre is at $(c' + 1\/2) h_l$ on each other axis,
for that axis's coordinate $c'$, and its area is $h_l^2$. Its `cell_low` and `cell_high` are the cell places beside it
at size level $l$, or, where such a place lies inside a bigger cell, that bigger active cell one size level up. The
distance between their centres along the axis, which a pressure difference across the face piece is divided by, is
half the width of each: $(h_"low" + h_"high") \/ 2$. For the four face pieces of a big cell's side it is
$h_(l - 1) \/ 2 + h_l \/ 2 = 3\/4 h_(l - 1)$, measured along the axis from the big cell's centre, whatever the
quarter. At the grid's edge only one cell is there, and the distance is half its width. In a grid piece, a
`kOutsidePiece` whose other cell lies in a block not loaded gives only half the width of the loaded cell: the grid
piece's tables cannot know whether the missing cell is of the same size level or one bigger, so that number is no
distance between two cells.

*The face pieces of one cell.* `FindFacePieces` lists the face pieces of an active cell's six sides, in ascending index
order. On each side it reads the face place of the cell's own size level that the side lies on; plane 4 is plane 0 of
the block of the same size level on the high side, when that block exists. When that place holds a face piece, the side
is that one face piece: beside a cell of the same size, beside a bigger cell, or at the grid's edge. Otherwise the place
across the side is split, and the side is the four face pieces of its small cells that face the cell, found where they
are kept, plane 4 included. By the size-step rule a side is one face piece or four, so a cell has at most
$6 times 4 = 24$ face pieces (`kMaxCellFacePieces`).

*The cells that touch one cell.* `FindTouchingCells` lists the touching cells of an active cell: the active cells that
share a side, an edge or a corner with it, in ascending index order. It reads the 26 places of the cell's size level
around it. An active place is one touching cell. A place inside a bigger cell, or with no block of its size level, gives
the active cell one size level bigger that covers it, listed once though it may cover up to four of the 26 places. A
split place gives its small cells next to the cell: four across a side, two across an edge, one across a corner. The
size-step rule makes them active and leaves no smaller cell there. Every cell that touches the cell is found this way. A
cell one size level bigger is aligned to pairs of places on each axis, so one that touches the cell covers at least one
of the 26 places. A cell one size level smaller lies inside one of them. The size-step rule leaves no other size. No
place gives more cells than a split place would, so a cell touches at most $6 times 4 + 12 times 2 + 8 times 1 = 56$
cells (`kMaxTouchingCells`), reached when all 26 places are split.

Both bounds hold whatever the tables say, so the two lists fit fixed arrays on either processor even for tables that
contradict themselves. A place that holds no active cell, an index of no place and the tables of no grid give empty
lists. For example, a big cell whose only other cells are the eight small cells of the split cell on its +X side
touches the four small cells beside that side and has nine face pieces: its five other sides whole and the four
quarters of that side.

*The cells around an edge.* `FindCellsAroundEdge` gives the cells around one of the 12 edges of an active cell, for the
viscosity solve. The edge is parallel to the third axis of two axes the caller names, on the side of each that a
quarter number picks. The cell is the first of four places; the others are the places of its size level across the
first axis, across the second, and across both. A place is given only when it holds an active cell of the cell's
size. A flag says whether every cell around the edge has that size: a split place, or a place inside a bigger active
cell, clears it; an absent place, at the grid's edge, does not. Where no block of the cell's size level lies, the
place is inside a bigger cell when the bigger block's place there is active, and absent otherwise. Two equal axes, an
axis that is not x, y or z, a quarter outside 0..3, a place without an active cell, an index of no place and the
tables of no grid give the empty result.

#figure(
  face-pieces-figure(),
  placement: auto,
  caption: [Left: the side of a big cell that four small cells touch is four face pieces, each with its own value.
    Middle: the same in a vertical slice, where two of the four show; each face piece is kept with its small
    cell. Right:
    cells two size levels apart may not touch.],
) <fig-face-pieces>

== Cell blocks <sec-cell-blocks>

Only the parts of space that hold material are kept, in cell blocks: 4 × 4 × 4 cell places of one size level, as Wang
et al. [5] and Fei, Huang and Gao [15] keep sparse grids on the GPU. A block exists when at least one of its places
holds an active or a split cell. Blocks therefore exist at every size level above a small cell, about 14 % more blocks
than the active cells alone would need ($1\/8 + 1\/64 + dots.h = 1\/7$) when the smallest cells fill a large, compact
region; a layer one block thick needs about a third more ($1\/4 + 1\/16 + dots.h = 1\/3$). Each place has a state
(`CellState`): active; split; inside a bigger cell, covered by an active cell one size level bigger; or absent.

A block holds 64 cell places, 240 face places (3 axes × 5 planes × 16) and 125 corner places (the 5 × 5 × 5 lattice
of its corners). An index counts places across blocks: the cell place $(x, y, z)$ of block $b$ is
$64 b + (4z + y) times 4 + x$, the face places of block $b$ start at $240 b$, axis by axis, and the corner place
$(x, y, z)$ of block $b$ is $125 b + (5z + y) times 5 + x$.

*The fixed order.* Blocks are listed in one order everywhere: by the low corner, measured in widths of a size-level-15
cell, by $z$, then $y$, then $x$, and then the bigger cell (the smaller size level) first. Cells are ordered the same
way. With Z up, a run of consecutive blocks is a horizontal layer of the grid. The order depends only on the cells,
never on the order of the input list or of the work, so every table built from it is the same on every run.

*Neighbour tables.* For each block the grid keeps its 27 neighbours of the same size level (ordered by their offset in
z, then in y, then in x, with "none" where no block is), its bigger block (the block of the next bigger cells that
covers it, size level $l - 1$; none at size level 0) and its 8 smaller blocks, by cell eighth. These tables and the
sorted list of block ids are the grid's whole spatial index: a search is a binary search in the block ids followed by
moves through the tables.

*Id versus index.* An id (`CellId`, `CellBlockId`, `FacePieceId`) names a place and a size in space; the same cell has
the same id in every grid version, and the id 0 is never valid. An index (`CellPlaceIndex`, `FacePlaceIndex`,
`CornerPlaceIndex`, `CellBlockIndex`) is a position in the arrays of one grid version: it depends on which blocks
exist, and after the cells change, the same cell may sit at another index. A value kept across a change is found again
by its id (`FindCellPlace`, `FindFacePlace`).

*Grid versions.* Each time the set of active cells changes, the new grid gets the next version number. Version 0 means
"bound to no grid"; a built grid has version 1 or more. Values carry the version of the grid they were laid out for,
and a function that takes values and a grid checks that the two agree.

*Buffers built once.* A grid version is built without allocating. `BuildGridBuffers` allocates once, for a room of
`max_cell_blocks` blocks, every array a grid version needs; `BuildGrid` fills them with the tables of one version
(@proc-build-grid), and the buffers then belong to that grid. `TakeBuffers` gives them back for the next version. The
temporary arrays a build needs, the builder buffers (`GridBuilderBuffers`), belong to the caller, who lends one set to
several grid buffers on one processor; no operation on a built grid reads them. The room comes from the run's plan,
and a grid that outgrows it is `kTooManyCellBlocks`, after which the run raises its limits. Value buffers
(`CellValues`, `FaceValues`, `CornerValues`, `CellBlockValues`: one value per cell, face or corner place, or per block)
are built once with room for the same number of blocks. `ApplyGridVersion` binds them to a new version: it records
the version and the block count, and their contents become unspecified until the owning module fills them again.
A function of another module that takes values and a grid compares the values' version and block count with the grid's
and fails with `kWrongGridVersion` when they differ. A grid piece has the whole grid's version, so values loaded for one
piece are bound again at every load. `ReadValues` and `WriteValues` are the only copies between value buffers and CPU
memory: they take the bound part only.

*Ranges and border.* A stage works on a range: a run of consecutive blocks in the fixed order (`CellBlockRange`). It
writes only values kept in blocks of the range, and it may read values of the range and of its border blocks. Since
every face piece is kept in exactly one block, every value has one writer. A run in batches loads only the range and its
border into a grid piece, where every neighbour entry naming a block not loaded reads as none. The border is defined by
that piece: the border of a set of blocks at a width is the smallest set of blocks outside it such that every reader of
that width, run from any place kept in the set (an active cell, a face piece or a corner place) on the tables of a piece
holding exactly the set and its border, gives the answer it gives on the whole grid. So the border holds every block a
reader reads, on the way or as a result, not only the blocks of the places it returns.

At the width `kTouchingCells` a stage may call, from an active cell of the set, `FindTouchingCells`, `FindFacePieces`
of the cell and of each cell touching it, `FindCornerPlaces`, `FindCellsAroundEdge`, `FindCellAt` for a position in
the cell's closed box, and `FindNearbyFaces` for a position that `FindCellAt` gives to the cell (a cell owns its low
sides, so a position on its high side belongs to the cell beyond, whose nearby faces may lie one step further); from
a face piece, its two cells (`FindCellLow`, `FindCellHigh`), its corners, and, for `ParticlesToGrid`, which adds
into each face piece from the particles of the cells that can list it, the cells of `FindCellsNearFace` at most one
size level bigger than the face piece: a cell two size levels bigger never lists it, since `FindNearbyFaces` keeps
face pieces within one size level of the particle's cell; from a corner place, its position. The width
`kFourCellWidths` adds, from a face piece, every cell near it (@sec-face-search) and every block that
`BuildFaceSearch` lists for it. That list depends on the block of the start and not on which of its face pieces is
the start, and `FindNextFace` reads only the blocks of the list, so a radius below 4 reads fewer blocks and is
covered. A stage calls these readers from places kept in the set only, never a lookup by id; from a border place it
needs a stacked layer.

The border at `kTouchingCells` holds what a face piece of the range needs to weigh a particle, given the particle's
sums (`NearbyFaceSums`); it does not hold what is needed to make the sums. A particle's list holds face pieces of
the cells that touch its own cell, and those can lie beyond the near border of another range. Example: the cells
$[0, 1)$ and $[1, 2)$ are active, and $[-1, 0)$ and $[2, 3)$ are split. The range of the block of $x$ in $[2, 4)$ holds
the face piece at $x = 2$. A particle of $[0, 1)$ near $x = 0.95$ lists it, and also the face piece at $x = -0.5$, which
is not in that range's border (unit test `ParticleSums_NeedAFaceBeyondTheNearBorder`). So the sums of a particle are
made in the range that holds the particle's cell, where the cell is in the set, and carried with the particle. A face
piece of another range reads the carried sums, so in a run in batches every range makes its sums before any range
reads them. And each particle is handled by the range that holds `FindCellAt` of the exact position its weights use: a
particle grouped at an old position and moved since is grouped again first.

In a grid piece, a reader called for a place outside what the rule covers gives none, the same answer as the edge of the
grid. For a position outside the cells of the range and the cells that touch them, `FindCellAt` gives no cell and
`FindNearbyFaces` says `kPositionOutsideGrid`, as for a particle that left the grid. A piece cannot tell the two apart,
so a stage must not rely on that answer for a place outside its range, and keeps each particle inside the closed
boxes of the cells of its range. There is also no reader from a corner place to the cells that have it as a corner: a
stage that computes a corner value from the cells around it needs those cells' blocks, which the border does not
promise. The pressure solve and the solids keep values at corners (@sec-cell-blocks) and must write a corner from the
cell that owns it, or add that reader to the rule.

Three examples show what a border of returned places alone would miss. A face piece between a small cell and a
bigger cell is kept at the small cell's size level, at plane 0 of the block above when the small cell sits at local 3
of the block below (@sec-face-pieces): a stage that writes the face piece and reads its two cells reads the block
below, which may hold no cell touching a cell of the set. A corner is kept at the smallest size level whose lattice
holds the point: a stage that writes a cell's value and reads its corners reads the block that keeps them, which may
hold no cell touching that cell. And the readers walk through bigger blocks: `FindCellAt` walks down from the
size-level-0 block through every bigger block of the cell it finds; `FindCellLow`, `FindCellHigh` and
`FindTouchingCells` reach a bigger cell through the bigger block of their own block; `FindTouchingCells` and
`FindFacePieces` reach small cells through the block that holds the split place, and `FindFacePieces` of a cell
touching the set reads the block of the small cells across its split side even when the face pieces it lists are kept
in the next block. With 2 size levels and the size-level-0 cells of coordinates 0 to 3 on each axis, all active but
those of coordinates 0 to 2, which are split, every place a reader returns from the size-level-1 block $(0, 0, 0)$ is
kept at size level 1, yet `FindCellAt` finds no cell in a piece without the size-level-0 block. So the border holds
every bigger block, up to size level 0, of each block of the set and of each block read. A few of them lie on no
walk; the rule takes them rather than tell, reader by reader, which bigger blocks a walk passes.

Each block can be asked from its own side (`ComputeIsBorderBlock`), through the grid's neighbour tables and the
readers themselves: whether one of its active cells, face pieces or corner places is read from a place of the set, or
whether a block under it is in the set or read. Each relation is reversed exactly. Touching is symmetric; a face piece
is on a side of exactly its two cells, so `FindFacePieces` of a cell is the reverse of `FindCellLow` and
`FindCellHigh`; a corner of a cell of size level $l$ is kept at size level $l$, $l + 1$ or $l + 2$, so the cells that
can have a corner kept in a block of size level $m$ are the 307 places of size levels $m - 2$ to $m$ whose boxes meet
the block's, and `FindCornerPlaces` run forward on those that the set reads says which corners land in the block. The
face search is reversed through its three position rules: for a block $X$ of the same size level as the start's block
$S$, $S$ is one of the 27 positions around $X$; for $X$ one size level bigger, $S$ lies under one of them; for $X$ one
size level smaller, $S$ is one of the 27 positions around the bigger block of $X$. A first look at the blocks around
$X$, at most two size levels away, skips the blocks that no reader of the set reads directly. The same set comes
forward, one place at a time (`ApplyCellPlaceReadMarks`, `ApplyFacePlaceReadMarks`): each place of the set runs its
readers and marks every block they read, with its bigger blocks; marks are only ever set to 1, so places marked at
once cannot conflict, and the marked blocks outside the set, in index order, are the border. A pass that runs several
stages on one load needs stacked borders: the border of its last stage around the range, then the border of the stage
before around the range plus that border, and so on. The buffer form of `FindBorderBlocks` does one layer per call, in
`BorderSearchBuffers` built once: it sets one mark byte per block to 0 (a mark left from an earlier call would add a
block silently), runs the marking functions from every place of the range and of the layers so far, and lists the
marked blocks outside the range. Every block of the layers so far is marked through its own cell places, so the list
holds them without a merge, and the list of one call is the next
call's layers so far. On the GPU the same call is a forward pass of launches in one stream. One thread resets the status
and one thread per block clears its mark. One thread per entry of the layers so far offers its failure key with an
integer atomic minimum: an entry not above the one before it, at or above the block count, or inside the range. One
thread per place of the set (304 per block: 64 cell places and 240 face places, the range's blocks first) runs the
marking function and stops at once when a key was offered. These threads write only the byte value 1, with plain
stores, to marks that no thread reads: a marking function writes the whole chain of bigger blocks every time and tests
no mark, and a byte store leaves the neighbouring bytes unchanged, so the marks after the launch are those of any order
of the places. The marks are never set for the set beforehand. The scan is a later launch, so it starts after every mark
was written: one thread per block writes 1 for a marked block outside the range (0 for every block after a failure, so
the prefix sum adds no word of an earlier call), an exclusive prefix sum gives each listed block its place, and one
thread per block writes itself there and the total into the status. Only the status comes back, so the list is the
CPU's, in index order. The thread that runs the marking functions keeps their results (the touching cells, the cells
near a face, the corners and, at four cell widths, a face search) as locals, so it needs a stack above the driver's
default of 1,024 bytes per thread, which the driver holds for every resident thread outside the run's memory; the memory
plan counts it as a driver stack entry. In a run in one piece the range is every block and the border is
empty. A sum over the grid that is not a sum per destination is one partial sum per block, kept in the fixed block
order, then added by @proc-tree-sum, so a total is the same in one piece and in any cut into ranges.

*Grid pieces.* A grid piece (`GridPiece`) is one range of a whole grid with its border, built as a grid of its own for
a run in batches. It lists its blocks in the whole grid's fixed order, never the range first. The border blocks lie
before or after the range, so the range is a run of the piece too, and a sum per destination visits the blocks it reads
in the same order as in the whole grid. The piece copies the whole grid's tables block by block. A neighbour that was
not loaded becomes "none"; a `kBetweenCells` face piece whose other cell lies in a block not loaded becomes
`kOutsidePiece`; and a plane-4 place stays `kNotAPiece` when its high neighbour exists in the whole grid but was not
loaded, so a grid piece never has a face piece that the whole grid does not have.

*Where a corner is stored.* The pressure solve and the solids keep values at cell corners. A corner is a point of the
5 × 5 × 5 lattice of every block whose box holds it, sides included, so one point can be a lattice point of several
blocks: two blocks of one size level that share a side, or blocks of two size levels. It is stored once. Among the
blocks of the grid whose lattice holds the point, it is stored in a block of the smallest cells (the largest size level
value); and among several blocks of that size level, in the one that comes last in the fixed order (the highest $z$,
then $y$, then $x$: the block on the high side, as for face plane 4). The size level counts whether or not a cell of
that block touches the point. A corner of a small cell in the middle of a big cell's side or edge therefore belongs to
the small cells only, and so does a corner of the big cell that no small cell touches, when it lies on the lattice of a
block of the small cells.

`FindCornerPlaces` gives the 8 corners of an active cell, bit 0 of the corner number for the high $x$ side, bit 1 for
high $y$, bit 2 for high $z$, and the 4 corners of a face piece, which are those of its smaller cell on that side. For a
cell of size level $l$ it looks, from size level $min(l + 2, L - 1)$ up to $l$, at the up to 8 blocks of that size
level around the point with `FindCellBlock`, and stops at the first size level that has one. No block of a smaller size
level can hold the point on its lattice. A block of size level $m$ is the box of one place of size level $m - 2$; for
$m >= l + 3$ the point can lie on that box only as one of its corners, and the block exists only when that place is
split, so by the siblings rule an active cell of size level $m - 1$ or more has the point as a corner. That cell touches
the cell of size level $l$, and the size-step rule gives $m <= l + 2$. A grid piece holds some of the whole grid's
blocks, so the bound holds there too. At size level $m$ the point's coordinate $p$, in cells of size level $m$, is an
integer, and the blocks whose lattice holds it are $floor(p\/4)$ on each axis, and $floor(p\/4) - 1$ as well when $p$
is a multiple of 4. The cell's own block holds the point, so the search ends at the cell's size level at the latest.
Two cells that share a corner each start at or beyond the size level that stores it, and neither goes past its own
size level, whose block holds the point: they find the same corner place. A place that holds no active cell or no face
piece, an index of no place and the tables of no grid give no corner, and `ComputeCornerPosition` gives $+infinity$ on
each axis for an index of no place; a caller tells such an index by the index, since a real corner of a frame whose
positions overflow gives $+infinity$ too. A corner place belongs to one set of tables: on a grid piece's border a
corner may be stored at another size level than in the whole grid, so values are matched between a grid piece and the
whole grid by position or id, never by corner place.

`ComputeCornerPosition` gives a corner place's position, origin $+ m h_l$ on each axis, for the integer coordinate $m$
of
the lattice point at its block's size level $l$: one product and one sum, as for a cell centre. The same point at size
level $l + j$ has the coordinate $m 2^j$ and the width $h_l \/ 2^j$, so the product is the same number, rounded the same
way. One point has one corner place, so it has one position, with the same bits on the CPU and the GPU.

For example, in a grid of one size level with cells of width 1 m, origin 0, and the cells $(3, 0, 0)$ (block 0) and
$(4, 0, 0)$ (block 1), the corner at $(4, 0, 0)$ m is lattice point $(4, 0, 0)$ of block 0 and $(0, 0, 0)$ of block
1; it is stored in block 1, at corner index 125. With the cell $(3, 0, 0)$ alone it is index 4. In the grid of a big
cell $(0, 0, 0)$ of size level 0 beside the eight small cells of $(1, 0, 0)$, block 0 is the size-level-0 block and
block 1 the size-level-1 block, which covers 0 to 2 m. The point $(1, 0.5, 0.5)$ m is the low corner of the small cell
$(2, 1, 1)$: lattice point $(2, 1, 1)$ of block 1, index $125 + 32 = 157$, none of the big cell's corners. The big
cell's corner at $(1, 0, 0)$ m is index 127, the low corner of the small cell $(2, 0, 0)$ too, and its corner at
$(0, 0, 0)$ m is index 125, though no small cell touches it: every corner of the big cell is stored in block 1.

#procedure(
  [`BuildGrid` on the CPU, the tables of one grid version],
  <proc-build-grid>,
  inputs: [the frame, the version (1 or more), the active cells in any order, and grid buffers with lent builder
    buffers.],
  outputs: [a grid that owns the buffers: the block ids in the fixed order, the neighbour tables, the cell states,
    the face piece kinds and the counts.],
  fails: [the first failure by step, then by position: bad input, a coordinate out of range, overlapping cells, a
    missing sibling cell, a broken size-step rule (both cells named), too many cell blocks. Nothing is returned; the
    buffers stay with the caller. No step allocates.],
  ([Check frame, version and ids], [Every frame value finite, $h_0 > 0$, 1 to 16 size levels, a smallest width of
    at least $2^(-1022)$, version 1 or more; one list, its count within its buffer (else bad input); at most the room's
    cell places (else too many cell blocks, before any id is read); then every id valid, of a frame size level, each
    coordinate within ±`kMaxCoord`.]),
  ([Sort, find overlaps], [Copy the ids into the builder buffers and sort them in the fixed order. Fail when a cell
    is listed twice or a bigger listed cell holds a smaller one.]),
  ([Mark split cells, check siblings], [Every bigger cell that holds an active cell is split. Each active cell checks
    that its seven siblings exist, active or split; when it is eighth 0 it adds the split cell one size level up and
    checks that cell's siblings, and so on up while the added cell is eighth 0 too.]),
  ([Size-step rule], [Every two touching active cells differ by at most one size level. Each pair is checked at its
    smaller cell, so a broken pair is reported once.]),
  ([Blocks and their count], [Every block that holds an active or split cell, sorted into the block-id table in
    the fixed order. Fail with too many cell blocks when the active and split cells exceed 64 times the room, or the
    blocks exceed the room.]),
  ([Neighbour tables], [For each block, its 27 neighbours of the same size level, its bigger block and its 8 smaller
    blocks, by binary search in the block ids.]),
  ([Cell states], [Each cell place: active, split, inside a bigger cell, or absent.]),
  ([Face piece kinds], [Each face place from the two cell places beside it (@sec-face-pieces); plane 4 is
    `kNotAPiece` where the block of the same size level on the high side exists.]),
  closing: [On the GPU, the same steps run in parallel through the same shared rules (@proc-build-grid-in-parallel),
    and give the same tables byte for byte.],
)

*Counting without tables.* A run's room, the `max_cell_blocks` of its buffers, is planned before any buffers exist:
the plan counts the cell blocks of the scene and sizes the buffers from that count. Building the tables to count
them would need buffers sized by the very number asked for; sized for the worst case of one block per cell, they
would cost about 1.1 kB per active cell. So `ComputeGridCounts` counts from the sorted ids alone
(@proc-count-grid), in one CPU array of 8 bytes per active or split cell, at most 9.2 bytes per active cell. It makes
the checks of `BuildGrid` without the length check and the block check, since no room exists yet. It states the face
rule of @sec-face-pieces a second time, from the places beside each cell instead of from built tables, and tests compare
its counts with those of built grids. Every place beside a cell is found by integer arithmetic on its coordinate. A
place beyond the coordinate limit of its own size level, outside ±`kMaxCoord`, holds no cell of that size level, and
its small cells lie beyond the limit of theirs. No id is made for such a place, since the id would name another cell.

#procedure(
  [`ComputeGridCounts` on the CPU, the counts of a list without its tables],
  <proc-count-grid>,
  inputs: [the frame, the version (1 or more) and the list of $n$ active cells in CPU memory, in any order.],
  outputs: [`GridCounts`: the cell blocks, the active cells in all and per size level, and the face pieces.],
  fails: [the first failure of steps 1 to 4 of @proc-build-grid: bad input, a coordinate out of range, overlapping
    cells, a missing sibling cell, a broken size-step rule. Out of memory when the copy cannot be allocated (the bytes
    asked are reported). Never too many cell blocks, since there is no room yet.],
  ([Check and sort the list], [Copy the $n$ ids into one array with room for $n + floor(n\/7)$ ids, and run steps 1
    to 4 of @proc-build-grid on the copy, without the length check, siblings included. The copy is then in the fixed
    cell order.]),
  ([Count the active cells], [One pass over the copy: the active cells of each size level and their total.]),
  ([Count the face pieces], [One active cell at a time. Beside each of its six sides, the place of the cell's own
    size level is _active_ when a cell of that size level is listed there, and _split_ when one of the four small
    cells of that place that touch the side is listed (the size-step rule makes all four active). Each lookup is a
    binary search in the sorted ids, made only for a coordinate in range at its own size level. The low side counts
    one face piece unless the place is split. The high side counts one unless the place is active (that cell counts
    the piece on its low side) or split (its small cells count theirs). A place inside a bigger cell and an absent
    place both leave one face piece with the cell, so the count needs no lookup of the bigger cell.]),
  ([Count the blocks], [Append the split cells that each active cell adds (its big cell when it is eighth 0, and
    on up while the added cell is eighth 0 too), sort every id by the key of its block (@sec-parallel-build), and
    count the runs of equal block ids.]),
)

== The weight curve

A particle passes values to the faces near it, and reads values back from them, with weights. The weight of a face
along one axis is the weight curve of the distance $t$ from the particle, in cell widths:

$ Q(t) = cases(
  3\/4 - t^2 & "if" |t| < 1\/2,
  (3\/2 - |t|)^2 \/ 2 & "if" 1\/2 <= |t| < 3\/2,
  0 & "otherwise",
) \
Q'(t) = cases(
  -2 t & "if" |t| < 1\/2,
  -(3\/2 - |t|) "sign"(t) & "if" 1\/2 <= |t| < 3\/2,
  0 & "otherwise",
) $ <eq:weight-curve>

Papers call it the quadratic B-spline. It is zero at and beyond 1.5 cell widths, and its slope $Q'$ is continuous.
For faces one width apart it has three sums that hold wherever the particle is: for any $t$, over every integer $k$,

$ sum_k Q(t - k) = 1, quad sum_k (k - t) Q(t - k) = 0, \
  sum_k (k - t)^2 Q(t - k) = 1\/4. $

The weights of faces of one size therefore add up to 1, their mean is the particle's own position, and their spread
along each axis is a quarter of a cell width squared. In three dimensions the weight is the product of the curve on
the three axes, $Q(d_x) Q(d_y) Q(d_z)$, and the three sums hold axis by axis. For example, at $t = 0.4$ the three faces
at offsets $-1.4$, $-0.4$ and $0.6$ get $Q = 0.005$, $0.59$ and $0.405$, which add up to 1.

== Nearby faces and their weights <sec-nearby-faces>

Each axis has its own faces, so a particle has three lists of nearby faces, one per axis, each with its own weights,
as Ding, Shinar and Schroeder [2] do on a staggered grid. Where every cell around the particle has one size, a list
holds 27 faces with the weights of the curve. Near a change of size, a solid or the liquid surface, faces of other
sizes enter the list or some faces are left out, and the weights must follow: they are scaled to add up to 1, and the
fit of how the velocity varies around the particle is measured from the faces actually kept. @proc-nearby-face-weights
lists the steps for one position and one axis; this part explains them, with $h$ the width of the active cell that
holds the position $bold(x)_p$ and $bold(x)_f$ the centre of a face piece $f$.

#procedure(
  [`FindNearbyFaces` and `ComputeNearbyFaceWeights`, the nearby faces of one axis and their weights],
  <proc-nearby-face-weights>,
  inputs: [the grid's tables, a position $bold(x)_p$, an axis, a cell block near the position, and the caller's
    kept mask.],
  outputs: [`NearbyFaceWeights`: the candidates, the mask, $W$, $bold(G)$, $bold(m)$, $S$, its inverse and the fit.
    For each kept face, `ComputeFaceWeight` gives $hat(w)_f$, the offset, the offset from the mean and the weight's
    change.],
  fails: [no active cell holds the position (`kPositionOutsideGrid`), or more candidates than the list's 216 places
    (`kInternalFault`, a broken table); the list then has no face.],
  ([Find the cell], [Find the active cell that holds $bold(x)_p$; $h$ is its width and $l$ its size level.]),
  ([List the candidates], [The face pieces of this axis that belong to the cell or to a cell touching it, of size
    level $l - 1$, $l$ or $l + 1$, whose centre satisfies $-1.5 h < x_f - x_p <= 1.5 h$ on each axis, tested in
    whole quarter widths: $n_p - 5 <= n_f <= n_p + 6$ with $n_p$ the position's quarter of the cell, clamped to 0 to
    3, and $n_f$ the face centre's, both counted from the cell's low corner. List them in ascending face index.]),
  ([Leave faces out], [The caller clears the bits of the faces it may not use. The grid never looks at physics.]),
  ([Raw weights], [For each kept face, $w_f$ and $bold(g)_f$ of @eq:scaled-weights.]),
  ([Sums], [Over the kept faces in list order: $W = sum w_f$ and $bold(G) = sum bold(g)_f$. If $W <= 0$, no kept
    face has weight: the fit is `kNone`, every weight, change, mean and spread is zero, and steps 6 to 9 do not run,
    so nothing divides by $W$.]),
  ([Scale], [$hat(w)_f = w_f \/ W$ and its change $(bold(g)_f - hat(w)_f bold(G)) \/ W$. The weights add up to 1
    and their changes to 0.]),
  ([Mean], [$bold(m) = sum hat(w)_f (bold(x)_f - bold(x)_p)$; each face's offset from the mean is $bold(x)_f -
    bold(x)_p - bold(m)$.]),
  ([Spread], [$S$ of @eq:weighted-spread, in m².]),
  ([Fit], [Factorise $S \/ h^2$ by Cholesky. If a pivot is at most
    `kMinPivotShare` times the trace or $W$ is below `kMinWeightSum`, it is `kMeanOnly`; else `kFull`, with
    $S^(-1)$.]),
)

*Candidates.* The candidates are the face pieces of the axis that belong to the cell holding the position or to a
cell touching it, whose own size level is at most one away from that cell's, and whose centre lies within one and a
half cell widths of the position on each axis, from just above $-1.5 h$ up to and including $+1.5 h$:
$-1.5 h < x_f - x_p <= 1.5 h$. The low end is open and the high end closed because a cell owns its low side. The
choice is made in whole quarter widths, never on rounded distances. On each axis, the quarter of the cell that holds
the position, $n_p = floor(4 q) - 4 c$ with $q = (x_p - "origin") \/ h$ and $c$ the cell's coordinate, clamped to 0 to
3, fixes a window twelve quarter widths long. The floor and the subtraction of whole numbers are exact, so no rounding
enters after the division. The face centre is counted in quarter widths of the cell from the cell's low
corner, an integer $n_f$ found from the ids alone, and the face is a candidate when $n_p - 5 <= n_f <= n_p + 6$. Every
face centre of the three size levels lies on that quarter lattice, so the integer test loses nothing, and both
processors make the same choice. Among cells of one size the window holds exactly three planes of faces on every
axis. Along the face's own axis the faces sit at $n_f = 4k$, and the window holds $-4$, 0 and 4 for $n_p$ of 0 or 1,
and 0, 4 and 8 for $n_p$ of 2 or 3; across it they sit at $n_f = 4k + 2$, and the window always holds $-2$, 2 and 6.
So every position has 27 faces, at a cell side, at a cell centre or at a corner alike, and a face exactly $1.5 h$ away
has weight zero. The candidates are listed in ascending face index.

*How many.* Among cells of one size a position has 27 candidate faces on each axis. Where cell sizes change it has
more: 76 when all 26 neighbours are split, and 124 in the largest case known, a cell whose eight corner neighbours and
six of its edge neighbours have its own size while every other neighbour, and everything beyond, is one size level
smaller. 124 is the exact maximum where only two cell sizes meet. No valid grid can pass 212. To see why, count in
quarter widths from the cell's low corner. The smallest candidate is a square of half a cell width. A candidate lies
on a face of the cell or of a cell that touches it, so along the axis it sits on one of at most six planes inside the
window, and across each of the two other directions it covers at least one of at most six half-width places inside
the window. A smallest piece is one such place. A bigger piece has its centre in the window, so the place one
quarter to one side of that centre is in the window too, and the piece covers it. Faces of one axis never overlap, so
no two candidates share a place: at most 6 × 6 × 6 = 216. Four of those places can hold no face: the 2 × 2 places of
the plane through the cell's own middle, $n_f = 2$, which lie inside the active cell. That leaves 212.

The list has room for 216 faces per axis (`kMaxNearbyFaces`), and the kept mask has one bit per place in the list, in
four 64-bit words. Running out of room would mean a broken table, not a grid: it is reported as an internal fault
(`kInternalFault`) and never overruns the list.

*Leaving faces out.* The caller clears the bits of a mask (`NearbyFaceMask`) for the candidates the particle may not
use: a face on the other side of a plate, a face inside a solid, a face with no value. The grid never looks at
physics; the mask is the caller's. Hu et al. [3] leave out grid values across a cut in this way, so that material on
the two sides of a thin solid does not exchange momentum.

*Scaling.* The raw weight of a kept face is the curve's product times its area in cell faces, $"area"_f \/ h^2$: 1
for a face of the cell's own size, 1/4 for a smaller face, 4 for a bigger one. Its change with the particle's position
follows from the slope. The raw weights and their changes are added over the kept faces in list order, and each is
divided by the sum:

$ w_f = Q(d_x) Q(d_y) Q(d_z) "area"_f / h^2, quad bold(d) = (bold(x)_f - bold(x)_p) / h, \
  bold(g)_f = -"area"_f / h^3 vec(Q'(d_x) Q(d_y) Q(d_z), Q(d_x) Q'(d_y) Q(d_z), Q(d_x) Q(d_y) Q'(d_z)), \
  W = sum_f w_f, quad bold(G) = sum_f bold(g)_f, \
  hat(w)_f = w_f / W, quad (partial hat(w)_f) / (partial bold(x)_p) = (bold(g)_f - hat(w)_f bold(G)) / W. $
<eq:scaled-weights>

The scaled weights $hat(w)_f$ add up to 1, and their changes add up to 0, whatever was left out.

*Mean and spread.* The mean offset $bold(m)$ is the weighted mean of the faces' offsets from the particle, and the
weighted spread $S$ (m², symmetric) measures how far the kept faces spread around that mean:

$ bold(m) = sum_f hat(w)_f (bold(x)_f - bold(x)_p), \
  S = sum_f hat(w)_f (bold(x)_f - bold(x)_p - bold(m)) (bold(x)_f - bold(x)_p - bold(m))^T. $ <eq:weighted-spread>

Each face's offset from the mean, $bold(x)_f - bold(x)_p - bold(m)$ (`offset_from_mean`), is the one used in fits and
in writes. The fit of how the velocity varies around the particle, as Jiang et al. [1] make it, needs the inverse of
$S$. It is
computed by a Cholesky factorisation of $S \/ h^2$, and the result has one of three outcomes (`NearbyFaceFit`):

/ `kFull`: weights, mean and an invertible spread: a fitted velocity change can be computed.
/ `kMeanOnly`: the weights are valid but the kept faces are too few or too flat to fit a change: a pivot of the
  factorisation is at most $10^(-12)$ times the trace (`kMinPivotShare`), or $W$ is below 0.1 (`kMinWeightSum`).
/ `kNone`: $W <= 0$: no kept face has weight, and the particle cannot read or write the grid on this axis.

*The complete case.* When all 27 faces of one size are kept, the curve's sums give $W = 1$, $bold(G) = 0$, $bold(m) =
0$ and $S = h^2\/4$ times the identity, up to rounding (@fig-nearby-faces). The inverse spread is then $4\/h^2$ times
the identity, the constant a uniform grid uses in place of the inverse. Here the inverse is always measured from the
faces kept, so the uniform case is the special case of one rule and never a separate path in the code.

A transfer uses the weights this way, per axis, with $v_f$ the value on face $f$. The fitted change, one matrix row,
is $bold(r) = (sum_f hat(w)_f v_f (bold(x)_f - bold(x)_p - bold(m))^T) S^(-1)$; the value at the particle is
$v_p = sum_f hat(w)_f v_f - bold(r) dot bold(m)$; and a particle writes $v_p + bold(r) dot (bold(x)_f - bold(x)_p -
bold(m))$ to face $f$ with weight $hat(w)_f$. Since the weighted offsets from the mean add up to zero, the written
momentum is exactly the particle's. The exact change of the read value with the position is
$sum_f v_f (partial hat(w)_f) / (partial bold(x)_p)$.
A face can also compute the weight one particle gives it (`ComputeFaceWeight` with the particle's sums for the
face's axis, `NearbyFaceSums`: $W$, $bold(G)$ and $bold(m)$), with the same bits as the particle's own list. The
face decides by the rule of the list whether the particle's list holds it: the face is a face piece of the
particle's cell or of a cell touching it, of a size level within one of the cell's, with its centre in the window of
the position. The window alone is not the rule, since it also admits face pieces of the small cells of a split place
that lie beyond the cells touching the particle's. A face the list does not hold gets weight 0. The face cannot see
the particle's kept mask: it gives a listed face the weight it would have had if the particle had kept it, so the
caller asks only about faces the particle's mask keeps, which means a mask whose rule the caller can apply to the
face alone.

*Properties.* These hold for every grid and every mask whose fit is not `kNone`:

- Constants are reproduced for any kept set: the weights add up to 1.
- Linear fields are reproduced whenever the fit is `kFull`. For $v_f = a + bold(b) dot (bold(x)_f - bold(x)_p)$, the
  fitted matrix row is $bold(b)$, because the weighted offsets from the mean add up to zero and $S S^(-1)$ is the
  identity, and the value at the particle is $a$.
- Weights are never negative, and are zero only for a face exactly $1.5 h$ away on some axis, where the curve is zero:
  a candidate's other offsets lie strictly inside the curve's range, and every area is positive.
- Weights jump when a particle changes cell across a size change: the width $h$, the candidates and the area factors
  all follow the cell that holds the particle. This is a known open question of the method, the force passed between
  cells of different sizes.

#figure(
  nearby-faces-figure(),
  placement: auto,
  caption: [The nearby faces of the X axis in a vertical slice. Left: a particle in a big cell beside smaller cells,
    with a plate on its other side. Faces beyond the plate are left out (hollow); each kept face is a dot whose area
    is its weight. The weights move the mean (arrow) toward the kept faces, and the spread (dashed ellipse) is measured
    from them. Right: with 9 faces of one size all kept (27 in three dimensions), the mean offset is 0 and the spread
    is $h^2\/4$ on every axis: the dashed circle has radius $h\/2$.],
) <fig-nearby-faces>

=== Cells near a face and the face search <sec-face-search>

Two readers start from a face piece instead of a particle. `FindCellsNearFace` lists the cells near a face piece: its
`cell_low`, its `cell_high` and the cells touching either, in ascending index order. Every cell whose particles can
list the face piece as a nearby face is among them, since a particle's candidates are face pieces of its own cell and
of the cells touching it. Some listed cells can never list it: a cell touching a bigger `cell_low` may be two size
levels bigger than the face piece. In a built grid each of the two cells touches the other, so the list has at most
$57 + 57 - 2 = 112$ cells (`kMaxCellsNearFace`). Tables that contradict themselves could break that, so the list stops
at 112 and never passes its array. The largest count known is 82: two cells of one size beside each other, with all 34
places of their size level around the pair split.

A face search (`BuildFaceSearch`, then `FindNextFace` until it gives no face) walks, in ascending face index, the face
pieces of the start's axis whose size level is one bigger than the start's, the same or one smaller, and whose centre
lies strictly within $r$ widths $h_l$ of the start's centre on each axis, $r <= 4$ (`kMaxFaceSearchRadius`). A larger
radius is taken as 4, and a radius that is not above 0, NaN included, finds nothing. The test is exact: both centres
are integers in quarter widths of $h_l$, and the distance on each axis is compared with $4 r$. Two face pieces of one
size level therefore find each other or neither, in any frame and on both processors. For example, in a grid of one
size level, radius 1.5 around a face piece finds the $3 times 3 times 3$ face pieces of its axis centred on it.

The search reads the blocks at three sets of positions around the start's block $B$, of size level $l$: the 27
positions $B + d$ of size level $l$, the 27 positions of size level $l - 1$ around $floor(B\/2)$, and the 8 positions
of size level $l + 1$ under each of the first 27, at most 270 blocks (`kMaxFaceSearchBlocks`). The positions are fixed
whether or not a neighbour exists. These blocks hold every face piece the search can find. Count in widths $h_l$.
Along the axis the start lies from $4 B$ to $4 B + 4$, plane 4 included, so a face piece within 4 widths lies strictly
between $4 B - 4$ and $4 B + 8$. It is kept in the block of its high cell place, or at plane 4 of the block below, so
its plane lies in that block's range, sides included. A block of size level $l$ two positions from $B$, one of size
level $l + 1$ outside $2 B - 2$ to $2 B + 3$, and one of size level $l - 1$ two positions from $floor(B\/2)$ all lie
too far. The same holds across the axis, where the centres lie strictly inside the blocks.

Face pieces two or more size levels away are never found, for two reasons. Toward smaller cells: a cell one size level
smaller is half as wide, the next a quarter, and so on, so face pieces of any smaller size level can lie within 1.5
widths of a face piece. A search over every smaller size level would read a number of blocks with no bound: it could not
keep its state in a fixed array on the GPU, and a grid piece would have no finite border for it. Toward bigger cells the
cost is bounded, at most 8 more blocks per size level within 4 widths, so leaving them out is a choice: one rule for
both directions, size levels $l - 1$ to $l + 1$, the size levels of a particle's candidates above, so the search loses
no face piece that a particle in a cell near the start could use. In a grid piece the search reads the blocks at these
positions that the piece holds, and finds their face pieces only.

== Memory per block <sec-grid-memory>

Every array of a grid version has room for `max_cell_blocks` blocks, so its memory is counted per cell block. Each
array is one memory piece of the run's reserved block (@sec-run-memory) and one entry of the memory estimate
(@sec-memory-estimate), listed in the order the builders take their memory pieces. @tab-grid-memory gives the
entries. The five tables are the grid itself, 456 bytes per block, about 7.1 bytes per cell place. The builder buffers
are what a build needs while it runs. The CPU builder sorts the cell ids in place and needs nothing else; the GPU
build also keeps sort keys, two orders and the working buffers of @proc-stable-order, whose `OrderBuffers` take 6 bytes
per item and 4,096 bytes more on the GPU. Three of its entries hold one more item than the cell places, because a prefix
sum writes its total after the last entry: hence the 4 bytes once of each order, and the 6 bytes once of the order
buffers. A GPU grid keeps nothing in CPU memory: the only value that comes back from a build on the GPU is its status
record.

#figure(
  align(left, {
    set par(justify: false)
    table(
      columns: (1fr, auto, auto, auto),
      align: (left, left, right, left),
      table.hline(stroke: 0.6pt),
      table.header([*Entry*], [*Counted per*], [*Bytes per block*], [*Where*]),
      table.hline(stroke: 0.4pt),
      [block ids], [cell block], [8], [run],
      [same-size-level neighbour blocks], [cell block], [108], [run],
      [bigger and smaller blocks], [cell block], [36], [run],
      [cell states], [cell place], [64], [run],
      [face piece kinds], [face place], [240], [run],
      [*grid tables*], [], [*456*], [run],
      table.hline(stroke: 0.4pt),
      [_CPU grid:_ builder buffers: cell ids], [cell place], [512], [CPU],
      [_CPU grid:_ piece map], [cell block], [4], [CPU],
      [*one CPU grid*], [], [*972*], [456 run, 516 CPU],
      table.hline(stroke: 0.4pt),
      [_GPU grid:_ builder buffers: cell ids], [cell place], [512], [run],
      [_GPU grid:_ builder buffers: sort keys], [cell place], [512], [run],
      [_GPU grid:_ builder buffers: order], [given count], [256, and 4 once], [run],
      [_GPU grid:_ builder buffers: second order], [given count], [256, and 4 once], [run],
      [_GPU grid:_ builder buffers: order buffers], [given count], [384, and 4,102 once], [run],
      [_GPU grid:_ piece map], [cell block], [4], [run],
      [_GPU grid:_ build status], [given count], [one record], [run],
      [*GPU builder buffers, one set*], [], [*1,920, and 4,110 once*], [run],
      [*one GPU grid with lent builder buffers*], [], [*460, and one record*], [run],
      table.hline(stroke: 0.4pt),
      [_border search:_ border marks], [given count], [1], [run],
      [_border search, GPU:_ border sums], [given count], [4, and 4 once], [run],
      [_border search, GPU:_ border order buffers], [given count], [6, and 4,102 once], [run],
      [_border search, GPU:_ border status], [given count], [one record], [run],
      [_border search, GPU:_ GPU stack], [given count], [3,600 per resident thread], [driver],
      table.hline(stroke: 0.4pt),
      [`CellValues` of `double`], [cell place], [512], [run],
      [`FaceValues` of `double`], [face place], [1,920], [run],
      [`CornerValues` of `double`], [corner place], [1,000], [run],
      table.hline(stroke: 0.6pt),
    )
  }),
  caption: [The grid's memory per cell block. "Run" is the memory of the processor the run chose, "CPU" always CPU
    memory. Every entry names one array; the bold lines are totals. The value buffers at the end belong to the
    modules that keep values on the grid.],
) <tab-grid-memory>

Border search (`BorderSearchBuffers`) keeps one mark byte per block on both processors. On the GPU it also keeps one
word per block plus one for the exclusive prefix sum of the listed marks, the working buffers of that sum and one status
record: 11 bytes per block, 4,106 bytes once and one record. A CPU run therefore needs 1 byte per block more for border
search, and one set serves every call, since a call sets its marks first.

A run on the GPU keeps two grids, the committed one and the one of the step's try, which share one set of builder
buffers: 2 × 460 + 1,920 = 2,840 bytes per block, plus the 4,110 bytes and two status records once. A grid of
1,000,000 blocks (64 million cell places) and its try therefore take about 2.84 GB. The grid's own GPU functions keep
under the driver's default stack of 1,024 bytes per thread, so the grid has no entry held by the driver. A GPU
function that computes nearby-face weights keeps 1,152 bytes per thread for one axis, or 3,456 for three, which is
above that default, so the module it belongs to counts a stack held by the driver, as contact does for its particle
visit.

== Building the tables in parallel <sec-parallel-build>

A run is all CPU or all GPU (@sec-run-on). In a GPU run the list of active cells is made on the GPU, the grid is
built there, and the stages read it there. The cell list never leaves the GPU and no table comes back: the only value
copied to the CPU is the build's status record, which holds the counts or the first failure. @proc-build-grid is
written for one thread that walks the list in order. On the GPU many threads run at once, and a step whose result
depended on which thread came first would give another grid from run to run. So each sequential step becomes the
same five parts: compute a key for each item, put the items in a stable order by that key (@proc-stable-order),
mark the items where something starts, take the exclusive prefix sum of the marks, and let each marked item write
its result at the place its sum gives, a scatter. Each part is one thread per item, or a call of `BuildStableOrder`
or `ComputeExclusivePrefixSum`, whose results do not depend on timing (@sec-stable-order). The only values threads
share are an integer minimum and integer counts, and neither depends on the order in which threads add to them.

*Numbering the blocks.* Step 5 of @proc-build-grid lists the blocks in the fixed order. In parallel, every active
and split cell takes the key of its block, and the stable order puts the cells of one block side by side, the blocks
in the fixed order. Say the sorted cells, here of one size, lie in the blocks $[A, A, B, C, C, C]$. An entry is
marked 1 when it is the first of its run: the first entry, or one whose block differs from the entry before. The
marks are
$[1, 0, 1, 1, 0, 0]$, their exclusive sums $[0, 1, 1, 2, 3, 3]$, and the total, 3, is the block count. The first
entry of each run writes its block at the index its sum gives: $A$ at 0, $B$ at 1, $C$ at 2. Any entry finds its
block index as its sum plus its mark minus 1. The first entry of a run has the index as its sum and a mark of 1; every
other entry has a sum one past the index, since the run's first entry was counted, and a mark of 0. A mark is a
function of the sorted list alone, so every thread computes it again where it needs it, and no thread waits for
another. @fig-parallel-build shows the same steps on a grid of two sizes with its own list,
$[A, B, B, C, C, C, C]$, where the split cell that its small cells add joins the run of its block.

#figure(
  parallel-build-figure(),
  placement: auto,
  caption: [Numbering the blocks in a vertical slice of a grid of two sizes. Six active cells, numbered in input
    order, lie in three blocks: $A$ and $B$ of the big cells, which the dashed line at $x = 0$ divides, and $C$ of
    the small cells, whose low corner is $B$'s and which comes after $B$ because the bigger cells come first. The
    split cell 6 is added after the active cells. The first entry of each run writes its block; the sums give every
    block its index in the fixed order. Any starting order gives the same marks. In three dimensions the slice has
    four more small cells behind it, which join run $C$.],
) <fig-parallel-build>

*The keys are the fixed order.* The order of blocks and cells is kept because the sort keys are the fixed order
itself, written as three unsigned integers, one per pass of the stable order, the least significant first. For a
coordinate $c$ of a cell of size level $l$, the biased coordinate is

$ b(c, l) = (c + 2^(18 + l)) dot 2^(15 - l) = c dot 2^(15 - l) + 2^33, $

the cell's low corner in widths of a size-level-15 cell, moved up by $2^33$. A cell's key is $16 b(x, l) + l$ in the
first pass, $b(y, l)$ in the second and $b(z, l)$ in the third. After the three passes the cells are ordered by $z$,
then $y$, then $x$, and at one low corner the bigger cell, of the smaller $l$, comes first. A block's key is the same
with the block's low corner, $4 floor(c \/ 4)$, in place of $c$, so every cell of a block has the key of its block.
No key is ever negative, so no negative number is shifted, and both processors compute the same bits with integer
adds and shifts. A valid coordinate is at least $-(2^18 - 1)$, so $c + 2^(18 + l) >= 1$. A block's low corner is at
least $-2^18$, so its sum is at least 0, and 0 only for the lowest block of size level 0. The largest biased value is
$(2^18 - 1 + 2^18) dot 2^15 = 2^34 - 2^15$, so the first pass stays below $2^38$. An id that is not a cell id at
all, such as the id 0, has the largest key, $2^64 - 1$, in every pass, and sorts last. An id of a size level the frame
does not use, or with a coordinate out of range, keeps an ordinary key.

*The first failure.* @proc-build-grid reports the first failure by step, then by position. In parallel each failing item
computes one integer, its failure key, $"step" dot 2^56 + "position" dot 2^8 + "detail"$, and the GPU keeps the smallest
with an integer minimum. The minimum of a set of integers does not depend on the order in which the threads offer them.
The step counts the checks in the order of @proc-build-grid: 1 the ids, 2 overlaps, 3 siblings, 4 the size-step rule,
then 5 the active and split cells against the room's cell places and 6 the blocks against the room, the two checks of
its step 5, and 7 a block that has no bigger block in the list, which no valid list produces and which is reported as an
internal fault with no cell. Step 5 has a second key, with detail 1: more active and split cells than the sort by block
orders, also a state no valid list produces, reported the same way. The position is the index in the input list for
step 1, the place in the fixed cell order for steps 2 to 4, 0 for steps 5 and 6, and the block's index for step 7. The
detail says which sibling is missing, in which of the 26 directions the size-step rule is broken, or which key of step 5
it is. The input checks of P1, the frame, the version, the shape of
the list and its length, come before every keyed check on both processors, so they never take part in the minimum. Each
later step first reads the smallest key and does nothing when an earlier step failed, with one exception: every array
that a prefix sum reads is written in full at every launch, with zeros after a failure, so the sum never reads values
left by an earlier grid or an earlier step. A stable order may still read keys left from before, which cannot make it
fail, and its result is then never used. So no step acts on data that an earlier step found bad. Once the key is known,
one thread runs that check again at that position to name its cells. The CPU runs each step to its end before the next
and scans the positions in order, so it stops at the same failure.

*No cell of an earlier build.* The builder buffers are reused, so they hold the cells of the build before. The split
cells are counted on the GPU, so the CPU does not know how many entries the sort by block must order. It sorts a
count it does know, $min(n + floor(n \/ 7), 64 dot #raw("max_cell_blocks"))$, where $n$ is the number of active
cells: every split cell has eight small cells, each active or split, so there are at most $n \/ 7$ split cells.
Before that sort, every place after the last active or split cell, up to that count, is filled with the id 0,
which is never valid, sorts last and is never marked. Without the fill those places would still hold cells of the build
before, and they would become blocks.

*Why the tables equal the CPU's.* Each table entry is a function of the sorted list, not of the order in which the
work was done. A block's neighbours come from a binary search in the sorted block ids, a cell state from the sorted
cells and the block tables, and a face piece kind from the two cell states beside its place. The CPU builder and the
GPU write every entry through the same shared rules (@sec-run-on), with integers only, and their sorted lists are the
same, since the stable order is. So the tables are the same byte for byte. The counts are sums of integers, kept with
integer adds, which do not depend on the order either. @proc-build-grid-in-parallel lists the twelve steps, each with
what one thread handles. The code numbers them P1 to P12, apart from the steps of @proc-build-grid.

#procedure(
  [`BuildGrid` on the GPU, the tables of one grid version in parallel],
  <proc-build-grid-in-parallel>,
  breakable: true,
  step-prefix: "P",
  inputs: [the frame, the version (1 or more), the list of $n$ active cells in any order in a buffer on the GPU, with
    $n$ as a CPU number, and grid buffers on the GPU with their lent builder buffers.],
  outputs: [the tables of @proc-build-grid in the grid buffers, and the status record, the one value copied back to
    the CPU.],
  fails: [as @proc-build-grid, the failure of the smallest key. No grid is returned, and the buffers stay usable.
    After a failed GPU call their contents are unspecified, and the run stops. A state that no input can produce is
    reported as an internal fault, never as bad input.],
  ([Check the inputs, on the CPU], [The frame, the version, the shape of the list (bad input) and its length,
    as in step 1 of @proc-build-grid: a list of more than $64 dot #raw("max_cell_blocks")$ cells fails with too many
    cell blocks before any id is read and before anything runs on the GPU. Then one thread resets the status record:
    no failure, every count zero. Nothing is written from the CPU.]),
  ([Check the ids, one thread per input cell], [Each id as in step 1 of @proc-build-grid; a bad id offers its
    failure key, step 1 at its input index, to the integer minimum.]),
  ([Sort into the fixed cell order, one thread per input cell], [For each of the three passes, each thread writes
    its cell's key of that pass, and a stable order follows, starting from the order of the pass before. Then the
    thread of place $j$ copies the input cell `order[j]` to place $j$ of the cell ids. A bad id still has a key: the
    sort runs, and its result is never used.]),
  ([Overlaps, one thread per sorted cell], [Each thread first reads the smallest key and returns when an earlier
    step failed, as in every step after this one, except where it writes a value that a prefix sum reads: it then
    writes 0. A cell equal to the one before, or held by a listed bigger cell,
    offers its key, as in step 2 of @proc-build-grid.]),
  ([Siblings, one thread per sorted cell], [The siblings of the cell, and those of the split cells its eighth-0
    chain adds, as in step 3 of @proc-build-grid. A check of a list that is still being split (`CheckSizeStepRule`)
    skips this step.]),
  ([Size-step rule, one thread per sorted cell], [The first of the 26 directions, in $z$, then $y$, then $x$ order,
    in which an active cell two or more size levels bigger touches the cell, as in step 4 of @proc-build-grid.]),
  ([Add split cells and fill the tail of the array], [(a) One thread per active cell writes how many split cells
    it adds, or 0 when an earlier step failed; an exclusive prefix sum gives each its first place and the total.
    (b) One thread reads the total, the split cells, and fails with too many cell blocks (key step 5) when the
    active and split cells pass
    $64 dot #raw("max_cell_blocks")$. Otherwise, when they pass the sort count,
    $min(n + floor(n \/ 7), 64 dot #raw("max_cell_blocks"))$, it reports an internal fault (key step 5, position 0,
    detail 1): the bound $n \/ 7$ rules this out, and the sort of P8 would leave cells out. Both keys are of step 5,
    so every later step does nothing after either. (c) Unless an earlier step or (b) failed, one thread per active
    cell writes the split cells it adds after the $n$ active cells, at $n$ plus its sum. (d) Unless an earlier step
    or (b) failed, one thread per place below the sort count writes the id 0 at every place from $n$ plus the split
    cells on.]),
  ([Sort cells by block and number the blocks], [Three passes of the block keys order the places below the sort
    count; the id 0 sorts last. Sorted entry $j$ is the cell at place `order[j]`, an active cell when `order[j]`
    is below $n$. One thread per sorted entry writes its mark, 0 when an earlier step failed, else 1 for a valid entry
    that is the first of its block and 0 for any other. An exclusive prefix sum of the marks; one thread reads the
    total, the block count, and fails with too many cell blocks (key step 6) above `max_cell_blocks`. Then, unless the
    block count or an earlier step failed, each marked entry writes its block id at its sum, so the block ids come out
    in the fixed order.]),
  ([Neighbour tables, one thread per block], [The 27 neighbours, the bigger block and the 8 smaller blocks, as in
    step 6 of @proc-build-grid. A block of size level 1 or more without its bigger block then offers its key, step 7
    at its index, an internal fault.]),
  ([Cell states, one thread per cell place], [(a) One thread per cell place writes absent. (b) One thread per
    sorted entry writes active for an active cell or split for an added one, at its block index, its sum plus its
    mark minus 1, and its place in the block, and adds 1 to the active cells of its size level with an integer add.
    An entry of the fill writes nothing.
    (c) Two launches, one thread per cell place in each. The first computes the state each place takes, inside a
    bigger cell for a place still absent whose covering place in the bigger block is active, and writes it to the
    sort keys, free after P8; it reads only active marks, and writes no state. The second copies each place's
    answer into its state. No launch reads a place that another of its threads writes.]),
  ([Face piece kinds, one thread per face place], [Each kind from the two cell states beside the place, as in step 8
    of @proc-build-grid. A thread adds up the face pieces of its places and adds the sum to the count with one
    integer add.]),
  ([Counts and status, one thread], [On success, the active cells as the sum over the 16 size levels; on failure,
    the error of the smallest key and the cells its check names. The CPU then copies the status record back: the
    one transfer of the build. On success the buffers belong to the new grid; a record without a failure that counts
    more blocks than the room, or other active cells than $n$, is an internal fault.]),
  closing: [A GPU function that needs a count found during the build, the split cells or the blocks, reads it from
    the status record, so no count comes back during the build: the launches of P9, P10 (a), (c) and P11 are sized
    for the room, P10 (b) for the sort count, and a thread past the count does nothing. Each launch starts only
    once the one before it started. P10 and P11 come after every keyed step, so any key stops them. Steps P1 to
    P6, then the failure half of P12, are also the check of a list on the GPU before a build.],
)

== The cell size rule <sec-cell-size-rule>

The grid keeps small cells where the flow needs detail and big cells elsewhere. The cell size rule says, one cell at a
time, how wide that cell should be: its wanted cell width. Two parts of the simulation read it: the step, when it
chooses the active cells of a try (`ChooseActiveCells`), and cell resize, after a committed step (`ResizeCells`). The
grid owns the formulas (`grid/cell_size_rule.h`); the passes that apply them to every cell belong to those modules.

*What a cell knows.* The rule reads a few facts of one cell (`CellSizeFacts`): its depth $d$ below the liquid surface
and its distance $e$ to the nearest solid, both in metres and $+infinity$ when nothing is near; the size $G$ of its
velocity change, in 1/s, how fast the velocity changes per metre around the cell; its cell size factor $f$, a positive
number without unit; and two flags, whether the cell touches an air pocket and whether gas lies in or beside it
(`wants_smallest_cells`). With $w_"min"$ the width of the smallest size level the settings allow
(`smallest_size_level`) and $w_"max" = h_0$ the width of size level 0, the wanted cell width is

$ w & = w_"min" quad "if" d < n_"surface" w_"min" "or" e < n_"solid" w_"min", "and otherwise" \
  w_"depth" & = w_"min" max(1, d / (n_"surface" w_"min")), \
  w_"solid" & = w_"min" max(1, e / (n_"solid" w_"min")), \
  w_"shear" & = cases(c_u U_"ref" \/ G & "if" G > 0, +infinity & "otherwise",) \
  w & = max(w_"min", min(min(w_"depth", w_"solid", w_"shear") f, w_"max")), $ <eq:wanted-cell-width>

The first line keeps the smallest width, whatever the factor, at pockets and gas, within the surface reach and within
the solid reach: for a cell that touches an air pocket, a cell with gas in or beside it, a cell less than $n_"surface"$
smallest widths below the surface and a cell less than $n_"solid"$ smallest widths from the nearest solid. A fact or a
reference speed that is not a number gives $w_"min"$ too, the side that keeps detail. The surface reach $n_"surface" =
8$ and the solid reach $n_"solid" = 2$ count smallest cell widths, unlike the reach of a solid, and are settings
(`surface_reach_in_cell_widths`, `solid_reach_in_cell_widths`). A negative depth or distance counts as the surface or
the solid. The result always lies between $w_"min"$ and $w_"max"$.

*Why these terms.* The flow changes fastest over short distances at the surface and beside a solid, so cells there
stay the smallest. Beyond the surface reach or the solid reach, a term grows in proportion to the distance: a cell
twice as deep may be twice as wide. The width then doubles only across $n$ cells of one size, so the number of cells
across any depth is about the same at every size level, and the size changes slowly enough that the size-step rule
(@sec-cell-sizes) rarely has to step in; where the shear term changes fast between neighbours, it does. For example,
with $h_0 = 1$ m and 5 size levels, $w_"min" = 0.0625$ m: a depth of 1 m gives $0.0625 times 1 \/ 0.5 = 0.125$ m, a
distance of 0.5 m to a solid gives $0.0625 times 0.5 \/ 0.125 = 0.25$ m, and a cell with both gets the smaller,
0.125 m. Away from both, the shear term keeps cells small where the velocity changes over a short distance: a cell is
about as wide as the distance over which the velocity changes by $c_u U_"ref"$. Its factor $c_u = 0.25$
(`shear_width_factor`) is a starting value, to be calibrated against scenes. A liquid that moves as a whole, with
$G = 0$, has no shear term and needs no small cells.

*The reference speed.* The shear term measures a velocity change against

$ U_"ref" = max(u_"max", sqrt(g w_"max")), $

where $u_"max"$ is the largest liquid speed, in m/s, and $g$ the size of gravity, in m/s². The second speed is that of
a gravity wave in liquid as deep as the biggest cell is wide. A nearly still liquid still carries waves at about that
speed, so its small velocity changes are measured against it and do not split cells. The reference speed has no
floor: with zero gravity and a liquid at rest it is 0, so any velocity change above 0 gives a shear term of 0 and the
smallest width, while a velocity change of 0 still gives no shear term.

*The cell size factor.* An artist can ask for more or less detail in parts of the scene with a volume of cell size
factors: a table of positive factors over the scene. A cell reads the smallest factor over its box. Each factor is
stored as a whole number $c$ from $-24$ to $24$, its cell size factor code, and the factor is $f = 2^(c \/ 4)$: four
steps make one doubling, so the 49 factors run from $2^(-6)$ to $2^6$. The table of the 49 factors is computed with no
library function: a power of two, started at $2^(-6)$ and doubled before every fourth factor, times the double nearest
$2^0$, $2^(1\/4)$, $2^(1\/2)$ or $2^(3\/4)$, written as a hexadecimal literal. A double times a power of two is exact
while the result stays in range, here $2^(-6)$ to $2^6$, so every machine holds the same 49 bit patterns. A code
stored outside $-24$ to $24$ reads as the nearest limit, and outside the volume the code is 0, factor 1.

The smallest code over any box comes from layers of halved size. Layer 0 holds one code per entry, one entry for every
$2^m$ smallest cells on each axis ($m$ is `entry_spacing_log2`), and layer $L$ has $ceil(n \/ 2^L)$ entries per axis for
$n$ entries of layer 0, each the smallest code over the 2 × 2 × 2 entries of layer $L - 1$ that exist. A cell $s$ size
levels above the smallest size level (the smallest size level minus its own, so $s = 0$ for the smallest cells) is as
wide as $2^s$ smallest cells. If $s <= m$ it lies inside one entry of layer 0, the one at index $floor(i 2^s \/ 2^m)$
for its coordinate $i$ on each axis. Otherwise it reads layer $L = s - m$ at index $i$, an entry that is the smallest
code over exactly the entries its box covers that exist, so a cell reads one entry of the layer whose entries are as
wide as the cell, not every entry its box covers. A negative index, or an index at or beyond the layer's entry count,
gives code 0, as does a layer the volume does not hold. A cell whose box reaches past the last entry of the volume on
an axis, which happens when $(i + 1) 2^L > n$, reads the smaller of that entry and 0, since outside the volume the
code is 0: the smallest code over the whole box holds for every cell, and a positive code never enlarges a cell
that straddles the edge of the volume. A cell inside one entry of layer 0 never reaches outside. The lookup uses
integers only, so every machine reads the same entry. The factor multiplies the width the physics asks for rather
than replacing it, so it scales the rule without hiding it: a factor of 0.5 halves the width wherever the physics alone
allows bigger cells, a factor of 2 doubles it, the clamp still bounds the result, and the cells kept smallest above
stay smallest. A factor of 1 gives the rule's own width bit for bit, since a product by 1 is exact. The table is the
same on every machine, because one formula, written once for both processors, computes it from the scene, so every
machine chooses the same cells.

*The wanted size level.* A cell's wanted size level is the first $l$, from 0 up, with $h_l <= w$, and the last size
level of the frame when none is. It comes from comparing $w$ with a table of the widths $h_l$, never from
$log_2(h_0 \/ w)$ or from $2^(-l)$ by `pow`: those functions may round their last bits differently on the two
processors. The widths of the table are exact halvings (@sec-cell-sizes), and the rule uses only products,
quotients, comparisons, min, max and one correctly rounded square root, in one fixed order with no sum, so no fused
multiply-add can arise. The CPU and the GPU therefore give the same bytes. The rule reads only the facts of its own
cell, so any cut of the grid into batches gives each cell the same size level too. For example, in the frame above,
$w = 0.125$ m gives size level 3, the double just below it gives 4, and 0.2 m gives 3.

*New cells.* When liquid enters space with no cell, the step adds a cell there. The new cell borders air, so its depth
is 0 and the rule wants $w_"min"$. It starts instead one size level smaller than its nearest committed cell, and later
resizes bring it down to its wanted width.

The nearest committed cell of a position $bold(x)$ is the nearest among the active cells of the committed grid that hold
one of 27 searched positions, $bold(x) + (i, j, k) h_0$ with each of $i$, $j$, $k$ in ${-1, 0, 1}$, each found as any
position finds its cell (@sec-cell-blocks). The cell whose centre is nearest $bold(x)$, by squared distance, wins; at an
equal distance the cell that comes first in the fixed order wins, so the answer does not depend on the order of the
search. The new cell's size level is that cell's size level plus one, but never smaller than the smallest size level the
settings allow; when no searched position lies in an active cell, it is that smallest size level. A smallest size level
beyond the frame reads as the frame's last. For example, with $h_0 = 1$ m, 3 size levels and the size-level-0 cell at
(0, 0, 0) beside the 8 size-level-1 cells of the size-level-0 cell at (1, 0, 0), from (−0.5, 0.5, 0.5) m the searched
position (0.5, 0.5, 0.5) m finds the big cell, so the new cell gets size level 1. From (2.25, 0.25, 0.25) m the searched
position (1.25, 0.25, 0.25) m finds the small cell at (2, 0, 0), whose centre is 1 m away, so the new cell gets size
level 2; the small cell at (3, 0, 0), whose centre is only 0.5 m away, holds no searched position.

The cell found may therefore not touch the new cell, and a nearer, smaller cell that no searched position finds can lie
between them. When such a cell is two or more size levels smaller than the new cell, the new cell breaks the size-step
rule (@sec-cell-sizes). So does the cap when cells already there are smaller than the settings' smallest size level. The
step that adds cells therefore closes the size-step rule over its new list of cells, as the start grid below does, and
the check of a cell list turns down a list that breaks it.

Distances are measured in biggest cell widths from the frame's origin, so in any frame the frame check accepts no
distance overflows, and an underflow never changes which cell is nearer. Every offset $i h_0$ is exact, and each
distance uses only halvings, a quotient, subtractions, products and sums in a fixed order, with no fused multiply-add on
either processor, so the CPU and the GPU choose the same cell.

*The start grid.* The grid of the first step comes from the same rule, read from the scene's start shapes instead of
from particles. A cell's depth and distance are taken at its centre minus half its diagonal, a bound that no point of
the cell is below, so no point of the cell wants a smaller cell than the cell gets. The start has no velocity change,
so no shear term. The wanted cells are then closed under the size-step rule: a cell two or more size levels bigger
than a cell it touches is split, again and again, until none is left.

// Every floating figure and procedure is placed before the references start.
#place.flush()

= References

#block[
  #set text(size: 9pt)
  #set par(justify: false, first-line-indent: 0pt, hanging-indent: 1.8em, spacing: 0.75em)
  #include "references.typ"
]
