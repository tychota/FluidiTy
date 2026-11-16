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
way.

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
only. Each copy also has a form that takes a slice: `count` items starting at index `first`. Sizes, slices and overlap are checked
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
take and give back memory. `ReadProcessorDescription` gives the run report the processor's name, the software that
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

A piece that does not fit in what is left of the block is refused with `kRunMemoryTooSmall`: nothing is taken and
nothing is written, and the block is unchanged. It is an error, not a broken precondition, because the caller
cannot guarantee it: what fits depends on a plan made elsewhere, and limits raised while the run goes on can
cause it. The run reserves exactly what its estimate gives, so the error usually means the estimate missed an
array.

The bytes to reserve come from the same rounding, written once in `ComputeRunMemoryPieceBytes`. Over the entries
kept in one memory (`MemoryWhere`), `ComputeRunMemoryBytes` adds the copies times the piece bytes of one copy, since
every copy is its own buffer and its own piece. This is a contract for every module: each copy of an entry is
exactly one buffer of items × bytes each. A module that keeps 24 bytes per particle as three buffers of 8 bytes
writes three entries, since three pieces round three times: for 10 particles, 3 × 256 = 768 bytes, not 256. A GPU run reserves one block on the GPU for the `kRunOn` entries and one on the CPU for the `kCpu` entries;
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
same bits. On the GPU, the dot product writes its products to the sum buffers first and then sums them with the same steps;
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
by digit, and chunk by chunk within a digit. Its exclusive prefix sum gives, for each digit and chunk, the place where that chunk's first item with that digit lands: all items with a smaller digit come before
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
rest of the step sliding along the surface (the slide). The straight path of one step is the sweep.

The same code moves particles of every material: liquids, snow, springy solids and gas markers. Only two values
differ between them: the friction number and the plate side a particle may carry. The code has no case for one
material. Contact does not decide how a solid moves from step to step, nor the force on it; it reports, for each
particle, what the solid did to it, and the solids module adds those reports up.

Contact works on both velocities of a particle (@sec-particle-velocities). The sweep follows
`velocity.for_this_step`, since that is the velocity that moves the particle. At a hit both velocities get the
response, each with its own approach speed. In this section, $T$ is the step duration (the same value as $Delta
t$), $"tol"$ the contact tolerance, $d$ the distance from a point to a solid and $g = d - "tol"$ the gap
(@tab-notation).

Two functions do the work. `MoveParticle` moves one particle; it lives in a shared header, so the CPU loop and the
GPU thread run the same code (@sec-run-on). `MoveParticles` runs it for every particle of one material in one
range of cell blocks: a CPU loop, or one GPU thread per particle. Particles are independent, so contact adds no
floating-point values across particles, and nothing in it depends on the order in which threads finish. A call first
checks itself once: sizes, pointers, overlapping arrays, the scalars and every solid. Then it visits each particle
once, moves it and writes its outputs. At every hit on one surface, the speeds involved times the time left must stay
within $2^40 "tol"$, the bound that keeps a response's rounding below $"tol"\/64$; otherwise the move is too close to
call (`kMovesTooFastForStep`), and a shorter step cures it. Speeds beyond any scene, a sum of absolute axes above
$2^480$ m/s for a velocity or a solid's motion, are bad input. The outputs belong to the step's try: after a failure their
contents are unspecified, and the step discards the whole try. When several particles fail, the one reported is the
failing particle with the lowest id, then the lowest index, found by integer minima. The CPU
and the GPU give equal discrete results (kinds, solids, faces, counts, errors) and floating-point results within a
tolerance; with no turning solid the expected difference is zero (@sec-conventions).

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

/ Flat solid: fills $p_z <= 0$. It is given by a point on its surface and its `normal_out_of_solid`. Its own z axis
  is the normal, and its own x axis is the cross product of the world axis on which the normal has its smallest
  part with the normal, normalised; a plate builds its axes the same way. Its one face is `kHighZ`.
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

where a domain box's origin is its centre and its half size is half its extent. The sphere and the cylinder use
their true distance to the surface (@eq:sphere-distance, @eq:cylinder-distance). The normal,
`normal_out_of_solid`, belongs to the face that gives $d$, written back in world axes with $A$. A flat solid's
normal is its own +z. A plate's is +z with face `kHighZ` when $p_z >= 0$, else −z with `kLowZ`. A box's face is
the one on the axis that gives the maximum: the low face when $p_k <= 0$, else the high one. A domain box's face is
the closed side that gives the minimum, with its normal pointing into the box. The face order is `kLowX`, `kHighX`,
`kLowY`, `kHighY`, `kLowZ`, `kHighZ`, `kCurved`, `kLowRim`, `kHighRim`. When two faces of a box or a domain box
give the same value, the first in that order is taken; so a box's middle plane $p_k = 0$ gives its low face. Two
other shapes settle their ties by their own rules instead: a plate and a cylinder's caps take `kHighZ` on their
middle plane $p_z = 0$ (the rule above and @sec-round-solids), and a tie inside a cylinder goes to its wall.

A box's $d$ is the gap to its nearest face plane, not the straight-line distance to its surface. The two agree
everywhere except outside an edge or a corner, where $d$ is smaller. For a box with half sizes (1, 1, 1) at the
origin, the point (3, 0, 0) is at $d = 2$ from face `kHighX` with normal (1, 0, 0); the point (0.9, 0, 0.5) is
inside at $d = -0.1$, again from `kHighX`; the point (−2, −2, 0) is at $d = 1$ from two faces at once, and the
tie gives `kLowX`. For a domain box from (0, 0, 0) to (1, 1, 1) with an open top, the point (0.5, 0.5, 0.95) is at
$d = 0.5$: four sides tie and left wins. With the top closed it would be at 0.05 from the top.

=== The two sides of a plate

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
classified, in the sweep, when the particle settles and in the end check. A committed particle that starts inside
the thickness is therefore pushed out to its own side, never through to the other one. For a plate of half
thickness 0.01 m and a `start_overlap_tolerance` of 0.05 m (the tests' values), a particle committed to
`kAlongNormal` at $p_z = -0.001$ m is at $d = -0.011$ m and goes to $p_z
= 0.01 + "tol"$; an uncommitted particle at the same place is nearer to `kLowZ` and goes to $p_z = -(0.01 +
"tol")$. A committed particle more than $"tol"$ beyond the other face, $s_"plate" p_z < -("half_thickness" +
"tol")$, is on the wrong side of its plate, an error. This is checked before the depth inside the half space: a
particle on the other side is not inside its half space but on the wrong side, and the error says so. A `plate_id`
that names no plate among the solids is a bad input.

=== Checks before the step

`CheckSolid` checks one solid: finite values and a step duration $T > 0$; axes that form a rotation, each column a
unit vector up to a squared length within $2^(-40)$ of 1, each pair of columns at right angles up to a dot product
of at most $2^(-40)$ in absolute value, and a positive determinant, so the axes are not a mirror image; positive
sizes, with a sphere's three half sizes equal and a cylinder's first two equal (both are stored in `half_size`: the
radius three times for a sphere, the radius twice and the half height for a cylinder); a nonzero id; a
`friction_scale` of zero or more; a domain box that does not move, keeps exactly the world axes ($A = I$) and has at
least one closed side; and a turn of at most a quarter turn during the step, $|bold(omega)| T <= pi\/2$. A larger
turn asks for a shorter step. For the sphere, the cylinder, the flat solid and the plate only the turn that changes
the shape counts (@sec-round-solids, @sec-first-hit): a sphere never fails this check, a cylinder may spin about its
own axis at any rate, and a flat solid or a plate about its normal. The
step calls it once per step for every solid; `MoveParticle` trusts that call and does not repeat it.

== The contact tolerance <sec-contact-tolerance>

A computer evaluates $d$ with rounding. A particle placed exactly on a surface, at $d = 0$, may later be computed
at $d = -10^(-17)$ m by another module with its own order of operations, and so be found inside the solid. The
proof of concept guarded against this with error bounds around every sign and by moving velocities by one
representable step. This code uses one length instead, the contact tolerance:

$ "tol" = 2^(-30) w_"min" + 2^(-40) L $ <eq:contact-tolerance>

where $L$, the `largest_coordinate`, is the largest $|x|$, $|y|$ or $|z|$ a solid origin takes during the step, plus
each solid's reach from its origin, and $w_"min"$, the `smallest_cell_width`, is the width of the smallest cell of
the grid
(`ComputeContactScale`, `ComputeContactTolerance`). The reach is the length of the half size of a box, a sphere, a
cylinder or the domain box; the domain box lets the particles' extent count, since every particle is inside it.

Why $2^(-40)$: write $epsilon = 2^(-52)$ for the machine epsilon, the gap between 1 and the next double. Computing
$d$ takes a subtraction of the origin, a rotation made of three-term dot products and the shape's own formula.
Together they misplace $d$ by at most about $32 epsilon L$. The relative term
$2^(-40)$ is $4096 epsilon = 128 times 32 epsilon$, so $"tol"$ is at least 128 times that error. The term in the
cell width keeps $"tol"$ positive for a scene at the origin.

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
  tolerance is never enlarged to make contact pass: a scene it cannot protect is refused.

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

=== A solid that does not turn

A box turns when its angular velocity is not exactly (0, 0, 0). A flat solid or a plate turns when the part of its
angular velocity across its normal, $bold(omega) - (bold(omega) dot bold(e)_z) bold(e)_z$ with $bold(e)_z$ the normal,
is longer than $2^(-46) |bold(omega)|$: both are unbounded sideways, so a spin about the normal leaves the shape as it
is, and a part across that short is the rounding a spin about a tilted normal leaves, not a turn. Such a part keeps
the normal within $2 times 2^(-46)$ of its start however far the solid spins, so the face moves by less than
$"tol"\/8$ at any point within $2 sqrt(3) L$ of its origin, $L$ the largest coordinate of the contact scale
(`kNegligibleTurnShare`; the proof is in the code). The spin
still moves the surface, and friction feels it through the solid velocity. A sphere never counts as turning, and a
cylinder only when its axis tilts (@sec-round-solids). One that does not turn only translates, so each of its face
planes moves at $bold(v)_"lin"$, and the gap to a face changes linearly with time:

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

=== A particle resting on a surface

After every hit the particle sits on the surface at $d = "tol"$ and is swept again for the rest of the step. If that
surface counted as hit again at once, every slide would end after a few steps in "too many hits". So the surface a
particle rests on is treated apart, by the touching rule: when the start's gap $g(0)$ is in $[-"tol", 0]$ for the
surface that gives $d$, that surface is not hit when $"rate" times "duration" >= -"tol"$, and it is hit at $t = 0$
otherwise. The surface that gives $d$ is: for a plate or a box, the face with the largest gap; for a flat solid or a
closed side of a domain box, its one face. The reason: the solid lies entirely behind that face plane (true for a
plane, a convex solid and a domain side alike), so the gap along the path is at least $g(0) + "rate" times t >=
g(0) - "tol"$. `MoveParticle` starts every sweep at $d = "tol"$ (a hit or the start move put it there, a rim hit
within $"tol"\/1024$ of it), so $g(0) = 0$ and the path never goes below $d = 0$; the part of the window below 0 only
absorbs rounding.
A sphere and a cylinder follow the same rule at the nearest point of their surface (@sec-round-solids). The
touching rule belongs to the closed forms of a solid that does not turn; a turning solid has none (next part).
A start deeper than $"tol"$ inside the grown solid, $g(0) < -"tol"$, is hit at $t = 0$; `MoveParticle` never sweeps
from there. For a flat solid with surface at $z = 0$, a particle at $z = "tol"$ moving at $(5, 0, -"tol"\/4)$ m/s for
1 s slides freely; at (5, 0, −1) m/s it hits at 0.

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
"tol"$ only up to rounding, possibly a few rounding units outside, and the response leaves an approach of a few rounding
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

=== A turning solid

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
second-order advance passed a wall on 27.5 % of measured paths. Such a shape keeps the first-order advance alone
(`HasConvexDistance`). The rounding margin is the first-order one: the distance is off by at most $"tol"\/128$, and
$dot(g)$, $K$ and the root carry relative rounding only, so the path stays farther than $"tol" - "tol"\/64$ before
the hit.

The search stops with a hit when $g(t) <= "tol"\/4$, and with no hit when $t >= "duration"$. More than 128 advances
(`kMaxAdvanceCount`) is "too close to call, slow approach to a turning solid", and the step is tried again shorter.
With the second-order advance this is rare: it remains for a path that hovers near the solid or grazes it, with almost no
motion along its normal (a graze is the same thing near the tangent point), where each advance is only about $sqrt(2 g\/K)$. A particle at rest $2^(-17)$ m above the top face
of a box of half size 0.5 m turning at (0, 0, 1) rad/s, on its turn axis, never gets closer, but needs 182 advances
to reach the end of a 1 s sweep. On the generated paths of the tests, 30,000 sweeps toward turning boxes, plates, flat
solids, cylinders, a quarter-turn arm and a paddle, there was no close call. A box that turns uses this search, and
so do a flat solid or a plate that turns about an axis across its normal and a cylinder whose axis tilts; a sphere,
a cylinder that only spins about its own axis, and a flat solid or a plate that only spins about its normal keep
their closed forms (@sec-round-solids).

The touching rule does not apply to a turning solid. A particle resting on one, at $d = "tol"$, starts with $g(0)
= 0 <= "tol"\/4$ and is hit at $t = 0$ in every step; the response and the slide in the solid's axes
(@sec-contact-response) then carry it with the solid for the rest of the step. This is intended: the face it rests
on is turning, so the plane argument of the touching rule does not hold, and carrying the particle in the solid's
axes is what keeps it on the face.

Nor does the argument that keeps a touched solid from being hit again. For a solid that does not turn, the sweep
after a response reads the rate the response targeted, bit for bit, at the same point and pose. Toward a turning
solid the surface velocity at the particle, $bold(v)_"lin" + bold(omega) times bold(r)$, changes as the particle
slides, and the normal turns, so a rate read where a sweep starts says nothing about the gap later; and the search
hits the solid at 0 from any gap up to $"tol"\/4$, whatever the velocity. Before the slide was built, every sweep
after a hit on a turning solid hit it again at 0, three times, and the fifth hit of the step ended in "too many
hits". The slide in the solid's axes replaces that argument: a hit on a turning solid ends the sweeps of the step,
so the solid is never swept again from the moment it was hit, and a hit at 0 just after a response, on a turning
solid within $"tol" + "tol"\/16$, is the moment of that response, which read the solid with the same distance at the
same point and pose: no second response, one hit. In the solid's own axes the solid stands still and its distance
is convex, and the slide moves the particle on a straight line there that does not approach along the normal the
response read, so the distance stays at least its value at the hit for the whole time left, exactly: no rate, no
allowance and no curvature bound enter. What is not exact is rounding: the turns into and out of those axes, below
$"tol"\/64$, and the normal written in them, a few rounding units on a face; beside a cylinder's rim the normal is
read from a vector about $"tol"$ long, so it turns by the own point's rounding over $"tol"$, about $10^(-7)$ rad at
$"tol" = 2^(-30)$ m and coordinates near 1 m. Measured on 12,000 generated moves next to turning boxes, flat solids,
plates, cylinders and a paddle, the slide ended no nearer than $0.999998 "tol"$ before the settle.

A worked example, from the plan of the tests: a box of half size 0.5 m at the origin turns at (0, 0, 1) rad/s, and a
particle rests at $0.6 (cos 0.3, sin 0.3, 0)$ m, at $d = 0.073$ m from face `kHighX`. In the box's axes the particle
goes round at radius 0.6 m, so the bound is 0.6 m/s and $K = 1 times (0 + 1 times 0.6) = 0.6 "m/s"^2$, and the face
reaches it when $0.6 cos(t - 0.3) = 0.5 + "tol"$, at $t = 0.3 + arccos((0.5 + "tol")\/0.6) approx 0.8857$ s. With
$"tol" = 2^(-30)$ m, the search takes 3 advances and stops at 0.88568554050660 s, $1.4 times 10^(-10)$ s before that
time, within the test's $2^(-30)$ (this paper's count, from the code's run). The first-order advance alone takes 30.

The solids that do not turn are searched first, then the turning solids, each only up to the earliest hit found so
far: a later hit could not win, and a turning search that reaches that time is free, since the path stayed at $d >=
"tol"$ up to it. So a close call beyond an earlier hit is never reached. Of two turning solids that fill the same
space through the step (same shape, size, pose and motion, bit for bit), only the first in array order is searched:
the second, capped at the first one's hit, would sample other times and stop elsewhere within the hit gap, so a copy
would change the hit.
When several solids are hit, the earliest time wins, and equal times keep the lower solid index.

#procedure(
  [`FindFirstHit`, the first hit of a straight path],
  <proc-find-first-hit>,
  inputs: [The start $bold(x)_0$, `velocity_for_this_step` $bold(v)$, `start_elapsed` $t_0$ and `duration`, all
    finite; the solids, the contact tolerance ($>= 0$; 0 gives the plain surface) and the particle's plate side.
    The inputs are not checked here.],
  outputs: [`FirstHit`: the time since $t_0$, the index of the solid hit (`kNoSolid` when the path is free), and the
    distance, normal and face at the hit.],
  fails: [a search toward a turning solid needs more than 128 advances: "too close to call, slow approach to a
    turning solid".],
  ([Solid that does not turn], [First, for each solid in array order that does not turn (a box with angular
    velocity exactly zero, a flat solid or plate whose angular velocity has no part across its normal longer than
    $2^(-46) |bold(omega)|$, so one that at most spins about its normal, any sphere, a cylinder whose axis does not tilt), sampled at $t_0$: if the
    start's gap is in $[-"tol", 0]$ for the surface that gives $d$, the solid is hit at 0 when $"rate" times
    "duration" < -"tol"$ and not hit otherwise; if the gap is below $-"tol"$, it is hit at 0. Otherwise use the
    closed form of its shape: one face plane for a flat solid, a committed plate or each closed domain side; entry
    and exit times over the faces for a plate or a box; @proc-sphere-first-hit for a sphere;
    @proc-cylinder-first-hit for a cylinder. Keep the time if it is at most `duration`, with the face at the hit.]),
  ([Turning solid], [Then for each solid in array order that turns, but one that fills the same space as an
    earlier one, with `duration` capped at the earliest hit so far: from $t = 0$, repeat $t <- t + "advance"$ with
    $g$ measured at the pose of $t_0 + t$, the advance the larger
    of $g(t)\/"bound"$ and the positive root of $g + dot(g) s - K s^2\/2$ (@eq:second-order-advance) when the
    distance is convex (every turning shape here), $g(t)\/"bound"$ otherwise. Stop with a hit at $t$ when $g(t) <=
    "tol"\/4$, with no hit when $t >= "duration"$, and fail after 128 advances. There is no touching rule here: a
    particle resting on the solid is hit at 0.]),
  ([Keep the earliest], [Return the hit with the smallest time; equal times keep the lower solid index, whatever the
    order of the search. With no hit,
    return `kNoSolid`.]),
)

== Round solids: sphere and cylinder <sec-round-solids>

A sphere and a finite cylinder are round, so the gap to them is not a face gap and does not change linearly along
a path. Contact gives them their true distance to the surface, a closed form for the first hit of a path, and
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

=== First hit of a path

The path meets the solid grown by $"tol"$. Here $bold(q)$ is the start relative to the solid's origin at time $t_0$
and $bold(w) = bold(v) - bold(v)_"lin"$ the velocity relative to the solid, both turned into the solid's own axes.

*Sphere.* The grown sphere has radius $R = "radius" + "tol"$, and the path meets it where $|bold(q) + bold(w) t| =
R$. With $alpha = bold(w) dot bold(w)$, $beta = bold(q) dot bold(w)$ and the gap at the start $g_0 = |bold(q)| -
"radius" - "tol"$,

$
  gamma = g_0 (|bold(q)| + R), quad t_"hit" = gamma / (-beta + sqrt(beta^2 - alpha gamma)).
$ <eq:sphere-hit-time>

$gamma$ equals $bold(q) dot bold(q) - R^2$, but computed from the gap its sign is exactly the sign of the gap that
the touching rule below reads, so the two can never disagree about a start on the grown sphere. There is no hit when
$alpha = 0$, when the path does not approach the centre ($beta >= 0$), or when the root does not exist ($beta^2 <
alpha gamma$); otherwise the hit is kept when $t_"hit" <= "duration"$. The root equals the textbook $(-beta -
sqrt(beta^2 - alpha gamma))\/alpha$, but it never divides by a small $alpha$ and never subtracts two nearly equal
numbers. For a fixed sphere of radius 0.5 m at the origin and a particle from (−2, 0.3, 0) at (2, 0, 0) m/s (this
paper's numbers): $alpha = 4$, $beta = -4$, $gamma = 4.09 - R^2$, and $t_"hit" = 1 - sqrt(R^2 - 0.09)\/2$, which
is $0.8 - 0.625 "tol"$ s. The hit point is (−0.4, 0.3, 0) less $1.25 "tol"$ along x, with normal (−0.8, 0.6, 0).

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
the grown wall, $d_"wall" - "tol" > 0$, the interval is the one written above, and it is empty when the path does not
move toward the axis ($alpha_rho = 0$ or $beta_rho >= 0$) or the root does not exist ($beta_rho^2 < alpha_rho
gamma_rho$). Otherwise the path is inside the grown wall from $t = 0$ until the larger root, written $-gamma_rho \/
(beta_rho + sqrt(beta_rho^2 - alpha_rho gamma_rho))$ when $beta_rho > 0$ so that nothing cancels, or for ever when
$alpha_rho = 0$. The caps' interval is the one written above when the start is beyond a grown cap, $d_"cap" - "tol" >
0$ with $sigma_0$ the sign of $q_z$, and empty unless the path moves toward that cap ($sigma_0 w_z < 0$); otherwise
the path is between the grown caps from $t = 0$ until it reaches the grown cap it moves toward, or for ever when
$w_z = 0$. A start the distance puts beyond the grown wall or a cap is beyond it here, bit for bit, since both read
the same computed gaps. The hit is the later entry, provided it comes no later than either exit: the same test of
entry and exit as for a box. Both limits count, so a path that meets the sharp rim edge exactly is a hit. A path
that goes into the cylinder through the rim edge leaves both regions long after it enters them, so the rounding of
the two entry times can change which one is later but cannot make the path miss; only a path that grazes the edge
sits on the limit, where a hit and a miss are both safe. (Accepting each entry only where the path is inside the
other region at that time is the same in exact arithmetic, but there a path through the edge can have both entries
rejected by rounding and pass through the solid.) The face, normal and distance reported are those of
@eq:cylinder-distance at the hit position.

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

=== Starts on or near a round solid

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
  ([Find the root], [Compute the three coefficients of @eq:sphere-hit-time, the constant from the gap. No hit when
    the velocity is zero, the path does not approach the centre or the root does not exist.]),
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
  ([Bounding surfaces], [Compute the true distance at the start. For a start beside a rim, read the wall,
    unbounded along the axis, and the cap's plane on the start's side: no hit when the start rests on either, a gap
    in $[-"tol"\/8, "tol"\/16]$ sinking at most $"tol"\/64$ during the sweep, whatever the distance's gap.]),
  ([Read the start], [The gap $g_0$ of the true distance. Below $-"tol"$: hit at 0. In $[-"tol", 0]$: the touching
    rule.]),
  ([Inside the wall], [Compute the times the path is inside the grown wall, unbounded along the axis, with
    @eq:cylinder-hit-time, the constant from the gap: from the first root to the second when the start is beyond
    the grown wall ($d_"wall" - "tol" > 0$), empty when it does not approach the axis or the root does not exist;
    from 0 to the larger root, or for ever, otherwise.]),
  ([Between the caps], [Compute the times the path is between the grown caps' planes: from $(d_"cap" - "tol") \/
    (-sigma_0 w_z)$ to the far grown cap when the start is beyond a grown cap, empty when it does not move toward
    that cap; from 0 to the grown cap it moves toward, or for ever, otherwise.]),
  ([Find the entry], [The later of the two entries, when neither interval is empty and it comes no later than
    either exit and no later than `duration`; no hit otherwise. A start in a rim corner has the entry 0.]),
  ([Keep the hit], [An entry on the wall or a cap proper, as @eq:cylinder-distance reads the point there, is the
    hit. Beside a rim, the rim search from the entry: advances of $g\/(-dot(g))$ on the true distance until $g <=
    "tol"\/1024$ (a hit), $dot(g) >= 0$ or the end of the sweep (no hit), at most 16.]),
)

=== Slide and spin

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
it then uses the stepping search of @sec-first-hit and the slide in its own axes of @sec-contact-response, and the
quarter-turn limit applies to $|bold(omega)_"across"| T$. A spin given along the axis stays below that bound up to
rounding. Below the bound the closed form keeps the axis of the sweep's start, while $bold(u)$ keeps the tilt, and the
touching rule reads the surface's normal speed where the sweep starts. A particle that slides across the tilting surface
keeps that speed while the surface under its new position moves the other way, so the gap falls by up to $2
|bold(omega)_"across"| max|bold(r)| T < "tol"\/4$, since $max|bold(r)| <= |"half_size"|$; a sweep the closed form
decides keeps the $"tol"\/8$. So the true grown shape is entered by up to $"tol"\/4$ without a hit, outside the
$"tol"\/16$ of @sec-first-hit; no hit into the solid is missed, the particle ends at least
$"tol" - "tol"\/16 - "tol"\/4 = 0.6875 "tol"$ from it, above the end check's $"tol"\/2$, and the settle lifts it
back (measured: 0.759 tol before the settle, for a cylinder of radius 0.1 m and half height 2 m at 0.99 of the
bound, the tests' numbers). Measured from $bold(v)_"lin"$ alone, a cap sinking at $3 times 10^(-10)$ m/s under a
particle it had just stopped looked approached, beyond any rounding bound, and was hit again at 0 until "too many hits",
at every step length. Why not treat a spinning roller as turning: the slide on a turning solid carries the particle in
the solid's axes, so it would drag a particle round the roller even with friction 0. Through the solid velocity, the
spin acts by friction only. For a cylinder of radius 0.5 m at the origin spinning at (0, 0, 10) rad/s, a particle
hitting the wall at (−0.5, 0, 0) with velocity (1, 0, 0) m/s meets a surface moving at (0, −5, 0) m/s: the approach
speed is −1 m/s, and the particle keeps (0, 0, 0) m/s with friction 0 and takes (0, −0.5, 0) m/s with friction 0.5 (this
paper's numbers, by @eq:contact-response).
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
afterwards. That round trip would erase
a small sideways velocity next to a fast solid. With friction 0 at a flat solid with normal (0, 0, 1) sliding sideways
at $10^8$ m/s, a particle at $(10^(-9), 0, -1)$ m/s keeps its x part of $10^(-9)$ m/s in world coordinates, while the
round trip computes $(10^(-9) - 10^8) + 10^8 = 0$, since doubles near $10^8$ are $1.5 times 10^(-8)$ apart (this paper's
example). Second, the stick test compares $mu$ with the quotient, $mu >= s_"side"\/(-a)$, not the
product with the sideways speed, $mu (-a) >= s_"side"$. The two forms round differently at the limit: with sideways speed 3 m/s and approach −0.7 m/s,
the friction $3\/0.7$ rounded to a double sticks by the quotient, while the product rounds to $2.9999999999999996 < 3$
and would slip at $4.4 times 10^(-16)$ m/s. Overflow is no reason for either form: an infinite product still compares
correctly.

Each of the particle's two velocities gets the response with its own approach speed. A particle whose
`for_this_step` is (0, 0, −1) m/s and whose `for_next_step` is (1, 0, −3) m/s meets a fixed flat solid with
friction 0.5: `for_next_step` has approach −3 m/s and sideways speed 1 m/s, and $0.5 >= 1\/3$, so it sticks at
zero; `for_this_step` has approach −1 m/s and no sideways speed, so it too becomes zero. The record keeps both
approach speeds, −3 and −1 m/s.

=== Moving one particle

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
point, still at the rest gap of both surfaces. The slack includes the rounding of composing $bold(w)$, which grows
with the multipliers, so no iterate is trusted: the last one is kept only after its true distances are read. Under a ball of
radius $R$ resting on a floor, a *tangent contact*, the free space is a *cusp* whose angle shrinks to 0 at the
contact. The first iterate from deep in the cusp overshoots out along the floor, where the two tangent planes meet;
the next ones come back to the rest point $2 sqrt(R "tol")$ from the contact, both distances exactly $"tol"$
(@fig-move-off-near-surfaces, right): 5 iterates at $R \/ "tol" = 2^29$, within $10^(-4)$ of it after the
largest number, 8, at $R \/ "tol" = 2^40$. The iterates stop when one moves less than $"tol" \/ 64$ and is within
$"tol" \/ 1024$ of the rest gap of every solid; a surface that an iterate comes within $"tol"$ of joins the next
one. The segment from $bold(x)_0$ to each iterate is swept against every other surface, so a long move cannot pass
through a thin solid: a move that would cross one stops where it first comes within $"tol"$ of it, and that surface
joins the next iterate, linearised there. Surfaces with the same normal share one equation, the most demanding. The
particle is displaced without a matching velocity, so each change is shortened to at most $2^22 "tol" = h \/ 256 + L \/ 2^18$ (a 256th of a cell plus 4 micrometres in a one-metre scene), which covers
the rest point of every tangent contact, $2 sqrt(R "tol") <= 2^21 "tol"$. The last iterate is kept when every solid
is at least $"tol" - "tol" \/ 128$ from it by its true distances. A start with no such point within that length is
the scene error "starts in narrow gap": two solids facing each other less than $2 "tol"$ apart, a start within
rounding of the contact point of a ball on a floor, a pocket too small for a point at $"tol"$ from each side, or a
start deep at the apex of a flat wedge sharper than about $2^(-21)$ rad. The grown solids overlap there, so a sweep
stops a moving particle before it, and a shorter step does not help. The same error names a start near surfaces
facing more than four different ways, which one move cannot read at once; surfaces with the same normal, such as a
ground lying on the domain's bottom or floors stacked closer than $"tol"$, share one target, the most demanding, up to
16 surfaces in all. A known limit stands beside the start near the contact point: a wall within $"ulp"(R) \/ (2 sqrt("tol" \/ R))$
of a cusp's rest point (0.0014 tol for a ball of 0.5 m, 128 tol for 1 km), where the ball's rounded distance moves
the linearised vertex past the wall, so no subset of the targets has an answer although the rest point fits. The
failure names the near solid latest in array order.

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
particle moving away at (1, 0, 2) m/s from $(0, 0, "tol"\/4)$ is moved to $z = "tol"$ silently and ends at
$(0.1, 0, 0.2 + "tol")$.

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
changes the result by rounding only, at most $2.2 times 10^(-14)$ of the speed. Surfaces whose normals agree within
$2^(-40)$ on every axis share one *direction*, with the normal of the first.

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
gives $(0, 1, 0)$; the responses one surface after the other gave $(-0.25, 1, 0.433)$ or $(0.183, 1, 0)$ by which
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
- several, moving differently: the mean, over every order of the used directions (two or six), of their own
  frictions applied one after the other from $bold(v)_0$, each taking $min(L_k, "sideways speed")$ off the sideways
  velocity relative to its own solid at that moment. The mean does not depend on any order; when the solid velocities
  differ only by rounding it agrees with the case above to $1.7 times 10^(-14)$ of the speed. It approximates the exact answer of the same
  friction model: median 0.046, p90 0.19, p99 0.37 and at worst 0.66 of the largest speed over 2,418 draws with
  friction up to 2 and solids up to 1 m/s apart. Adding each surface's own friction from the same $bold(v)_0$ instead
  reverses the sliding when two solid velocities are one rounding unit apart. A known limit of the approximation: each step
  stops at sticking to its own solid at that moment, but a later step of the other surface moves the velocity on and
  the mean keeps both, so a surface's own friction can push the particle past sticking to its solid, reversing the
  sliding relative to it: in 1,331 of 24,401 responses that used the mean in a review probe (5.5 %), by up to 0.22 and
  at p99 0.072 of the summed speeds. The exact answer for solids that move differently is left to the owner.

The result is $bold(v)_0 + bold(F)$ when it approaches no active direction beyond its bound. Otherwise, and always
when the solids move differently, a second frictionless change is made from $bold(v) + bold(F)$, not from
$bold(v)_0 + bold(F)$: the pushes are found afresh, so friction lifts a particle off a surface only when it overcomes
what pressed it there. Beside a floor, a wall rising at 2 m/s and a particle pushed into both at 1 m/s each: friction
0.5 leaves (0, 0, 0), friction 1.5 leaves (0, 0, 0.5); from $bold(v)_0 + bold(F)$ friction 0.5 would lift it at
0.5 m/s. Where normals depend on each other (three faces sharing one line, coincident faces) the smallest change has
several sets of pushes, and the friction, which reads them, would jump with the rounding that picked one (up to 0.70
of the speed for a $10^(-13)$ change of the velocity). So among the changes no longer than the shortest plus its bound
the solver keeps the one with the largest $sum_k mu_k lambda_k$, the largest value of a linear function over the
admissible pushes, which varies continuously with the velocity. And the solver's "no change" reads the bound so far,
not its own: a real approach of $10^(-13)$ m/s beside a floor sliding at 100 m/s is removed.

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
friction; the geometry and the speeds decide, so a shorter step does not help, and the step module decides what a
lasting squeeze does. A squeeze names the solid the failing velocity approaches fastest. The response's bound,
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

A slide on a turning solid is carried in the solid's own axes, as the proof of concept did. A sphere, a cylinder
that only spins about its own axis, and a flat solid or a plate that only spins about its normal, are not turning
solids here (@sec-round-solids, @sec-first-hit). After the response, the relative
`for_this_step` at the hit, $bold(v) - bold(u)(bold(x))$ with $bold(u)$ the surface velocity at the hit position and
pose, is turned into those axes with the solid's axes at the middle of the time left, $bold(v)_"local" =
A_"mid"^T (bold(v) - bold(u)(bold(x)))$; its part into the face, along the normal of the hit written in own axes, is
set to zero; the particle advances in those axes for the time left $t_"left"$, $bold(p) + bold(v)_"local" t_"left"$,
a straight line there; and it is placed with the solid's pose at the end of the step. Both velocities stay as the
response made them. Why the middle axes: the solid carries the surface's motion at the particle exactly (the own
point stays), and the relative motion runs along a line in world axes while the solid turns under it. Turned with the
axes of the hit, a frictionless particle at rest on a turntable drifted out by $r a^2\/2$ a step, $a = |bold(omega)|
t_"left"$ the turn (6.4 % of $r$ over 50 steps of 0.1 s at 0.5 rad/s); taking the world line for the whole
velocity moved that drift to a stuck particle, and added energy. With the middle axes, the midpoint rule of the
turn, a stuck particle keeps its own point exactly, and a frictionless one at rest on a face turning about its normal
ends at radius $r (1 + a^4\/48 + dots)$, $r a^3\/24$ behind its start: third order a step, though not always
negligible, +2 % of $r$ a step at $a = 1$ rad, near the largest turn a step allows. The path's mean velocity agrees
with the velocity the
response returned to first order in $a$. Turned with the middle axes, a relative velocity tangent at the hit may
point into the face by up to $sin(a\/2)$ of itself when the face turns toward that line; the clamp keeps that part
out. When the face turns away from the line, that part points out and is kept, a known choice: the particle ends up
to $sin(a\/2) |bold(v) - bold(u)| t_"left"$ off the face, as a free flight in the solid's axes, while the velocity
the response returned has no part along the normal (a top face tilting at 0.5 rad/s under a sliding particle: up to
41,617 $"tol"$ at $T = 1\/120$ s, $5.3 times 10^(-3)$ m at 0.1 s). It does not build up from step to step, since the
next sweep and gravity bring the particle back, and the path converges at first order in the step. The proof of
@sec-first-hit is unchanged: the carry is still a straight line in own axes with no part along
the hit's normal into the solid. The cost is one more pose. The hit loop ends there; why no later sweep may meet
the solid again is in @sec-first-hit. Before that, every other solid $j$ is tested: the call is "too close to call,
turning solid near another solid", naming the turning solid, when

$ d_j - "tol" <= t_"left" (|bold(v)_"local"| + |bold(v)_"lin"| + |bold(omega)| r + |bold(v)_("lin", j)| +
|bold(omega)_j| r_j), $

with $d_j$ the distance to solid $j$ at the hit, $bold(v)_"lin"$, $bold(omega)$ the turning solid's motion,
$bold(v)_("lin", j)$, $bold(omega)_j$ solid $j$'s, and $r$, $r_j$ the distances from the hit position to each
solid's origin. The right side bounds how far the particle and solid $j$ can close in $t_"left"$. The slide puts the
particle at $bold(x)(s) = bold(o) + bold(v)_"lin" s + R(s) A (bold(p) + bold(v)_"local" s)$, $R(s)$ the turn by
$bold(omega) s$, so $bold(x)(s) - bold(x)(0) = bold(v)_"lin" s + (R(s) - I) A bold(p) + R(s) A bold(v)_"local" s$; a
turn by an angle $a$ moves a vector by at most $2 sin(a\/2) <= a$ times its length, so the particle moves by at most
$s (|bold(v)_"lin"| + |bold(omega)| r + |bold(v)_"local"|)$. In solid $j$'s axes the same argument adds $s
(|bold(v)_("lin", j)| + |bold(omega)_j| r_j)$, and a distance changes no faster than the point it is read at. The
bound is on the displacement, not on the length of the curved path, so no term in $t_"left"^2$ is needed. A solid
that fills the same space as the turning solid through the step is left out (the slide keeps the particle out of the
one, so out of the other), and with no time left nothing moves and nothing is tested. A box of half size 0.5 m
turning at (0, 0, 1) rad/s on a fixed floor that touches its bottom, and a particle at rest on its side 0.05 m above
the floor: hit at 0, $|bold(v)_"local"| = (0.5 + "tol") cos(t_"left"\/2)$ and $r = 0.6727$ m, so the test fails for
$t_"left" >= 0.0426$ s: a step of 0.1 s is too close to call, one of 0.04 s slides. Known limit: two flush boxes
turning together cannot
be crossed at their seam, since a turning solid has no bounding surface and the slide sweeps nothing; within
$"tol"$ of both, every step length is too close to call.

At the end the particle moves once more off every surface it is closer to than $"tol"$, at the end pose, as at the
start: every solid, touched or not, since a move off the touched ones alone could push it into another (a wall beside
a ball), and since a sweep may have brought it within $"tol"$ of a solid it never touched. When there
is no such point the particle stays. Then the end check of @sec-contact-tolerance runs against every solid. On any failure the result is left unchanged, and the failure
context names the solid by its stable `solid_id`, never by its index in the array, with a length: the depth $-d$ for
a start too deep inside, the distance $-(s_"plate" p_z + "half_thickness")$ beyond a committed plate, the end
distance $d$ for an end too close, and 0 otherwise. `MoveParticles` adds the particle's id.

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
  inputs: [`start_position`, both velocities, the solids in ascending `solid_id`, `step_duration`, `friction`,
    `start_overlap_tolerance`, `contact_tolerance` and the particle's `plate_side`.],
  outputs: [The end position, both velocities after the response, the kind, the record of the first hit, the hit
    count and the reaction.],
  fails: [a bad input; a start too deep inside a solid, on the wrong side of its plate, or in a narrow gap; too close
    to call, for one of the six reasons named in the text. The result is then unchanged, and the failure names the solid by its
    `solid_id` (never its index) with a length.],
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
  ([Joint response], [Fail on a fifth hit. Move to the hit, nothing more. Read the active surfaces: those of every
    solid within $"tol" + "tol"\/16$ of the particle, except a solid passed at a bounding surface's level, normals
    read afresh, and the surface hit with its own normal; drop exact copies, sort by value, group into
    directions; fail with "too many hits" past eight directions or 16 surfaces. Fill the record if it is
    empty and count one hit. For each velocity: with one active surface, the response with that solid's velocity and
    `friction` times `friction_scale`; with several, when it approaches one beyond its bound, the smallest
    frictionless change, then the friction from each used direction's push (the single-surface response for one, the
    stick or slip toward one solid velocity, or the mean over the orders), then, when the result approaches an active
    surface or the solids move differently, a second frictionless change from the velocity plus the friction. With no
    change, stick to the surface approached fastest. Fail with "squeezed" when a stick still approaches a surface, a
    frictionless velocity is faster than 64 times the largest speed involved, or, with several surfaces, the bound
    times the time left exceeds $"tol"\/64$. Charge each push and friction share of `for_next_step` to its solid, in
    ascending solid index, when it reports its reaction; fail on a second reacting solid. Raise each velocity's
    approach bound. A solid that does not turn: go to step 3 for the time left. A turning solid (never a sphere, nor
    a cylinder that only spins about its axis, nor a flat solid or plate that only spins about its normal): skip the
    response when the hit is at 0 just after a response and within $"tol" + "tol"\/16$ (the same moment, read
    already); fail if another solid $j$ is within the closing bound of the text, carry the rest of the step in its
    axes, and go to step 5.]),
  ([Settle and check the end], [At the end pose, move the particle once to the rest gap of every surface of every
    solid closer than $"tol"$, as at the start; when there is no such point, leave it. Fail when any solid is at
    $d < "tol"\/2$; otherwise write the result.]),
)

// Every floating figure and procedure is placed before the references start.
#place.flush()

= References

#block[
  #set text(size: 9pt)
  #set par(justify: false, first-line-indent: 0pt, hanging-indent: 1.8em, spacing: 0.75em)
  #include "references.typ"
]
