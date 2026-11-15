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
floating-point values across particles, and nothing in it depends on the order in which threads finish. When
several particles fail, the one reported is the failure with the lowest particle id, found by an integer minimum.
On failure the trial outputs of `MoveParticles` hold unspecified values and the step does not commit them. The CPU
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
the one on the axis that gives the maximum: the low face when $p_k < 0$, else the high one. A domain box's face is
the closed side that gives the minimum, with its normal pointing into the box. When two faces give the same value,
the first in the face order `kLowX`, `kHighX`, `kLowY`, `kHighY`, `kLowZ`, `kHighZ`, `kCurved`, `kLowRim`,
`kHighRim` is taken.

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
turn asks for a shorter step. For the sphere and the cylinder only the turn that changes the shape counts
(@sec-round-solids): a sphere never fails this check, and a cylinder may spin about its own axis at any rate. The
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

A flat solid, a plate or a box turns when its angular velocity is not exactly (0, 0, 0); a sphere never counts as
turning, and a cylinder only when its axis tilts (@sec-round-solids). One that does not turn only translates, so
each of its face planes moves at $bold(v)_"lin"$, and the gap to a face changes linearly with time:

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
g(0) - "tol"$. `MoveParticle` starts every sweep at $d = "tol"$ (a hit, the rim move or the start move put it
there), so $g(0) = 0$ and the path never goes below $d = 0$; the part of the window below 0 only absorbs rounding.
A sphere and a cylinder follow the same rule at the nearest point of their surface (@sec-round-solids). The
touching rule belongs to the closed forms of a solid that does not turn; a turning solid has none (next part).
A start deeper than $"tol"$ inside the grown solid, $g(0) < -"tol"$, is hit at $t = 0$; `MoveParticle` never sweeps
from there. For a flat solid with surface at $z = 0$, a particle at $z = "tol"$ moving at $(5, 0, -"tol"\/4)$ m/s for
1 s slides freely; at (5, 0, −1) m/s it hits at 0.

=== A turning solid

A turning solid's gap is not linear in time, and its faces do not keep their direction. Contact steps toward it
instead (citation needed). The gap cannot shrink faster than

$ "bound" = |bold(v) - bold(v)_"lin"| + |bold(omega)| times r_"max", $

where $r_"max"$ is the larger distance from the path to the solid's origin at the two ends of the sweep. In the
solid's own axes the particle moves at $bold(v) - bold(v)_"lin" - bold(omega) times (bold(x) - bold(o))$, whose
length is at most the bound, and a distance changes no faster than the point it is measured from. The particle's
distance to the moving origin is largest at one end of the sweep, because both move in straight lines. Starting at
$t = 0$, the search repeats $t <- t + g(t)\/"bound"$. Each advance is safe: in that time the gap shrinks by at most
$g(t)$, so the path never passes through the solid between two samples. The search stops with a hit when $g(t) <=
"tol"\/4$, and with no hit when $t >= "duration"$. More than 128 advances (`kMaxAdvanceCount`) is "too close to
call, slow approach to a turning solid", and the step is tried again shorter. The proof of concept bracketed the
time by bisection over a stack of intervals; stepping forward needs neither. A flat solid, a plate or a box that
turns uses this search, and so does a cylinder whose axis tilts; a sphere, and a cylinder that only spins about its
own axis, keep their closed forms (@sec-round-solids).

The touching rule does not apply to a turning solid. A particle resting on one, at $d = "tol"$, starts with $g(0)
= 0 <= "tol"\/4$ and is hit at $t = 0$ in every step; the response and the slide in the solid's axes
(@sec-contact-response) then carry it with the solid for the rest of the step. This is intended: the face it rests
on is turning, so the plane argument of the touching rule does not hold, and carrying the particle in the solid's
axes is what keeps it on the face.

A worked example, from the plan of the tests: a box of half size 0.5 m at the origin turns at (0, 0, 1) rad/s, and a
particle rests at $0.6 (cos 0.3, sin 0.3, 0)$ m, at $d = 0.073$ m from face `kHighX`. In the box's axes the particle
goes round at radius 0.6 m, so the bound is 0.6 m/s, and the face reaches it when $0.6 cos(t - 0.3) = 0.5 +
"tol"$, at $t = 0.3 + arccos((0.5 + "tol")\/0.6) approx 0.8857$ s. With $"tol" = 2^(-30)$ m, the search takes 30
advances and stops $4 times 10^(-10)$ s before that time, within the test's $2^(-30)$ (this paper's count).

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
  ([Solid that does not turn], [For each solid in array order that does not turn (a flat solid, plate or box with
    angular velocity exactly zero, any sphere, a cylinder whose axis does not tilt), sampled at $t_0$: if the
    start's gap is in $[-"tol", 0]$ for the surface that gives $d$, the solid is hit at 0 when $"rate" times
    "duration" < -"tol"$ and not hit otherwise; if the gap is below $-"tol"$, it is hit at 0. Otherwise use the
    closed form of its shape: one face plane for a flat solid, a committed plate or each closed domain side; entry
    and exit times over the faces for a plate or a box; @proc-sphere-first-hit for a sphere;
    @proc-cylinder-first-hit for a cylinder. Keep the time if it is at most `duration`, with the face at the hit.]),
  ([Turning solid], [For each solid in array order that turns: from $t = 0$, repeat $t <- t + g(t)\/"bound"$ with
    $g$ measured at the pose of $t_0 + t$. Stop with a hit at $t$ when $g(t) <= "tol"\/4$, with no hit when $t
    >= "duration"$, and fail after 128 advances. There is no touching rule here: a particle resting on the solid
    is hit at 0.]),
  ([Keep the earliest], [Return the hit with the smallest time; equal times keep the lower solid index. With no hit,
    return `kNoSolid`.]),
)

== Round solids: sphere and cylinder <sec-round-solids>

A sphere and a finite cylinder are round, so the gap to them is not a face gap and does not change linearly along
a path. Contact gives them their true distance to the surface, a closed form for the first hit of a path, and
three rules of their own: a rim move, a touching rule on a curved surface and a spin rule (@fig-round-solids).

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
"half_height" + "tol"$. Two candidates are tried, each kept only inside the other's range:

$
  t_"wall" & = gamma_rho / (-beta_rho + sqrt(beta_rho^2 - alpha_rho gamma_rho)), \
  & "kept when" |q_z + w_z t_"wall"| <= H, \
  t_"cap" & = (|q_z| - H) / (-sigma_0 w_z), \
  & "kept when" (q_x + w_x t_"cap")^2 + (q_y + w_y t_"cap")^2 <= R^2.
$ <eq:cylinder-hit-time>

The wall time is the sphere's root computed with the x and y parts only: $alpha_rho = w_x^2 + w_y^2$, $beta_rho =
q_x w_x + q_y w_y$ and $gamma_rho = (rho_0 - "radius" - "tol") (rho_0 + R)$, again from the gap, with $rho_0 =
sqrt(q_x^2 + q_y^2)$. It is tried when the start is outside the grown radius ($rho_0 > R$), the path moves in the
x–y plane ($alpha_rho > 0$) and toward the axis ($beta_rho < 0$), and the root exists ($beta_rho^2 >= alpha_rho
gamma_rho$). The cap time is tried for the cap the start is beyond, $|q_z| > H$ with $sigma_0$ the sign of $q_z$,
when the path moves toward it ($sigma_0 w_z < 0$); its disc test compares squared distances from the axis, with no
square root. The earliest kept time at most `duration` is the hit; equal times keep the wall, since both name the
same point. This is the entry into the part shared by the grown slab and the grown unbounded cylinder, so no third
equation is needed for the rim: both tests include the sharp edge (the $<=$), and a path through it is caught. The
face, normal and distance reported are those of @eq:cylinder-distance at the hit position, so a hit in the sharp
corner reports a rim.

That hit sits between $"tol"$ and $sqrt(2) "tol"$ from the true rim, never closer than $"tol"$. Its gap $d -
"tol"$ is then between 0 and $0.414 "tol"$, a value that neither the touching rule ($[-"tol", 0]$) nor a start
deeper inside covers. `MoveParticle` therefore moves a hit that this closed form finds in a rim region along the
rim normal to $d = "tol"$ before the response, a move of at most $(sqrt(2) - 1) "tol"$ toward the rim. A hit found
by the stepping search toward a turning cylinder is not moved: it stops at $d <= 1.25 "tol"$ in any region. The end
check ($d >= "tol"\/2$) never fires wrongly there: before the move $d >= "tol"$, after it $d = "tol"$. For a fixed
cylinder of radius 0.5 m and half height 1 m and a particle from $(0.5 + "tol"\/2, 0, 3)$ at (0, 0, −1) m/s (the
tests' numbers), the wall is never tried ($alpha_rho = 0$) and the top cap is hit at $2 - "tol"$ s, at $(0.5 +
"tol"\/2, 0, 1 + "tol")$, inside the grown disc. There $d_"wall" = "tol"\/2$ and $d_"cap" = "tol"$: the point faces
the high rim at the true distance $sqrt(1.25) "tol"$, and it is moved along the rim normal $(1, 0, 2)\/sqrt(5)$ to
$(0.5 + "tol"\/sqrt(5), 0, 1 + 2 "tol"\/sqrt(5))$, at $d = "tol"$, face `kHighRim`.

The other examples of a cylinder of radius 0.5 m and half height 0.25 m at the origin (this paper's numbers): from
(0.2, 0, 1) at (0, 0, −2) m/s the top cap is hit at $(1 - 0.25 - "tol")\/2 = 0.375 - "tol"\/2$ s, at $(0.2, 0,
0.25 + "tol")$; from (−1.5, 0.3, 0) at (2, 0, 0.2) m/s the wall is hit at $0.55 - 0.625 "tol"$ s near (−0.4, 0.3,
0.11), with normal (−0.8, 0.6, 0). From the same start at (2, 0, 0.5) m/s the cylinder is not hit: the start is not
beyond a cap, so there is no cap candidate, and the wall root comes when the path is at height 0.275 m, above the
grown cap, so it is rejected. Seen as geometry, the path crosses the cap's height 0.25 m at $rho = 0.58$ m, outside
the disc: it passes over the rim.

#figure(
  round-first-hit-figure(),
  placement: auto,
  caption: [Paths against the grown shapes, the tolerance exaggerated. The sphere grows into a larger sphere; the
    cylinder grows with sharp edges, so a path entering the corner stops farther than $"tol"$ from the rim and
    is moved along the rim normal to the red arc, where $d = "tol"$. The red arrow at the sphere's hit is the
    normal there.],
) <fig-round-first-hit>

=== Starts on or near a round solid

The start's gap $g_0 = d - "tol"$ uses the true distance. A start deeper than $"tol"$ inside the grown solid, $g_0
< -"tol"$, is hit at $t = 0$, as for a flat face. With $g_0$ in $[-"tol", 0]$ the touching rule applies, with
$"rate" = bold(n) dot bold(w)$ and $bold(n)$ the normal at the start (a spin about the centre or the axis adds
nothing along $bold(n)$): the solid is not hit during this sweep when $"rate" times "duration" >= -"tol"$, and it
is hit at $t = 0$ otherwise. It is safe because the true signed distance to a convex solid is a convex function of
position, so along the straight path $d(t) >= d(0) + "rate" times t >= d(0) - "tol"$. `MoveParticle` starts every
sweep at $d = "tol"$ (a hit, the rim move or the start move put it there), so $d(t) >= "tol" - "tol" = 0$; the
part of the window below 0 only absorbs rounding, and the settle at the end of `MoveParticle` puts the particle
back at $"tol"$. It is the same tolerance as for a flat face, with no term for the curvature. A cylinder start
inside the grown sharp cylinder ($rho_0 <= R$ and $|q_z| <= H$) with a positive gap $g_0$ lies in a rim corner (up
to rounding, since these tests and the gap are computed separately); it is hit at $t = 0$ when it approaches
($"rate" < 0$) and free otherwise. For a sphere of radius 0.5 m at the origin, a particle at $(-(0.5 + "tol"), 0,
0)$ moving at (0, 1, 0) m/s, along the tangent, is free; moving at (1, 0, 0) m/s it is hit at 0 (the tests'
numbers).

#procedure(
  [The first hit of a straight path on a sphere],
  <proc-sphere-first-hit>,
  inputs: [The start relative to the centre and the velocity relative to the sphere, in its own axes; `radius`,
    the contact tolerance and `duration`.],
  outputs: [The hit time since the start of the sweep, or no hit. The normal and face come from
    @eq:sphere-distance at the hit position.],
  fails: [never.],
  ([Read the start], [Compute the distance to the centre and the gap $g_0$. Below $-"tol"$: hit at 0. In $[-"tol",
    0]$: the touching rule, hit at 0 when $"rate" times "duration" < -"tol"$, no hit otherwise.]),
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
  ([Read the start], [Compute the true distance and the gap $g_0$. Below $-"tol"$: hit at 0. In $[-"tol", 0]$: the
    touching rule. Above 0 but inside the sharp grown cylinder (a rim corner): hit at 0 when the path approaches,
    no hit otherwise.]),
  ([Try the wall], [When the start is outside the grown radius and approaches the axis, and the root exists,
    compute the wall root of @eq:cylinder-hit-time, the constant from the gap; keep it when the path is within the
    grown height then.]),
  ([Try the cap], [When the start is beyond the grown half height and moves toward that cap, compute the time it
    reaches the cap's plane; keep it when the squared distance from the axis then is at most the squared grown
    radius.]),
  ([Keep the earliest], [The earliest kept time at most `duration` is the hit; equal times keep the wall. A hit in
    a rim region is moved to $d = "tol"$ by `MoveParticle`, not here.]),
)

=== Slide and spin

After a hit the response leaves a velocity along the tangent plane at the hit. The sweep goes on along that tangent
in a straight line. On a sphere or a cylinder wall the path leaves the surface, its distance growing as
$("speed" times "time")^2\/(2 "radius")$; on a cap it stays on the plane, as on a flat face. By the convexity above,
a straight path from the surface cannot enter the same solid again in this step, so a round solid is hit at most
once per sweep, and sliding round a sphere does not run into the limit of four hits. Gravity or the pressure brings
the particle back to the surface in later steps, and over many steps it follows the curve; a particle sliding from
the top of a frictionless sphere leaves it where $cos theta = 2\/3$, $theta$ measured from the top.

A turn about a sphere's centre, or about a cylinder's own axis, moves the surface but not the shape. Such a solid is
swept and slid as one that does not turn, with the closed forms above, and the spin enters only through the solid
velocity $bold(u)$, which always uses the full angular velocity. The turn that changes the shape is none for a
sphere, and for a cylinder the part of $bold(omega)$ across its axis, $bold(omega)_"across" = bold(omega) -
(bold(omega) dot bold(e)_z) bold(e)_z$, with $bold(e)_z$ the cylinder's own z axis (its `axis`) at the start of the
sweep. A cylinder is a turning solid
only when $|bold(omega)_"across"| T |"half_size"| > "tol"\/8$, when its shape moves by more than $"tol"\/8$ during
the step; it then uses the stepping search of @sec-first-hit and the slide in its own axes of
@sec-contact-response, and the quarter-turn limit applies to $|bold(omega)_"across"| T$. A spin given along the
axis stays below that bound up to rounding. Why not treat a spinning roller as turning: the slide on a turning
solid carries the particle in the solid's axes, so it would drag a particle round the roller even with friction 0.
Through the solid velocity, the spin acts by friction only. For a cylinder of radius 0.5 m at the origin spinning at
(0, 0, 10) rad/s, a particle hitting the wall at (−0.5, 0, 0) with velocity (1, 0, 0) m/s meets a surface moving
at (0, −5, 0) m/s: the approach speed is −1 m/s, and the particle keeps (0, 0, 0) m/s with friction 0 and takes
(0, −0.5, 0) m/s with friction 0.5 (this paper's numbers, by @eq:contact-response).

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
again, so rounding leaves no approach worth naming.

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
$bold(v) - a bold(n)$, not by moving to the solid's frame, removing it there and moving back. The round trip would erase
a small sideways velocity next to a fast solid. With friction 0 at a flat solid with normal (0, 0, 1) sliding sideways
at $10^8$ m/s, a particle at $(10^(-9), 0, -1)$ m/s keeps its x part of $10^(-9)$ m/s in world coordinates, while the
round trip computes $(10^(-9) - 10^8) + 10^8 = 0$, since doubles near $10^8$ are $1.5 times 10^(-8)$ apart (this paper's
example). Second, the stick test is written as a division, $mu >= s_"side"\/(-a)$, not as $mu (-a) >= s_"side"$: a very
large friction then never overflows.

Each of the particle's two velocities gets the response with its own approach speed. A particle whose
`for_this_step` is (0, 0, −1) m/s and whose `for_next_step` is (1, 0, −3) m/s meets a fixed flat solid with
friction 0.5: `for_next_step` has approach −3 m/s and sideways speed 1 m/s, and $0.5 >= 1\/3$, so it sticks at
zero; `for_this_step` has approach −1 m/s and no sideways speed, so it too becomes zero. The record keeps both
approach speeds, −3 and −1 m/s.

=== Moving one particle

`MoveParticle` moves one particle for one step in six steps (@proc-move-particle): check the inputs, classify the
start, find the first hit, respond, slide, and settle and check the end. Without any hit the end position is
$bold(x)_0 + bold(v) T$, one multiplication and one addition per axis, and both velocities come back unchanged.

The start is classified against every solid in array order, at the start of the step, and the first solid that
fails decides. For a committed plate, being more than $"tol"$ beyond its other face is checked first. Then a start
deeper than `start_overlap_tolerance` inside a solid is an error. A start closer than $"tol"$, down to that depth,
is moved along that solid's normal to $d = "tol"$ before the sweep, whatever its velocity, so a particle placed
next to a solid, by a source or by a change of cell size, never fails the end check for that reason. A later solid
in the array is classified from the moved start. What else happens depends on $d$ before the move:

- $d < 0$: both velocities get the response, the solid joins the surfaces the particle is on, the move counts as
  one hit, and the kind is "started inside solid".
- $0 <= d < "tol"$ and either velocity approaches the solid: the same, with the kind "hit".
- $0 <= d < "tol"$ and neither approaches: the move is silent. Velocities, kind, hit count and record are unchanged,
  and the solid does not join the surfaces: a touching particle that leaves is free.

The first solid that responds fills the record, with hit fraction 0. The kind is "started inside solid" when any
start distance was negative. A flat solid with surface at $z = 0$ rising at 2 m/s, under a particle at rest at $z =
"tol"$ for 0.1 s: the particle approaches the solid at −2 m/s, so it is a hit at fraction 0, both velocities become
(0, 0, 2) m/s, and the particle ends carried at $z = 0.2 + "tol"$. The same particle moving away at (1, 0, 2) m/s
from $(0, 0, "tol"\/4)$ is moved to $z = "tol"$ silently and ends at $(0.1, 0, 0.2 + "tol")$.

A hit that the closed form finds in a rim region of a cylinder is first moved along the rim normal to $d = "tol"$
(@sec-round-solids); the response then works from there. Every change of `velocity.for_next_step` made against a
solid with `reports_reaction`, by the response at a hit or by a removal during the slide below, is added to the
particle's reaction, with the position of the first such change. A removal of the approach to an earlier surface is
charged to that surface's solid; the corner case below, which sets the velocities to the newest solid's velocity,
is charged to the newest solid. Only `for_next_step` counts, because it is the velocity the particle keeps;
`for_this_step` only moves it during this step. The solids module turns these changes, times each particle's mass,
into the force on the solid. That force then comes from the very velocity changes that stopped the particles, so
the same geometry decides both, as Batty, Bertails and Bridson argue for coupling liquids and solids [21]. A
particle pushes at most one reacting solid per step; a second, different one is "too close to call, second reacting
solid".

The slide follows each hit. The solid joins the list of surfaces the particle is on. For each earlier surface that
the new velocities still approach, that approach is removed, without friction. If the newest surface is then
approached again, as in a corner sharper than a right angle, both velocities become the newest solid's velocity.
If an earlier surface is still approached after that, the solids are closing on the particle: "too close to call,
squeezed between solids". On a solid that does not turn, the sweep starts again from the hit position with the new
`for_this_step` for the time left. A fifth hit in one step (`kMaxHitsPerStep` is 4: the three sides of a corner,
plus one) is "too close to call, too many hits".

A slide on a turning solid is carried in the solid's own axes, as the proof of concept did. A sphere, and a
cylinder that only spins about its own axis, are not turning solids here (@sec-round-solids). The relative
`for_this_step` at the hit, $bold(v)_"local"$, is turned into those axes, its part into the face is set to zero, the
particle advances in those axes for the time left $t_"left"$, and it is placed with the solid's pose at the end of
the step. The hit loop ends there. Before that, every other solid $j$ is tested: the call is "too close to call,
turning solid near another solid" when

$ d_j - "tol" <= t_"left" (|bold(v)_"local"| + |bold(v)_"lin"| + |bold(omega)| r + |bold(v)_("lin", j)| +
|bold(omega)_j| r_j), $

with $d_j$ the distance to solid $j$ at the hit, $bold(v)_"lin"$, $bold(omega)$ the turning solid's motion,
$bold(v)_("lin", j)$, $bold(omega)_j$ solid $j$'s, and $r$, $r_j$ the distances from the hit position to each
solid's origin. The right side is an upper bound on how fast the particle and solid $j$ can close: each term bounds
one motion, the speed of a turning solid's points by $|bold(v)_"lin"| + |bold(omega)| r$.

At the end, every surface of the list that the particle is closer to than $"tol"$, at the end pose, moves it along
its normal to $d = "tol"$. Then the end check of @sec-contact-tolerance runs against every solid. On any failure
the result is left unchanged, and the failure context names the solid by its stable `solid_id`, never by its index
in the array, with a length: the depth $-d$ for a start too deep inside, the distance $-(s_"plate" p_z +
"half_thickness")$ beyond a committed plate, the end distance $d$ for an end too close, and 0 otherwise.
`MoveParticles` adds the particle's id.

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
  fails: [a bad input; a start too deep inside a solid or on the wrong side of its plate; too close to call, for one
    of the six reasons named in the text. The result is then unchanged, and the failure names the solid by its
    `solid_id` (never its index) with a length.],
  ([Check inputs], [Every value finite, `step_duration` $> 0$, `friction` $>= 0$, `contact_tolerance` $> 0$,
    `start_overlap_tolerance` $>= 0$, solids in strictly ascending `solid_id`, and, for a committed particle, a
    `plate_side` that names a plate among the solids. Otherwise fail with "bad input".]),
  ([Classify the start], [For each solid in array order, at the start of the step and from the start as moved by
    the earlier solids: fail when the particle is more than $"tol"$ beyond the other face of its committed plate,
    then when $d < -$`start_overlap_tolerance`. When $d < "tol"$, move the start along the normal to $d = "tol"$;
    then, if $d$ before the move was negative or either velocity approaches, respond to both velocities, add the
    solid to the surfaces, count one hit and fill the record if it is empty.]),
  ([Find the first hit], [@proc-find-first-hit from the current position along `for_this_step` for the time left,
    with the solids sampled at the time already used. With no hit, advance by `for_this_step` times the time left
    and go to step 6.]),
  ([Respond], [A hit that @proc-cylinder-first-hit found in a rim region is first moved along the rim normal to
    $d = "tol"$. Apply the response to both velocities with the solid velocity at the hit and `friction` times
    `friction_scale`. Fill the record if it is empty and count the hit. If the solid reports its reaction, add the
    change of `for_next_step`; fail on a second reacting solid.]),
  ([Slide], [Add the solid to the surfaces. Remove, without friction, any approach to an earlier surface; if the
    newest is then approached again, take its solid velocity; if an earlier one still is, fail. Add each change of
    `for_next_step` to the reaction when the solid it is charged to (the earlier surface's for a removal, the
    newest for the corner case) reports its reaction; fail on a second reacting solid. A solid that does not turn:
    go to step 3 for the time left, failing on a fifth hit. A turning solid (never a sphere, nor a cylinder that
    only spins about its axis): fail if another solid $j$ is within reach by the bound of the text, carry the rest
    of the step in its axes, and go to step 6.]),
  ([Settle and check the end], [For each surface the particle is on with $d < "tol"$ at the end pose, move it along
    that normal to $d = "tol"$. Fail when any solid is at $d < "tol"\/2$; otherwise write the result.]),
)

// Every floating figure and procedure is placed before the references start.
#place.flush()

= References

#block[
  #set text(size: 9pt)
  #set par(justify: false, first-line-indent: 0pt, hanging-indent: 1.8em, spacing: 0.75em)
  #include "references.typ"
]
