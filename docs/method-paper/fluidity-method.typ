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
holds what every module shares: the conventions, the notation, and how one operation runs on either processor.

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

== What "same GPU, same input, identical output" promises

The promise: the same build, run twice on the same GPU with the same input, writes the same bytes in every output
(papers call this determinism). The CPU path makes the same promise on one machine with one build. Three rules keep
it. Every sum is made of plain doubles added in a fixed order, never with floating-point atomics, whose order depends
on which thread arrives first. Reductions follow a fixed tree. When several items fail, the one reported is the
first by stable id. Repeat-run tests hold the code to this promise: they demand identical bytes.

What it does not promise: the CPU and the GPU do not give the same bits. They agree within a tolerance that each
module states, as an absolute floor plus a term relative to the size of the value. Nor are results bit-identical
across GPU models, compilers or machines. The reason is that floating-point addition depends on order. In doubles,
(0.1 + 0.2) + 0.3 gives 0.6000000000000001, while 0.1 + (0.2 + 0.3) gives 0.6. The CPU loop and the GPU function
add the same numbers in different orders, so their last bits can differ.

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

The code uses words, never paper symbols. @tab-notation pairs each symbol this paper uses with its code word; a
module that brings a new symbol adds it here.

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
starts as zero bytes, is aligned to 64 bytes, and is freed when the buffer is destroyed.

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

// Every floating figure and procedure is placed before the references start.
#place.flush()

= References

#block[
  #set text(size: 9pt)
  #set par(justify: false, first-line-indent: 0pt, hanging-indent: 1.8em, spacing: 0.75em)
  #include "references.typ"
]
