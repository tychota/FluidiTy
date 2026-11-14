// Figure <fig-buffer-crossings>: a buffer on each processor, the only two calls that move values across
// (WriteBuffer and ReadBuffer), and CopyBuffer, which stays on one processor.

#import "../figure_style.typ": *

#let buffer-crossings-figure() = figure-canvas({
  import cetz.draw: *

  // The two memories, as frames with their names on top.
  let frame-stroke = (paint: black, thickness: guide-stroke, dash: "dashed")
  rect((-0.3, -0.9), (4.4, 4.4), stroke: frame-stroke, radius: 0.12)
  rect((7.25, -0.9), (11.95, 4.4), stroke: frame-stroke, radius: 0.12)
  content((2.15, 4.1), [*CPU memory*])
  content((9.5, 4.1), [*GPU memory*])

  // Left: a plain CPU array of the caller, and a buffer on the CPU.
  labelled-box("cpu-array", (2.15, 3.0), 3.8, 0.95)[
    a CPU array \
    (#code-label("std::vector"))
  ]
  labelled-box("cpu-buffer", (2.15, 0.1), 3.8, 0.95)[
    #code-label("Buffer") on #code-label("RunOn::kCpu")
  ]

  // Right: two buffers on the GPU.
  labelled-box("gpu-buffer", (9.5, 3.0), 3.8, 0.95, box-fill: fill-emphasis)[
    #code-label("Buffer") on #code-label("RunOn::kGpu")
  ]
  labelled-box("gpu-other", (9.5, 0.1), 3.8, 0.95, box-fill: fill-emphasis)[
    a second GPU #code-label("Buffer")
  ]

  // The only crossings: WriteBuffer to the GPU, ReadBuffer back.
  line((4.05, 3.22), (7.6, 3.22), mark: arrow-mark, name: "write")
  line((7.6, 2.78), (4.05, 2.78), mark: arrow-mark, name: "read")
  content("write.mid", anchor: "south", padding: 0.08, code-label("WriteBuffer"))
  content("read.mid", anchor: "north", padding: 0.08, code-label("ReadBuffer"))

  // The same two calls fill and read a CPU buffer, without crossing.
  line((1.95, 2.525), (1.95, 0.575), mark: arrow-mark, name: "cpu-write")
  line((2.35, 0.575), (2.35, 2.525), mark: arrow-mark, name: "cpu-read")
  content("cpu-write.mid", anchor: "east", padding: 0.1, code-label("WriteBuffer"))
  content("cpu-read.mid", anchor: "west", padding: 0.1, code-label("ReadBuffer"))

  // CopyBuffer stays on one processor.
  line("gpu-buffer.south", "gpu-other.north", mark: arrow-mark, name: "copy")
  content("copy.mid", anchor: "west", padding: 0.1, code-label("CopyBuffer"))

  // Across processors CopyBuffer is not allowed: a precondition, checked in debug builds.
  let refused-stroke = (paint: palette.highlight, thickness: geometry-stroke, dash: "dashed")
  line("cpu-buffer.east", "gpu-other.west", stroke: refused-stroke, mark: coloured-mark(palette.highlight), name: "no")
  content("no.mid", anchor: "south", padding: 0.12, align(center, text(fill: palette.highlight)[
    #code-label("CopyBuffer"): \
    not allowed
  ]))
})
