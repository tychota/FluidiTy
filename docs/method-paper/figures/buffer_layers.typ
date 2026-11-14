// Figure <fig-buffer-layers>: the two layers of Buffer. The typed functions check sizes and slices, the untyped
// byte functions pick the processor, and the copy is memcpy on the CPU or the CUDA runtime on the GPU.

#import "../figure_style.typ": *

#let buffer-layers-figure() = figure-canvas({
  import cetz.draw: *

  // Top: the typed layer callers see.
  labelled-box("typed", (5.5, 4.6), 10.4, 1.25)[
    *typed functions* in #code-label("base/buffer.h") \
    #code-label("BuildBuffer"), #code-label("WriteBuffer"), #code-label("ReadBuffer"), #code-label("CopyBuffer") \
    check sizes and slices, which need the value type
  ]

  // Middle: the untyped layer, one copy of the code for every value type.
  labelled-box("bytes", (5.5, 2.35), 10.4, 1.25, box-fill: fill-emphasis)[
    *untyped byte functions* in #code-label("base/buffer_bytes.h") \
    defined in #code-label("base/buffer_cpu.cc") \
    pick the processor and refuse overlapping byte ranges
  ]

  // Bottom: what really copies, on each processor.
  labelled-box("cpu", (2.6, 0.1), 4.6, 1.05)[
    #code-label("memcpy") \
    in #code-label("base/buffer_cpu.cc")
  ]
  labelled-box("gpu", (8.4, 0.1), 4.6, 1.05)[
    the CUDA runtime \
    in #code-label("base/buffer_gpu.cu")
  ]

  line("typed.south", "bytes.north", mark: arrow-mark, name: "down")
  content("down.mid", anchor: "west", padding: 0.12, [raw bytes and the buffer's processor])

  line((2.6, 1.725), "cpu.north", mark: arrow-mark, name: "to-cpu")
  line((8.4, 1.725), "gpu.north", mark: arrow-mark, name: "to-gpu")
  content("to-cpu.mid", anchor: "east", padding: 0.12, code-label("RunOn::kCpu"))
  content("to-gpu.mid", anchor: "west", padding: 0.12, code-label("RunOn::kGpu"))
})
