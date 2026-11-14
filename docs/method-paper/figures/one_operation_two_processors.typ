// Figure <fig-run-on>: one public function, the shared formula header, the CPU loop and the GPU function that
// both use it, and the test that compares their results.

#import "../figure_style.typ": *

#let run-on-figure() = figure-canvas({
  import cetz.draw: *

  // Top: the one function callers see.
  labelled-box("public", (5.5, 6.3), 10.4, 1.05)[
    *one public function* \
    #code-label("ParticlesToGrid") in #code-label("transfer/particles_to_grid.h")
  ]

  // Middle: the formulas both implementations include.
  labelled-box("shared", (5.5, 4.3), 4.6, 1.25, box-fill: fill-emphasis)[
    *shared formula header* \
    weight curves, marked \
    #code-label("FLUIDITY_CPU_AND_GPU")
  ]

  // Below: the two private implementations.
  labelled-box("cpu", (2.2, 2.1), 4.4, 1.25)[
    *CPU loop* \
    #code-label("particles_to_grid_cpu.cc")
  ]
  labelled-box("gpu", (8.8, 2.1), 4.4, 1.25)[
    *GPU function* \
    #code-label("particles_to_grid_gpu.cu")
  ]

  // The caller's choice selects one implementation. The arrows pass beside the shared header.
  line((2.2, 5.775), "cpu.north", mark: arrow-mark, name: "to-cpu")
  line((8.8, 5.775), "gpu.north", mark: arrow-mark, name: "to-gpu")
  content(("to-cpu.start", 45%, "to-cpu.end"), anchor: "east", padding: 0.12, code-label("RunOn::kCpu"))
  content(("to-gpu.start", 45%, "to-gpu.end"), anchor: "west", padding: 0.12, code-label("RunOn::kGpu"))

  // Both implementations include the same formulas.
  let include-stroke = (paint: black, thickness: guide-stroke, dash: "dashed")
  line((4.1, 3.675), (3.6, 2.725), stroke: include-stroke, mark: small-arrow-mark)
  line((6.9, 3.675), (7.4, 2.725), stroke: include-stroke, mark: small-arrow-mark)
  content((5.5, 3.2), align(center)[same formulas,\ own loops and memory])

  // Bottom: the test that runs the same case on both and compares. The box is wider than the span of the two
  // implementations, so each result arrow drops straight onto its top edge.
  labelled-box("compare", (5.5, 0.0), 8.2, 1.05)[
    *module test:* same inputs on both \
    results agree within the module's tolerance
  ]
  line("cpu.south", ("cpu.south", "|-", "compare.north"), mark: arrow-mark)
  line("gpu.south", ("gpu.south", "|-", "compare.north"), mark: arrow-mark)
})
