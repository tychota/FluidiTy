// Figure <fig-run-memory>: one RunMemory of 2,048 bytes after three buffers were built from it: 10, 100 and 3
// doubles. Each piece starts at a multiple of 256 bytes and takes its byte count rounded up to 256; the rounding is
// drawn in the padding style of figure_style.typ. The next piece starts at the next free offset, which equals the
// bytes handed out; the rest of the block is unused.

#import "../figure_style.typ": *

#let run-memory-figure() = figure-canvas({
  import cetz.draw: *

  let reserved-bytes = 2048
  let block-width = 11.0
  let block-height = 0.9
  let x-of(offset) = block-width * offset / reserved-bytes
  // A byte count with a thousands comma, as the text writes it: 1536 becomes 1,536.
  let bytes-text(count) = if count < 1000 { str(count) } else {
    let below-thousand = str(calc.rem(count, 1000))
    str(calc.div-euclid(count, 1000)) + "," + "0" * (3 - below-thousand.len()) + below-thousand
  }

  // Each piece: (first byte, bytes asked, bytes taken, label).
  let pieces = (
    (0, 80, 256, [10 doubles]),
    (256, 800, 1024, [100 doubles]),
    (1280, 24, 256, [3 doubles]),
  )
  let next-free-offset = 1536

  // The pieces: the bytes asked in the emphasis fill, the rounding as padding.
  for (first, asked, taken, name) in pieces {
    rect((x-of(first), 0), (x-of(first + asked), block-height), fill: fill-emphasis, stroke: none)
    rect(
      (x-of(first + asked), 0),
      (x-of(first + taken), block-height),
      fill: white,
      stroke: (paint: padding-line-colour, thickness: guide-stroke, dash: "dashed"),
    )
    content((x-of(first + taken / 2), block-height + 0.12), anchor: "south", align(center)[
      #name \
      #bytes-text(asked) B → #bytes-text(taken) B
    ])
  }

  // The unused end of the block.
  rect((x-of(next-free-offset), 0), (x-of(reserved-bytes), block-height), fill: fill-plain, stroke: none)
  content((x-of((next-free-offset + reserved-bytes) / 2), block-height / 2), [unused])

  // The block outline and the borders between pieces, drawn last so they sit on top of the fills.
  rect((0, 0), (block-width, block-height))
  for offset in (256, 1280, 1536) {
    line((x-of(offset), 0), (x-of(offset), block-height))
  }

  // Offsets under the block.
  for offset in (0, 256, 1280, 1536, 2048) {
    line((x-of(offset), 0), (x-of(offset), -0.12), stroke: guide-stroke)
    content((x-of(offset), -0.16), anchor: "north", text(size: inset-label-size)[#bytes-text(offset)])
  }

  // The next free offset, where the next piece starts, and the end of the block.
  coloured-arrow(
    (x-of(next-free-offset), -1.25),
    (x-of(next-free-offset), -0.55),
    colour: palette.highlight,
    small: true,
  )
  content((x-of(next-free-offset) - 0.12, -1.0), anchor: "east", align(right)[
    next free offset \
    #code-label("handed_out_bytes()") \= 1,536
  ])
  content((x-of(reserved-bytes), -0.6), anchor: "north-east", align(right)[
    #code-label("reserved_bytes()") \
    \= 2,048
  ])
})
