// Figure <fig-tree-sum>: the fixed tree of math::TreeSum over seven values. Index 7 is padding, a zero that is
// never kept in memory, drawn in the padding style of figure_style.typ. Each pair sum is labelled with the indexes
// it covers, and each row of the tree is one layer of the GPU reduction.

#import "../figure_style.typ": *

#let tree-sum-figure() = figure-canvas({
  import cetz.draw: *

  let value-count = 7
  let index-count = 8
  let first-x = 2.2
  let spacing = 1.22
  let layer-height = 1.25
  let box-width = 0.95
  let box-height = 0.55

  // The x of the centre of the box that covers indexes first..last at any layer.
  let centre-x(first, last) = first-x + spacing * (first + last) / 2

  // Row names on the left, one per layer.
  let row-names = ([values], [layer 1], [layer 2], [layer 3])
  for (layer, name) in row-names.enumerate() {
    content((0, layer * layer-height), anchor: "west", name)
  }

  // Layer 0: the seven values and the padded index.
  for index in range(index-count) {
    let name = "r0-" + str(index)
    let centre = (centre-x(index, index), 0)
    if index < value-count {
      labelled-box(name, centre, box-width, box-height, str(index))
    } else {
      padding-box(name, centre, box-width, box-height, str(index))
      let (x, y) = centre
      content((x, y - box-height / 2 - 0.12), anchor: "north", text(
        size: inset-label-size,
        fill: padding-text-colour,
      )[padding: +0.0])
    }
  }

  // Layers 1 to 3: each pair sum covers the indexes of its two sides. Arrows run from each side to its sum. A side
  // that covers only padding gets the padding arrow.
  for layer in range(1, 4) {
    let width = calc.pow(2, layer)
    for pair in range(calc.div-euclid(index-count, width)) {
      let first = pair * width
      let last = first + width - 1
      let name = "r" + str(layer) + "-" + str(first)
      let box-fill = if layer == 3 { fill-emphasis } else { fill-plain }
      labelled-box(name, (centre-x(first, last), layer * layer-height), box-width, box-height, box-fill: box-fill)[
        #first–#last
      ]
      let half = calc.div-euclid(width, 2)
      for side-first in (first, first + half) {
        let side = "r" + str(layer - 1) + "-" + str(side-first)
        let side-is-padding = side-first >= value-count
        if side-is-padding {
          padding-arrow(side + ".north", name + ".south")
        } else {
          line(side + ".north", name + ".south", stroke: guide-stroke + black, mark: small-arrow-mark)
        }
      }
    }
  }

  // The result beside the top.
  content((rel: (0.15, 0), to: "r3-0.east"), anchor: "west", [the sum])
})
