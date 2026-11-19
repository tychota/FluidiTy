// Figure <fig-face-pieces>: face pieces between cells of two sizes. Left, the +X side of a big cell beside four small
// cells, seen from the +X side: the big side is four face pieces, one per small cell, each with its own value along +X.
// Middle, the same in a vertical slice, where two of the four pieces show, listed with the small cells. Right, a big
// cell beside cells two size levels smaller, which the size-step rule does not allow, crossed out.

#import "../figure_style.typ": *

#let face-pieces-figure() = figure-canvas({
  import cetz.draw: *

  let label = text.with(size: inset-label-size)

  // Left: the big cell's +X side seen from the +X side (Y to the right, Z up), where four small cells touch it. Each
  // quarter is one face piece with its own value along +X, drawn as a dot in a circle: an arrow pointing at the reader.
  let side-width = 2.4
  for quarter-y in (0, 1) {
    for quarter-z in (0, 1) {
      let low = (quarter-y * side-width / 2, quarter-z * side-width / 2)
      let high = (low.at(0) + side-width / 2, low.at(1) + side-width / 2)
      rect(low, high, fill: fill-emphasis, stroke: guide-stroke + palette.highlight)
      let centre = ((low.at(0) + high.at(0)) / 2, (low.at(1) + high.at(1)) / 2)
      circle(centre, radius: 0.16, fill: white, stroke: guide-stroke + palette.liquid)
      circle(centre, radius: 0.045, fill: palette.liquid, stroke: none)
    }
  }
  rect((0, 0), (side-width, side-width), stroke: geometry-stroke + black)
  content((side-width / 2, -0.2), anchor: "north", align(center, label[the big cell's +X side, seen from the +X side: \
    four face pieces, each a quarter]))
  line((side-width + 0.25, 0), (side-width + 0.75, 0), mark: small-arrow-mark, stroke: guide-stroke)
  line((side-width + 0.25, 0), (side-width + 0.25, 0.5), mark: small-arrow-mark, stroke: guide-stroke)
  content((side-width + 0.85, 0), anchor: "west", label[Y])
  content((side-width + 0.25, 0.6), anchor: "south", label[Z])

  // Middle: the same in a vertical slice. The two pieces on the big cell's side are listed with the small cells.
  let middle-origin = (4.6, 0.0)
  let unit = 1.6
  let allowed = ((0, 0, 0), (2, 0, 1), (3, 0, 1), (2, 1, 1), (3, 1, 1))
  let fill-small(cell) = if cell.at(2) == 1 { fill-emphasis } else { white }
  cell-grid(middle-origin, allowed, unit: unit, fill-of: fill-small, face-arrows: true)
  line(
    cell-grid-point(middle-origin, unit, (1, 0)),
    cell-grid-point(middle-origin, unit, (1, 1)),
    stroke: 1.6 * geometry-stroke + palette.highlight,
  )
  content(cell-grid-point(middle-origin, unit, (0.75, -0.2)), anchor: "north", label[one size level apart])

  // Right: a big cell beside cells two size levels smaller: not allowed, crossed out.
  let right-origin = (8.6, 0.0)
  let forbidden = ((0, 0, 0), (4, 0, 2), (4, 1, 2), (4, 2, 2), (4, 3, 2))
  cell-grid(right-origin, forbidden, unit: unit)
  let cross-low = cell-grid-point(right-origin, unit, (-0.05, -0.05))
  let cross-high = cell-grid-point(right-origin, unit, (1.3, 1.05))
  line(cross-low, cross-high, stroke: 1.6 * geometry-stroke + palette.highlight)
  line((cross-low.at(0), cross-high.at(1)), (cross-high.at(0), cross-low.at(1)), stroke: 1.6 * geometry-stroke
    + palette.highlight)
  content(cell-grid-point(right-origin, unit, (0.62, -0.2)), anchor: "north", text(
    size: inset-label-size,
    fill: palette.highlight,
  )[two size levels apart])
})
