// Figure <fig-staggered-grid>: where the values of the grid live. Left, one cell in the Z-up axes: the pressure at
// its centre and the x, y and z parts of the velocity on its three high faces, each drawn along its +axis. Right, a
// vertical slice of six cells (X to the right, Z up) with every cell centre and every face piece, and one face named
// with the cells on its two sides.

#import "../figure_style.typ": *

#let staggered-grid-figure() = figure-canvas({
  import cetz.draw: *

  let label = text.with(size: inset-label-size)

  // Left: one cell of width 2, its low corner at the drawing origin.
  let cell-width = 2.0
  let corner(x, y, z) = project((x * cell-width, y * cell-width, z * cell-width))
  let edges = (
    ((0, 0, 0), (1, 0, 0)),
    ((1, 0, 0), (1, 0, 1)),
    ((1, 0, 1), (0, 0, 1)),
    ((0, 0, 1), (0, 0, 0)),
    ((1, 0, 0), (1, 1, 0)),
    ((1, 1, 0), (1, 1, 1)),
    ((1, 1, 1), (1, 0, 1)),
    ((0, 0, 1), (0, 1, 1)),
    ((0, 1, 1), (1, 1, 1)),
  )
  let hidden-edges = (((0, 0, 0), (0, 1, 0)), ((0, 1, 0), (1, 1, 0)), ((0, 1, 0), (0, 1, 1)))
  for (start, end) in hidden-edges {
    line(corner(..start), corner(..end), stroke: (
      paint: palette.muted.darken(30%),
      thickness: guide-stroke,
      dash: "dashed",
    ))
  }
  for (start, end) in edges {
    line(corner(..start), corner(..end), stroke: guide-stroke + black)
  }

  // The pressure at the centre.
  let centre = corner(0.5, 0.5, 0.5)
  circle(centre, radius: 0.07, fill: black, stroke: none)
  content((centre.at(0) - 0.1, centre.at(1) + 0.1), anchor: "south-east", label[pressure])

  // One velocity part on the high face of each axis, drawn along +axis from the face centre.
  // The name sits at `name-at`, a point in cell widths, with `anchor`.
  let face-arrow(face-centre, direction, name, name-at, anchor) = {
    let start = corner(..face-centre)
    let tip-point = face-centre.zip(direction).map(((value, step)) => value + 0.32 * step)
    coloured-arrow(start, corner(..tip-point), colour: palette.liquid)
    circle(start, radius: 0.04, fill: palette.liquid, stroke: none)
    content(corner(..name-at), anchor: anchor, label[#name])
  }
  face-arrow((1, 0.5, 0.5), (1, 0, 0), [x part], (1.34, 0.5, 0.5), "west")
  face-arrow((0.5, 0.5, 1), (0, 0, 1), [z part], (0.5, 0.5, 1.34), "south")
  face-arrow((0.5, 1, 0.5), (0, 1, 0), [y part], (0.5, 1.34, 0.5), "west")

  axes-3d((-1.2, -0.7), length: 0.9)

  // Right: a vertical slice of three by two cells of one size, X to the right and Z up.
  let slice-origin = (4.6, 0.0)
  let unit = 1.5
  let cells = ((0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0))
  cell-grid(slice-origin, cells, unit: unit, centre-dots: true, face-arrows: true)

  // The face between cells (0, 1) and (1, 1): the cell on its low side and the cell on its high side.
  let face-x = slice-origin.at(0) + unit
  let face-low-y = slice-origin.at(1) + unit
  line((face-x, face-low-y), (face-x, face-low-y + unit), stroke: 1.6 * geometry-stroke + palette.highlight)
  content(
    (slice-origin.at(0) + unit / 2, slice-origin.at(1) + 1.68 * unit),
    label[#code-label[cell_low]],
  )
  content(
    (slice-origin.at(0) + 1.5 * unit, slice-origin.at(1) + 1.68 * unit),
    label[#code-label[cell_high]],
  )
  content((face-x, slice-origin.at(1) - 0.4), anchor: "north", text(
    size: inset-label-size,
    fill: palette.highlight,
  )[the red face: normal +X, flow out of #code-label[cell_low] counts positive])

  // The slice's own axes.
  let axes-origin = (slice-origin.at(0) + 3 * unit + 0.55, slice-origin.at(1))
  line(axes-origin, (axes-origin.at(0) + 0.6, axes-origin.at(1)), mark: small-arrow-mark, stroke: guide-stroke)
  line(axes-origin, (axes-origin.at(0), axes-origin.at(1) + 0.6), mark: small-arrow-mark, stroke: guide-stroke)
  content((axes-origin.at(0) + 0.72, axes-origin.at(1)), anchor: "west", label[X])
  content((axes-origin.at(0), axes-origin.at(1) + 0.72), anchor: "south", label[Z])
})
