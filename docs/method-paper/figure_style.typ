// The shared look of every figure in the method paper.
//
// Every figure imports this file and draws inside `figure-canvas`, so all figures share one stroke width, one
// palette, one label typeface and one arrow. A module that needs a new helper adds it here, not in its figure.
//
// CeTZ 0.5.3 is the release made for Typst 0.15: its typst.toml asks for compiler 0.15.0, the version that builds
// this paper. Since CeTZ 0.5, a z coordinate is ignored outside `ortho`, so 3D sketches here use the explicit
// projection `project` below instead.

#import "@preview/cetz:0.5.3"

// Strokes. Geometry is everything the figure is about; guides are construction lines, mirror lines and inset boxes.
#let geometry-stroke = 0.8pt
#let guide-stroke = 0.4pt

// Palette. The six lightness values (L* about 22, 38, 54, 71, 80, 94) stay apart in greyscale, so a printed
// copy keeps every distinction.
#let palette = (
  solid: rgb("#3b332c"),
  highlight: rgb("#a8281e"),
  liquid: rgb("#3f86c9"),
  particle: rgb("#e8a03c"),
  muted: rgb("#c3c5c9"),
  air: rgb("#e6f0f8"),
)

// Box fills derived from the palette: light enough to carry black text.
#let fill-plain = palette.muted.lighten(70%)
#let fill-emphasis = palette.particle.lighten(75%)

// Labels: the paper's sans at 7.92 pt against the 11 pt body, so labels read as part of the page. Code names in
// labels use the paper's code typeface, as they do in the text.
#let label-font = ("Hermes Maia OT 3", "Libertinus Sans")
#let code-font = ("Triplicate OT A", "DejaVu Sans Mono")
#let label-size = 7.92pt
// Small labels inside an inset or next to a small mark: corner numbers, axis names of a small panel.
#let inset-label-size = 7.04pt

// A code name inside a figure label.
#let code-label(name) = text(font: code-font, features: ("ss02",), size: 0.85em, name)

// The arrowheads: always filled, so none reads as hollow. Two sizes: `arrow-mark` on geometry strokes and
// `small-arrow-mark` on guide strokes and small turning arrows.
#let arrow-scale = 0.7
#let small-arrow-scale = 0.5
#let arrow-mark = (end: "stealth", fill: black, scale: arrow-scale)
#let small-arrow-mark = (end: "stealth", fill: black, scale: small-arrow-scale)

// The arrowhead in `colour`, for any CeTZ element that takes `mark:` (lines, arcs).
#let coloured-mark(colour, small: false) = (
  end: "stealth",
  fill: colour,
  scale: if small { small-arrow-scale } else { arrow-scale },
)

// Draws an arrow through `points` in `colour`, on a geometry stroke unless `stroke-width` says otherwise. Its
// arrowhead takes the same colour; `small: true` uses the small arrowhead.
#let coloured-arrow(..points, colour: black, stroke-width: geometry-stroke, small: false) = cetz.draw.line(
  ..points.pos(),
  stroke: stroke-width + colour,
  mark: coloured-mark(colour, small: small),
)

// Draws the figure `body` (a block of CeTZ draw calls) with the shared defaults. One unit is 0.92 cm, so a figure
// drawn 11.5 units wide fills the 106 mm text width. A drawing a little wider than the text area is scaled down, the
// labels with it, until it fits: a figure never runs into the margin (overflow_check.typ makes the paper fail when
// one does). A drawing that fits is left at its own size.
#let figure-canvas(body) = context {
  let canvas = {
    set text(font: label-font, size: label-size, number-type: "lining")
    cetz.canvas(length: 0.92cm, padding: 0.1, {
      cetz.draw.set-style(
        stroke: geometry-stroke + black,
        mark: (fill: black, scale: arrow-scale),
        content: (padding: 0.08),
      )
      body
    })
  }
  let text-width = page.width - page.margin.left.length - page.margin.right.length
  let drawing-width = measure(canvas).width
  if drawing-width > text-width {
    scale(text-width / drawing-width * 100%, reflow: true, canvas)
  } else {
    canvas
  }
}

// Oblique projection of a point (x, y, z) given in Z-up axes: X to the right, Y into the page drawn up and to
// the right at half length, Z up. Returns the 2D drawing point.
#let depth-angle = 30deg
#let depth-scale = 0.5
#let project(point) = {
  let (x, y, z) = point
  (x + depth-scale * y * calc.cos(depth-angle), z + depth-scale * y * calc.sin(depth-angle))
}

// Draws one arrow per entry of `axes`, each `(direction, name)` with a 3D direction in the projection above,
// starting at `origin` (a 2D drawing point). Names sit just beyond the arrow tips.
#let axis-triad(origin, axes, length: 1.3) = {
  import cetz.draw: *
  let (origin-x, origin-y) = origin
  for (direction, name) in axes {
    let (tip-x, tip-y) = project(direction.map(value => value * length))
    let (label-x, label-y) = project(direction.map(value => value * (length + 0.32)))
    line(origin, (origin-x + tip-x, origin-y + tip-y), mark: arrow-mark)
    content((origin-x + label-x, origin-y + label-y), text(weight: "bold", name))
  }
}

// Draws the right-handed X, Y, Z axes of this paper at `origin`: X right, Y into the page, Z up.
#let axes-3d(origin, length: 1.3) = axis-triad(
  origin,
  (((1, 0, 0), [X]), ((0, 1, 0), [Y]), ((0, 0, 1), [Z])),
  length: length,
)

// Draws a rounded box named `name` centred on `box-centre`, `width` by `height` units, holding `body`.
#let labelled-box(name, box-centre, width, height, body, box-fill: fill-plain) = {
  import cetz.draw: *
  let (center-x, center-y) = box-centre
  rect(
    (center-x - width / 2, center-y - height / 2),
    (center-x + width / 2, center-y + height / 2),
    name: name,
    radius: 0.12,
    fill: box-fill,
  )
  content(name + ".center", align(center, body))
}

// Padding: a value that the method assumes but never keeps in memory, such as the zeros that fill a tree sum up to
// a power of two. Drawn dashed in a muted grey, so it reads as "not really there" in colour and in greyscale.
#let padding-line-colour = palette.muted.darken(35%)
#let padding-text-colour = palette.muted.darken(45%)
#let padding-stroke = (paint: padding-line-colour, thickness: geometry-stroke, dash: "dashed")

// Draws a dashed padding box named `name` centred on `box-centre`, `width` by `height` units, holding `body` in
// the padding text colour. The same shape as `labelled-box`, so a padded entry lines up with the real ones.
#let padding-box(name, box-centre, width, height, body) = {
  import cetz.draw: *
  let (center-x, center-y) = box-centre
  rect(
    (center-x - width / 2, center-y - height / 2),
    (center-x + width / 2, center-y + height / 2),
    name: name,
    radius: 0.12,
    fill: white,
    stroke: padding-stroke,
  )
  content(name + ".center", align(center, text(fill: padding-text-colour, body)))
}

// Draws an arrow from `from` to `to` in the padding style, for an edge that carries only padding.
#let padding-arrow(from, to) = cetz.draw.line(
  from,
  to,
  stroke: padding-stroke,
  mark: coloured-mark(padding-line-colour, small: true),
)

// Solids and particles, first used by the contact figures. A solid is filled in a light tint of the solid colour
// and its surface is drawn in the solid colour itself; a particle is a small disc in the particle colour.
#let fill-solid = palette.solid.lighten(80%)
#let surface-stroke = 1.2 * geometry-stroke + palette.solid
#let particle-radius = 0.09

// Draws a particle at `point`: a disc in `colour`, outlined in black so it reads on any fill.
#let particle-dot(point, colour: palette.particle) = cetz.draw.circle(
  point,
  radius: particle-radius,
  fill: colour,
  stroke: 0.5 * geometry-stroke + black,
)

// Draws a dimension between two 2D points `from` and `to`: a thin line with a small arrowhead at each end.
#let dimension-line(from, to, colour: black) = cetz.draw.line(
  from,
  to,
  stroke: guide-stroke + colour,
  mark: (start: "stealth", end: "stealth", fill: colour, scale: small-arrow-scale),
)

// Cells of mixed sizes, first used by the grid figures. A figure draws a 2D slice of the grid: the first coordinate
// runs to the right and the second up. A cell is `(x, y, size_level)`, with integer coordinates in cells of its own
// size level, as `CellCoord` counts them. Lengths below are in widths of a size-level-0 cell, so a cell of size level
// `l` is 2^-l wide and every cell is aligned by construction.

// The low corner and the width of `cell`: `(low-x, low-y, width)`.
#let cell-box(cell) = {
  let (x, y, size-level) = cell
  let width = calc.pow(2.0, -size-level)
  (x * width, y * width, width)
}

// The index in `cells` of the cell whose box holds `point`, low sides included and high sides not; none when no
// cell holds it.
#let find-cell-at(cells, point) = {
  let (point-x, point-y) = point
  let found = none
  for (index, cell) in cells.enumerate() {
    let (low-x, low-y, width) = cell-box(cell)
    let is-inside-x = low-x <= point-x and point-x < low-x + width
    let is-inside-y = low-y <= point-y and point-y < low-y + width
    if found == none and is-inside-x and is-inside-y {
      found = index
    }
  }
  found
}

// The face pieces of the slice, each listed once with its smaller cell, as the grid stores them. Each piece is a
// dictionary: `axis` (0 for the first axis, 1 for the second), `centre` (a point), `length` (the piece's width),
// `size-level` (its smaller cell's), and `cell-low` and `cell-high` (indexes into `cells`, none at the slice's edge).
// A side beside smaller cells is not a piece: the smaller cells' sides are.
#let cell-grid-face-pieces(cells) = {
  let pieces = ()
  for (index, cell) in cells.enumerate() {
    let (low-x, low-y, width) = cell-box(cell)
    let step-across = width / 1024
    for axis in (0, 1) {
      // The middle of the cell's low side and of its high side along `axis`.
      let low-middle = if axis == 0 { (low-x, low-y + width / 2) } else { (low-x + width / 2, low-y) }
      let high-middle = if axis == 0 {
        (low-x + width, low-y + width / 2)
      } else {
        (low-x + width / 2, low-y + width)
      }
      let shifted(point, distance) = if axis == 0 {
        (point.at(0) + distance, point.at(1))
      } else {
        (point.at(0), point.at(1) + distance)
      }
      // High side: a piece with this cell on its low side, unless smaller cells lie across it.
      let high-neighbour = find-cell-at(cells, shifted(high-middle, step-across))
      let is-high-piece = high-neighbour == none or cell-box(cells.at(high-neighbour)).at(2) >= width
      if is-high-piece {
        pieces.push((
          axis: axis,
          centre: high-middle,
          length: width,
          size-level: cell.at(2),
          cell-low: index,
          cell-high: high-neighbour,
        ))
      }
      // Low side: a piece only beside a bigger cell or the slice's edge; beside a same-size cell it is that cell's
      // high side, listed there.
      let low-neighbour = find-cell-at(cells, shifted(low-middle, -step-across))
      let is-low-piece = low-neighbour == none or cell-box(cells.at(low-neighbour)).at(2) > width
      if is-low-piece {
        pieces.push((
          axis: axis,
          centre: low-middle,
          length: width,
          size-level: cell.at(2),
          cell-low: low-neighbour,
          cell-high: index,
        ))
      }
    }
  }
  pieces
}

// The drawing point of the slice point `point`, for a slice drawn at `origin` with a size-level-0 cell `unit` wide.
#let cell-grid-point(origin, unit, point) = (origin.at(0) + unit * point.at(0), origin.at(1) + unit * point.at(1))

// Draws the cells of `cells`, a list of `(x, y, size_level)`, with the low corner of coordinate (0, 0) at `origin`
// and a size-level-0 cell `unit` drawing units wide. `fill-of`, when given, maps a cell to its fill. `centre-dots`
// draws a dot at each cell centre, where the pressure lives. `face-arrows` draws on each face piece a short arrow
// along its +axis, where a velocity part lives, in `arrow-colour`.
#let cell-grid(
  origin,
  cells,
  unit: 1,
  fill-of: none,
  centre-dots: false,
  face-arrows: false,
  arrow-colour: palette.liquid,
) = {
  import cetz.draw: *
  for cell in cells {
    let (low-x, low-y, width) = cell-box(cell)
    let cell-fill = if fill-of == none { white } else { fill-of(cell) }
    rect(
      cell-grid-point(origin, unit, (low-x, low-y)),
      cell-grid-point(origin, unit, (low-x + width, low-y + width)),
      fill: cell-fill,
      stroke: guide-stroke + black,
    )
  }
  if centre-dots {
    for cell in cells {
      let (low-x, low-y, width) = cell-box(cell)
      let centre = cell-grid-point(origin, unit, (low-x + width / 2, low-y + width / 2))
      circle(centre, radius: 0.05, fill: black, stroke: none)
    }
  }
  if face-arrows {
    for piece in cell-grid-face-pieces(cells) {
      let half-arrow = 0.2 * piece.length
      let (centre-x, centre-y) = piece.centre
      let (arrow-start, arrow-tip) = if piece.axis == 0 {
        ((centre-x - half-arrow, centre-y), (centre-x + half-arrow, centre-y))
      } else {
        ((centre-x, centre-y - half-arrow), (centre-x, centre-y + half-arrow))
      }
      coloured-arrow(
        cell-grid-point(origin, unit, arrow-start),
        cell-grid-point(origin, unit, arrow-tip),
        colour: arrow-colour,
        stroke-width: guide-stroke,
        small: true,
      )
    }
  }
}
