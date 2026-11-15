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
// drawn 11.5 units wide fills the 106 mm text width.
#let figure-canvas(body) = {
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

// `cell-grid(...)`, which draws cells of mixed sizes, arrives with the grid module.
