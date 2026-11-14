// Figure <fig-axes>: the Y-up axes of the proof of concept, the Z-up axes of this paper, the quarter turn about X
// that carries one to the other, and an inset showing that swapping two axes would mirror a triangle.

#import "../figure_style.typ": *

// Adds two 2D points.
#let offset(point, shift) = (point.at(0) + shift.at(0), point.at(1) + shift.at(1))

// Draws a gravity arrow pointing down the page from `top`, labelled with its direction in its panel's axes.
#let gravity-arrow(top, direction) = {
  import cetz.draw: *
  coloured-arrow(top, offset(top, (0, -1.1)), colour: palette.highlight)
  content(offset(top, (0.14, -0.45)), anchor: "west", align(left)[gravity\ #direction])
}

// Draws the quarter turn about X as seen on the axes: a guide circle around a short X axis, and a large arc that
// lifts the direction toward the reader until it points up. A dot marks where the arc starts. `centre` is where
// the circle meets the X axis.
#let quarter-turn(centre, radius) = {
  import cetz.draw: *
  // A point of the circle around X, at `angle` from the direction into the page toward up.
  let circle-point(angle) = offset(centre, project((0, radius * calc.cos(angle), radius * calc.sin(angle))))
  let full-turn = range(0, 49).map(index => circle-point(index * 7.5deg))
  line(..full-turn, stroke: guide-stroke + palette.muted.darken(25%))
  line(offset(centre, (-1.1, 0)), offset(centre, (1.2, 0)), stroke: guide-stroke + black, mark: small-arrow-mark)
  content(offset(centre, (1.4, 0)), text(weight: "bold")[X])
  // From toward the reader (180 degrees) to up (90 degrees), heavier than the geometry so it reads first.
  let lift = range(0, 19).map(index => circle-point(180deg - index * 5deg))
  circle(lift.first(), radius: 0.07, fill: palette.highlight, stroke: none)
  coloured-arrow(..lift, colour: palette.highlight, stroke-width: 1.5 * geometry-stroke)
  content(offset(lift.first(), (-0.1, -0.08)), anchor: "north-east", text(size: inset-label-size)[toward you])
  content(offset(lift.last(), (-0.12, 0.08)), anchor: "south-east", text(size: inset-label-size)[up])
}

// Draws one inset panel: Y and Z axes as guides and the triangle with corners 1, 2, 3 given in (y, z), with a
// curved arrow inside that shows the order in which the corners go round. `title` goes under the panel. A panel
// with `mirror: true` also draws the mirror line y = z through its origin, under the triangle.
#let winding-panel(origin, corners, counter-clockwise, title, mirror: false) = {
  import cetz.draw: *
  let to-canvas(point) = offset(origin, point)
  if mirror {
    line(to-canvas((0, 0)), to-canvas((1.9, 1.9)), stroke: (paint: black, thickness: guide-stroke, dash: "dashed"))
    content(to-canvas((1.95, 1.95)), anchor: "south-west", text(size: inset-label-size)[mirror])
  }
  line(to-canvas((0, 0)), to-canvas((2.0, 0)), stroke: guide-stroke + black, mark: small-arrow-mark)
  line(to-canvas((0, 0)), to-canvas((0, 2.0)), stroke: guide-stroke + black, mark: small-arrow-mark)
  content(to-canvas((2.15, 0)), anchor: "west", text(size: inset-label-size)[Y])
  content(to-canvas((0, 2.15)), anchor: "south", text(size: inset-label-size)[Z])
  let triangle-stroke = geometry-stroke + palette.liquid
  line(..corners.map(to-canvas), close: true, fill: palette.liquid.lighten(60%), stroke: triangle-stroke)
  let corner-sum = corners.fold((0, 0), offset)
  let centroid = (corner-sum.at(0) / 3, corner-sum.at(1) / 3)
  for (index, corner) in corners.enumerate() {
    // Push each number a fixed distance beyond its corner, away from the centroid, so it never sits on an edge.
    // A white backing keeps it readable where the mirror line passes.
    let away = (corner.at(0) - centroid.at(0), corner.at(1) - centroid.at(1))
    let away-length = calc.sqrt(away.at(0) * away.at(0) + away.at(1) * away.at(1))
    let label-point = offset(corner, (0.2 * away.at(0) / away-length, 0.2 * away.at(1) / away-length))
    content(to-canvas(label-point), text(size: inset-label-size, str(index + 1)), frame: "rect", fill: white, stroke: (
      none
    ), padding: 0.03)
  }
  let (start-angle, stop-angle) = if counter-clockwise { (-80deg, 200deg) } else { (200deg, -80deg) }
  arc(to-canvas(centroid), start: start-angle, stop: stop-angle, radius: 0.2, anchor: "origin", stroke: (
    geometry-stroke + palette.highlight
  ), mark: coloured-mark(palette.highlight, small: true))
  content(to-canvas((1.0, -0.2)), anchor: "north", align(center, title))
}

#let axes-figure() = figure-canvas({
  import cetz.draw: *

  // Left: the proof-of-concept axes. Y is up and Z points at the reader.
  content((1.9, 3.25), [proof of concept: Y up])
  axis-triad((1.0, 0.9), (((1, 0, 0), [X]), ((0, 0, 1), [Y]), ((0, -1, 0), [Z])))
  gravity-arrow((3.05, 2.6), [(0, −1, 0)])

  // Middle: the quarter turn about X.
  quarter-turn((6.0, 1.75), 0.95)
  content((6.0, 0.3), [quarter turn about X])
  content((6.0, -0.12), [(_x_, _y_, _z_) → (_x_, −_z_, _y_)])

  // Right: this paper's axes, Z up and Y away from the reader. Gravity has not moved.
  content((8.9, 3.25), [this paper: Z up])
  axes-3d((8.0, 0.9))
  gravity-arrow((10.05, 2.6), [(0, 0, −1)])

  // Inset: what the turn keeps and what a swap of two axes would break.
  rect((0, -0.6), (11.4, -4.9), radius: 0.12, stroke: guide-stroke + palette.muted.darken(25%))
  content((0.25, -0.9), anchor: "west", text(weight: "bold")[Inset: one triangle in the Y–Z plane, seen from +X])
  let corners = ((0.55, 0.5), (1.6, 0.65), (0.8, 1.5))
  winding-panel((0.7, -3.85), corners, true, [before])
  // The quarter turn maps (y, z) to (−z, y), a turn by 90 degrees in the plane; 2.05 shifts the result back into
  // the panel without changing its shape.
  let turned = corners.map(corner => (2.05 - corner.at(1), corner.at(0)))
  winding-panel((4.5, -3.85), turned, true, [quarter turn: same order])
  // Swapping the two axes maps (y, z) to (z, y), a mirror image across the line y = z.
  let swapped = corners.map(corner => (corner.at(1), corner.at(0)))
  winding-panel((8.3, -3.85), swapped, false, [swap Y and Z: order reversed], mirror: true)
})
