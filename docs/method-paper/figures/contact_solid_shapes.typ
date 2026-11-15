// Figure <fig-solid-shapes>: the four flat-faced solid shapes of contact, each cut by the X-Z plane of its own axes
// and seen from the front. Each panel draws the solid's own x and z axes at its origin, a normal_out_of_solid, and
// the names the code uses: the faces of the flat solid, the plate and the box, the two plate sides, and the Blender
// names of the domain sides. The sphere and the cylinder have their own figure (contact_round_solids.typ).

#import "../figure_style.typ": *

// Adds two 2D points.
#let offset(point, shift) = (point.at(0) + shift.at(0), point.at(1) + shift.at(1))

// Maps a point `local` given in a solid's own (x, z) axes to the drawing, for a solid whose origin is drawn at
// `origin` and whose x axis is turned by `angle` from the page's horizontal.
#let to-page(origin, angle, local) = {
  let (x, z) = local
  (
    origin.at(0) + x * calc.cos(angle) - z * calc.sin(angle),
    origin.at(1) + x * calc.sin(angle) + z * calc.cos(angle),
  )
}

// Draws the solid's own x and z axes at `origin`, turned by `angle`, as short guide arrows labelled in italics.
#let own-axes(origin, angle, length: 0.75) = {
  import cetz.draw: *
  for (direction, name) in (((1, 0), [_x_]), ((0, 1), [_z_])) {
    let tip = to-page(origin, angle, direction.map(value => value * length))
    let label-point = to-page(origin, angle, direction.map(value => value * (length + 0.2)))
    line(origin, tip, stroke: guide-stroke + black, mark: small-arrow-mark)
    content(label-point, text(size: inset-label-size, name))
  }
  circle(origin, radius: 0.035, fill: black, stroke: none)
}

// Draws a normal arrow from `foot` along the unit `direction` (page coordinates), `length` long, in the highlight
// colour.
#let normal-arrow(foot, direction, length: 0.7) = coloured-arrow(
  foot,
  offset(foot, direction.map(value => value * length)),
  colour: palette.highlight,
)

// Draws a panel title at `point`.
#let panel-title(point, body) = cetz.draw.content(point, anchor: "north-west", text(weight: "bold", body))

#let solid-shapes-figure() = figure-canvas({
  import cetz.draw: *

  let panel-width = 5.5
  let left-x = 0
  let right-x = 5.9
  let label = text.with(size: inset-label-size)
  let red-label = text.with(size: inset-label-size, fill: palette.highlight)

  // Row 1, left: a flat solid, tilted, filling z <= 0 of its own axes. Its origin is its point_on_surface.
  {
    let origin = (left-x + 3.9, 10.9)
    let angle = 12deg
    panel-title((left-x, 13.0), [flat solid])
    let corners = ((-3.3, 0), (1.5, 0), (1.5, -1.0), (-3.3, -1.0)).map(point => to-page(origin, angle, point))
    line(..corners, close: true, fill: fill-solid, stroke: none)
    line(corners.at(0), corners.at(1), stroke: surface-stroke)
    own-axes(origin, angle)
    let foot = to-page(origin, angle, (-2.4, 0))
    let direction = (-calc.sin(angle), calc.cos(angle))
    normal-arrow(foot, direction)
    content(offset(foot, direction.map(value => value * 0.75)), anchor: "south", red-label(
      code-label("normal_out_of_solid"),
    ))
    content(to-page(origin, angle, (0.55, -0.42)), label[face #code-label("kHighZ")])
    content(to-page(origin, angle, (-1.4, -0.6)), label[fills _z_ ≤ 0])
  }

  // Row 1, right: a plate, filling |z| <= half_thickness, with its two sides.
  {
    let origin = (right-x + 2.4, 11.0)
    let half-thickness = 0.22
    panel-title((right-x, 13.0), [plate])
    rect(offset(origin, (-2.3, -half-thickness)), offset(origin, (2.1, half-thickness)), fill: fill-solid, stroke: none)
    line(offset(origin, (-2.3, half-thickness)), offset(origin, (2.1, half-thickness)), stroke: surface-stroke)
    line(offset(origin, (-2.3, -half-thickness)), offset(origin, (2.1, -half-thickness)), stroke: surface-stroke)
    own-axes(origin, 0deg, length: 0.6)
    normal-arrow(offset(origin, (-1.6, half-thickness)), (0, 1), length: 0.6)
    content(offset(origin, (-1.5, 0.62)), anchor: "west", red-label(code-label("normal")))
    content(offset(origin, (1.4, half-thickness + 0.06)), anchor: "south", label[#code-label("kHighZ")])
    content(offset(origin, (1.4, -half-thickness - 0.06)), anchor: "north", label[#code-label("kLowZ")])
    content(offset(origin, (0.15, 1.15)), label[side #code-label("kAlongNormal")])
    content(offset(origin, (0.15, -1.15)), label[side #code-label("kAgainstNormal")])
    dimension-line(offset(origin, (-2.15, -half-thickness)), offset(origin, (-2.15, half-thickness)))
    content(offset(origin, (-2.05, -0.62)), anchor: "west", label[2 #code-label("half_thickness")])
    line(offset(origin, (-2.15, -half-thickness)), offset(origin, (-2.0, -0.5)), stroke: guide-stroke)
  }

  // Row 2, left: a box turned about its y axis, with its half sizes and the normal of one face.
  {
    let origin = (left-x + 2.6, 7.0)
    let angle = 20deg
    let half-x = 1.2
    let half-z = 0.7
    panel-title((left-x, 9.0), [box])
    let corners = ((-half-x, -half-z), (half-x, -half-z), (half-x, half-z), (-half-x, half-z))
    line(..corners.map(point => to-page(origin, angle, point)), close: true, fill: fill-solid, stroke: surface-stroke)
    own-axes(origin, angle, length: 0.6)
    let foot = to-page(origin, angle, (half-x, 0.15))
    normal-arrow(foot, (calc.cos(angle), calc.sin(angle)), length: 0.6)
    content(offset(foot, (0.45, 0.3)), anchor: "south", red-label[#code-label("kHighX")])
    content(to-page(origin, angle, (-half-x - 0.12, 0)), anchor: "east", label[#code-label("kLowX")])
    content(to-page(origin, angle, (-0.75, half-z + 0.08)), anchor: "south", label[#code-label("kHighZ")])
    content(to-page(origin, angle, (-0.6, -half-z - 0.08)), anchor: "north", label[#code-label("kLowZ")])
    // Half size along x, drawn in the box's axes just below its bottom face.
    let from = to-page(origin, angle, (0, -half-z - 0.2))
    let to = to-page(origin, angle, (half-x, -half-z - 0.2))
    line(to-page(origin, angle, (0, -half-z)), to-page(origin, angle, (0, -half-z - 0.3)), stroke: guide-stroke)
    line(to-page(origin, angle, (half-x, -half-z)), to-page(origin, angle, (half-x, -half-z - 0.3)), stroke: (
      guide-stroke
    ))
    dimension-line(from, to)
    content(to-page(origin, angle, (0.75, -half-z - 0.5)), label[#code-label("half_size.x")])
  }

  // Row 2, right: the domain box seen from the front: liquid inside, closed sides solid outside, top open.
  {
    let low = (right-x + 0.7, 5.15)
    let high = (right-x + 4.3, 7.95)
    let thickness = 0.32
    panel-title((right-x, 9.0), [domain box])
    rect(low, high, fill: palette.air, stroke: none)
    // Closed sides: left, right and bottom, drawn as solid beyond the box.
    rect((low.at(0) - thickness, low.at(1) - thickness), (low.at(0), high.at(1)), fill: fill-solid, stroke: none)
    rect((high.at(0), low.at(1) - thickness), (high.at(0) + thickness, high.at(1)), fill: fill-solid, stroke: none)
    rect((low.at(0), low.at(1) - thickness), (high.at(0), low.at(1)), fill: fill-solid, stroke: none)
    line((low.at(0), high.at(1)), low, (high.at(0), low.at(1)), high, stroke: surface-stroke)
    // The open top.
    line((low.at(0), high.at(1)), high, stroke: (paint: palette.solid, thickness: guide-stroke, dash: "dashed"))
    content(((low.at(0) + high.at(0)) / 2, high.at(1) + 0.08), anchor: "south", label[top (+Z), open])
    content((low.at(0) - thickness - 0.08, (low.at(1) + high.at(1)) / 2), anchor: "east", label[left \ (−X)])
    content((high.at(0) + thickness + 0.08, (low.at(1) + high.at(1)) / 2), anchor: "west", label[right \ (+X)])
    content(((low.at(0) + high.at(0)) / 2, low.at(1) - thickness - 0.06), anchor: "north", label[bottom (−Z)])
    // Normals point into the box.
    normal-arrow(((low.at(0) + high.at(0)) / 2 + 0.9, low.at(1)), (0, 1), length: 0.55)
    normal-arrow((low.at(0), low.at(1) + 1.6), (1, 0), length: 0.55)
    normal-arrow((high.at(0), low.at(1) + 1.6), (-1, 0), length: 0.55)
    content(
      ((low.at(0) + high.at(0)) / 2, (low.at(1) + high.at(1)) / 2 + 0.55),
      align(center, label[front (−Y) and back (+Y): \ before and behind the page]),
    )
  }
})
