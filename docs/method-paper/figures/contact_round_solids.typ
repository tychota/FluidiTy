// Figures <fig-round-solids> and <fig-round-first-hit>: the sphere and the cylinder of contact.
//
// <fig-round-solids> draws the sphere with its centre, radius and one outward normal, and a cut of the cylinder
// through its own z axis with its radius, half height, the three distance regions outside it (wall, cap, rim)
// shaded, the face names the code reports, and the two points where the normal has no direction of its own (the
// sphere's centre, a point on the cylinder's axis).
//
// <fig-round-first-hit> draws the same two cuts grown by the contact tolerance, exaggerated so it shows: the sphere
// grown into a larger sphere, the cylinder grown into a larger cylinder with sharp edges. One path hits the sphere;
// three paths hit the cylinder through its wall, through its cap and in the sharp rim corner, where the hit is moved
// along the rim normal back to d = tol.

#import "../figure_style.typ": *

// Adds two 2D points.
#let offset(point, shift) = (point.at(0) + shift.at(0), point.at(1) + shift.at(1))

// Multiplies a 2D vector by a number.
#let scaled(vector, factor) = vector.map(value => value * factor)

// Draws a normal arrow from `foot` along the unit `direction`, `length` long, in the highlight colour.
#let normal-arrow(foot, direction, length: 0.6) = coloured-arrow(
  foot,
  offset(foot, scaled(direction, length)),
  colour: palette.highlight,
)

// Draws a panel title at `point`.
#let panel-title(point, body) = cetz.draw.content(point, anchor: "north-west", text(weight: "bold", body))

// Draws the solid's own x and z axes at `origin` as short guide arrows labelled in italics.
#let own-axes(origin, length: 0.45) = {
  import cetz.draw: *
  for (direction, name) in (((1, 0), [_x_]), ((0, 1), [_z_])) {
    line(origin, offset(origin, scaled(direction, length)), stroke: guide-stroke + black, mark: small-arrow-mark)
    content(offset(origin, scaled(direction, length + 0.18)), text(size: inset-label-size, name))
  }
  circle(origin, radius: 0.035, fill: black, stroke: none)
}

// The dashed stroke of a grown shape and of region boundaries.
#let grown-stroke = (paint: palette.solid, thickness: guide-stroke, dash: "dashed")
#let region-stroke = (paint: palette.muted.darken(20%), thickness: guide-stroke, dash: "dashed")

#let round-solids-figure() = figure-canvas({
  import cetz.draw: *

  let label = text.with(size: inset-label-size)
  let red-label = text.with(size: inset-label-size, fill: palette.highlight)

  // Left: the sphere, its own axes at the centre c, one normal, and the rule at the centre.
  {
    let centre = (1.7, 2.0)
    let radius = 1.1
    panel-title((0, 4.3), [sphere])
    circle(centre, radius: radius, fill: fill-solid, stroke: surface-stroke)
    own-axes(centre)
    content(offset(centre, (-0.12, 0.14)), anchor: "south-east", label[_c_])
    let direction = (calc.cos(40deg), calc.sin(40deg))
    let foot = offset(centre, scaled(direction, radius))
    normal-arrow(foot, direction)
    content(offset(foot, (0.5, 0.42)), anchor: "west", align(left, red-label[normal \ (_x_ − _c_) / |_x_ − _c_|]))
    content(offset(foot, (0.3, -0.22)), anchor: "west", label[#code-label("kCurved")])
    let edge-point = offset(centre, (-radius * calc.cos(35deg), -radius * calc.sin(35deg)))
    dimension-line(centre, edge-point)
    content(offset(centre, (-0.3, -0.36)), anchor: "north-west", label[#code-label("radius")])
    // The degenerate normal at the centre: minus the own x axis.
    let degenerate-arrow = coloured-arrow.with(colour: palette.highlight, stroke-width: guide-stroke, small: true)
    degenerate-arrow(centre, offset(centre, (-0.5, 0)))
    let centre-note = label[at the centre: \ normal −_x_]
    content(offset(centre, (0.0, -radius - 0.12)), anchor: "north", align(center, centre-note))
  }

  // Right: the cylinder cut through its axis, the three distance regions outside it and a normal in each.
  {
    let origin = (8.0, 2.0)
    let radius = 1.15
    let half-height = 0.75
    let reach = 0.85
    panel-title((4.6, 4.3), [cylinder, cut through its axis])
    for sign-x in (-1, 1) {
      for sign-z in (-1, 1) {
        rect(
          offset(origin, (sign-x * radius, sign-z * half-height)),
          offset(origin, (sign-x * (radius + reach), sign-z * (half-height + reach * 0.8))),
          fill: palette.particle.lighten(85%),
          stroke: none,
        )
      }
    }
    // The wall regions (outside the radius, between the cap planes) and the cap regions (beyond a cap, within the
    // radius), each in its own tint.
    for sign-x in (-1, 1) {
      rect(
        offset(origin, (sign-x * radius, -half-height)),
        offset(origin, (sign-x * (radius + reach), half-height)),
        fill: palette.air,
        stroke: none,
      )
    }
    for sign-z in (-1, 1) {
      rect(
        offset(origin, (-radius, sign-z * half-height)),
        offset(origin, (radius, sign-z * (half-height + reach * 0.8))),
        fill: palette.highlight.lighten(88%),
        stroke: none,
      )
    }
    for sign-z in (-1, 1) {
      line(
        offset(origin, (-radius - reach, sign-z * half-height)),
        offset(origin, (radius + reach, sign-z * half-height)),
        stroke: region-stroke,
      )
    }
    for sign-x in (-1, 1) {
      line(
        offset(origin, (sign-x * radius, -half-height - reach * 0.8)),
        offset(origin, (sign-x * radius, half-height + reach * 0.8)),
        stroke: region-stroke,
      )
    }
    rect(offset(origin, (-radius, -half-height)), offset(origin, (radius, half-height)), fill: fill-solid, stroke: (
      surface-stroke
    ))
    // The axis, dash-dotted through the whole cut.
    line(offset(origin, (0, -half-height - 0.6)), offset(origin, (0, half-height + 0.75)), stroke: (
      paint: black,
      thickness: guide-stroke,
      dash: "dash-dotted",
    ))
    content(offset(origin, (0.08, half-height + 0.75)), anchor: "south", label[#code-label("axis")])
    own-axes(origin, length: 0.4)
    // Wall: radial normal. Cap: +z normal. Rim: from the rim circle toward the point.
    normal-arrow(offset(origin, (-radius, 0.2)), (-1, 0), length: 0.55)
    content(offset(origin, (-radius - 0.08, 0.2 - 0.12)), anchor: "north-east", align(right, label[wall \ #code-label(
        "kCurved",
      )]))
    normal-arrow(offset(origin, (-1.02, half-height)), (0, 1), length: 0.5)
    content(offset(origin, (-0.98, half-height + 0.35)), anchor: "west", align(left, label[cap \ #code-label(
        "kHighZ",
      )]))
    normal-arrow(offset(origin, (radius, half-height)), (0.6, 0.8), length: 0.55)
    content(offset(origin, (radius + 0.4, half-height + 0.5)), anchor: "west", align(left, label[rim \ #code-label(
        "kHighRim",
      )]))
    content(offset(origin, (radius + 0.45, -half-height - 0.35)), anchor: "west", label[#code-label("kLowRim")])
    content(offset(origin, (0.1, -half-height - 0.35)), anchor: "west", label[#code-label("kLowZ")])
    // Radius and half height.
    dimension-line(offset(origin, (0, -0.22)), offset(origin, (radius, -0.22)))
    content(offset(origin, (radius / 2 + 0.05, -0.26)), anchor: "north", label[#code-label("radius")])
    let extension-stroke = guide-stroke + palette.muted.darken(30%)
    line(offset(origin, (radius - 0.05, 0)), offset(origin, (radius + 0.4, 0)), stroke: extension-stroke)
    dimension-line(offset(origin, (radius + 0.3, 0)), offset(origin, (radius + 0.3, -half-height)))
    content(offset(origin, (radius + 0.38, -half-height / 2)), anchor: "west", label[#code-label("half_height")])
    // The degenerate normal on the axis: minus the own x axis.
    let axis-point = offset(origin, (0, -0.45))
    circle(axis-point, radius: 0.04, fill: palette.highlight, stroke: none)
    let degenerate-arrow = coloured-arrow.with(colour: palette.highlight, stroke-width: guide-stroke, small: true)
    degenerate-arrow(axis-point, offset(axis-point, (-0.45, 0)))
    content(offset(axis-point, (-0.5, 0)), anchor: "east", red-label[−_x_])
  }
})

#let round-first-hit-figure() = figure-canvas({
  import cetz.draw: *

  let label = text.with(size: inset-label-size)
  let red-label = text.with(size: inset-label-size, fill: palette.highlight)
  // The tolerance, exaggerated so it shows.
  let tol = 0.5

  // Left: the sphere grown by tol, and one path stopped where it meets the grown sphere.
  {
    let centre = (2.9, 2.0)
    let radius = 1.1
    let grown = radius + tol
    panel-title((0, 4.75), [sphere grown by tol])
    circle(centre, radius: radius, fill: fill-solid, stroke: surface-stroke)
    circle(centre, radius: grown, stroke: grown-stroke)
    let path-height = 0.55
    let start = (0.2, centre.at(1) + path-height)
    let hit-x = centre.at(0) - calc.sqrt(grown * grown - path-height * path-height)
    let hit = (hit-x, start.at(1))
    coloured-arrow(start, offset(hit, (-particle-radius - 0.02, 0)), colour: palette.liquid)
    particle-dot(start)
    particle-dot(hit)
    let normal = scaled((hit-x - centre.at(0), path-height), 1 / grown)
    normal-arrow(hit, normal, length: 0.5)
    content(offset(hit, (-0.15, 0.55)), anchor: "south", label[hit: _d_ = tol])
    let low-direction = (calc.cos(-60deg), calc.sin(-60deg))
    dimension-line(offset(centre, scaled(low-direction, radius)), offset(centre, scaled(low-direction, grown)))
    content(offset(centre, scaled(low-direction, grown + 0.1)), anchor: "north-west", label[tol])
  }

  // Right: the cylinder grown with sharp edges, three paths, and the rim move.
  {
    let origin = (8.4, 1.9)
    let radius = 1.0
    let half-height = 0.85
    let grown-radius = radius + tol
    let grown-half-height = half-height + tol
    panel-title((5.0, 4.75), [cylinder grown with sharp edges])
    rect(offset(origin, (-radius, -half-height)), offset(origin, (radius, half-height)), fill: fill-solid, stroke: (
      surface-stroke
    ))
    let grown-corner = (grown-radius, grown-half-height)
    rect(offset(origin, scaled(grown-corner, -1)), offset(origin, grown-corner), stroke: grown-stroke)
    // The points at true distance tol round the high rim: a quarter circle about the rim point.
    let rim-point = offset(origin, (radius, half-height))
    arc(offset(rim-point, (tol, 0)), start: 0deg, stop: 90deg, radius: tol, stroke: (
      paint: palette.highlight,
      thickness: guide-stroke,
    ))
    // Path 1, through the wall.
    let wall-start = (5.4, origin.at(1) - 0.3)
    let wall-hit = (origin.at(0) - grown-radius, wall-start.at(1))
    coloured-arrow(wall-start, offset(wall-hit, (-particle-radius - 0.02, 0)), colour: palette.liquid)
    particle-dot(wall-start)
    particle-dot(wall-hit)
    content(offset(wall-start, (0.1, 0.12)), anchor: "south-west", label[wall])
    // Path 2, through the cap.
    let cap-start = (origin.at(0) - 0.35, 4.05)
    let cap-hit = (cap-start.at(0), origin.at(1) + grown-half-height)
    coloured-arrow(cap-start, offset(cap-hit, (0, particle-radius + 0.02)), colour: palette.liquid)
    particle-dot(cap-start)
    particle-dot(cap-hit)
    content(offset(cap-start, (-0.15, 0)), anchor: "east", label[cap])
    // Path 3, parallel to the axis into the sharp rim corner, then moved to d = tol along the rim normal.
    let rim-start = (origin.at(0) + grown-radius - 0.03, 4.05)
    let rim-hit = (rim-start.at(0), origin.at(1) + grown-half-height)
    coloured-arrow(rim-start, offset(rim-hit, (0, particle-radius + 0.02)), colour: palette.liquid)
    particle-dot(rim-start)
    content(offset(rim-start, (0.15, 0)), anchor: "west", label[rim corner])
    let from-rim = (rim-hit.at(0) - rim-point.at(0), rim-hit.at(1) - rim-point.at(1))
    let from-rim-length = calc.sqrt(from-rim.at(0) * from-rim.at(0) + from-rim.at(1) * from-rim.at(1))
    let moved = offset(rim-point, scaled(from-rim, tol / from-rim-length))
    line(rim-point, rim-hit, stroke: guide-stroke + palette.muted.darken(30%))
    particle-dot(rim-hit, colour: white)
    particle-dot(moved)
    content(offset(rim-hit, (0.32, 0.05)), anchor: "west", align(left, red-label[rim move \ to _d_ = tol]))
    // Dimensions: tol on the low wall side.
    dimension-line(offset(origin, (-radius, -half-height - 0.15)), offset(origin, (-grown-radius, -half-height - 0.15)))
    content(offset(origin, (-grown-radius - 0.05, -half-height - 0.15)), anchor: "east", label[tol])
  }
})
