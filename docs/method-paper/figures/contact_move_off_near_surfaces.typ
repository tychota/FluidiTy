// Figure <fig-move-off-near-surfaces>: the start rules' move off several near surfaces at once.
//
// Left: a wedge of 60 degrees, the particle resting at its apex on both faces. Faint: the rule of one move per solid,
// up off the floor to its rest gap, which takes the start tol / 2 into the other face, then along that face's normal,
// back to 0.25 tol from the floor. Bold: the one move to the point at tol from both, 2 tol along the bisector.
// Right: the cusp under a ball of radius R = 12 tol resting on the floor, drawn to scale. The start x0 is short of
// both surfaces; the first iterate, from the two tangent planes at x0, overshoots far out along the floor; the next
// ones come back to the rest point 2 Sqrt(R tol) = 6.93 tol from the contact (iterates of the method's model: 23.6,
// 9.2, 7.1, 6.93 tol).

#import "../figure_style.typ": *

#let move-off-near-surfaces-figure() = figure-canvas({
  import cetz.draw: *
  let label = text.with(size: inset-label-size)
  let faint = palette.muted.darken(20%)
  let rest-gap-stroke = (paint: palette.solid, thickness: guide-stroke, dash: "dashed")

  // Left panel: the 60 degree wedge, one tolerance drawn as `wedge-tol` units.
  let wedge-tol = 0.8
  let root-three = calc.sqrt(3)
  let top = 2.6
  rect((-0.4, -0.45), (3.3, 0), fill: fill-solid, stroke: none)
  line((0, 0), (top / root-three, top), (-0.4, top), (-0.4, 0), close: true, fill: fill-solid, stroke: none)
  line((-0.4, 0), (3.3, 0), stroke: surface-stroke)
  line((0, 0), (top / root-three, top), stroke: surface-stroke)
  // The rest gap of each face, and the point where they meet.
  let rest-point = (root-three * wedge-tol, wedge-tol)
  line(rest-point, (3.3, wedge-tol), stroke: rest-gap-stroke)
  line(rest-point, ((top + 2 * wedge-tol) / root-three, top), stroke: rest-gap-stroke)
  // Faint: one move per solid in turn.
  let after-floor = (0, wedge-tol)
  let after-face = (1.5 * wedge-tol * root-three / 2, wedge-tol - 0.75 * wedge-tol)
  coloured-arrow((0, 0), after-floor, colour: faint, stroke-width: guide-stroke, small: true)
  coloured-arrow(after-floor, after-face, colour: faint, stroke-width: guide-stroke, small: true)
  particle-dot(after-face, colour: palette.muted)
  content(after-face, anchor: "west", padding: 0.14, text(size: inset-label-size, fill: faint.darken(30%))[0.25 tol])
  // Bold: one move off both faces.
  coloured-arrow((0, 0), rest-point, colour: palette.highlight)
  particle-dot((0, 0))
  particle-dot(rest-point)
  content(rest-point, anchor: "north-west", padding: 0.12, label[tol from both])
  content((3.25, wedge-tol), anchor: "south-east", padding: 0.06, label[rest gap])
  content((-0.4, top + 0.1), anchor: "south-west", text(weight: "bold")[60° wedge])

  // Right panel: the cusp under a ball of radius 12 tol on the floor, one tolerance drawn as `cusp-tol` units.
  let cusp-tol = 0.28
  let origin-x = 4.0
  let right-x = 10.95
  let radius = 12 * cusp-tol
  let at(x-tol, z-tol) = (origin-x + x-tol * cusp-tol, z-tol * cusp-tol)
  rect((origin-x - 0.25, -0.45), (right-x, 0), fill: fill-solid, stroke: none)
  line((origin-x - 0.25, 0), (right-x, 0), stroke: surface-stroke)
  // The part of the ball from 1 tol before the contact to 9 tol after it, cut off above.
  let ball-points = range(0, 41).map(step => {
    let x = (-1 + 10 * step / 40) * cusp-tol
    (origin-x + x, radius - calc.sqrt(radius * radius - x * x))
  })
  let cut-height = ball-points.last().at(1) + 0.3
  line(
    ..ball-points,
    (ball-points.last().at(0), cut-height),
    (ball-points.first().at(0), cut-height),
    close: true,
    fill: fill-solid,
    stroke: none,
  )
  line(..ball-points, stroke: surface-stroke)
  // The rest gap of the floor, and the rest point.
  let rest-x = 2 * calc.sqrt(12)
  line(at(rest-x, 1), (right-x, cusp-tol), stroke: rest-gap-stroke)
  // The iterates, measured from the start x0 = 1.04 tol from the contact, 0.018 tol above the floor.
  let start = at(1.039, 0.018)
  let iterates = ((23.622, 1), (9.218, 1), (7.114, 1), (rest-x, 1))
  for (index, iterate) in iterates.enumerate() {
    let colour = if index == 0 { faint } else { palette.highlight }
    coloured-arrow(start, at(..iterate), colour: colour, stroke-width: guide-stroke, small: true)
  }
  particle-dot(start)
  content((start.at(0), -0.08), anchor: "north", label[_x_#sub[0]])
  for (index, iterate) in iterates.slice(0, 3).enumerate() {
    particle-dot(at(..iterate), colour: if index == 0 { palette.muted } else { palette.particle })
    content(at(..iterate), anchor: "south", padding: 0.12, label[_x_#sub[#(index + 1)]])
  }
  particle-dot(at(rest-x, 1))
  content((at(rest-x, 1).at(0), -0.08), anchor: "north", label[rest point, _x_#sub[4]])
  content((origin-x - 0.25, cut-height + 0.1), anchor: "south-west", text(weight: "bold")[cusp under a ball])
  content(at(16, 1), anchor: "south", padding: 0.06, label[rest gap])
})
