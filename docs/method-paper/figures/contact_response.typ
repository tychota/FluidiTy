// Figure <fig-contact-response>: one velocity meeting a fixed flat solid, split into its approach part (along the
// normal, into the solid) and its sideways part. The approach is removed; friction then takes friction x approach
// speed off the sideways speed. Left: friction 0.5, the particle slips on with sideways speed 2.5. Right: friction 3,
// friction would take off all 3 m/s, so the particle sticks and takes the solid's velocity.

#import "../figure_style.typ": *

// Draws one panel with its lower left corner at `corner`: the incoming velocity (3, -1) m/s on a fixed flat solid,
// its two parts, and what is left after the response with `friction`.
#let response-panel(corner, friction, title) = {
  import cetz.draw: *
  let label = text.with(size: inset-label-size)
  let red-label = text.with(size: inset-label-size, fill: palette.highlight)
  let at(point) = (corner.at(0) + point.at(0), corner.at(1) + point.at(1))
  let unit = 1.05
  let sideways-speed = 3
  let approach-speed = 1
  let particle = (0.45, 1.75)
  let surface-y = 0.45

  content(at((0, 3.1)), anchor: "north-west", text(weight: "bold", title))

  // The fixed flat solid and its normal.
  rect(at((0, -0.1)), at((5.3, surface-y)), fill: fill-solid, stroke: none)
  line(at((0, surface-y)), at((5.3, surface-y)), stroke: surface-stroke)
  coloured-arrow(at((4.95, surface-y)), at((4.95, surface-y + 0.6)), colour: palette.highlight, small: true)
  content(at((5.05, surface-y + 0.45)), anchor: "west", red-label[_n_])

  // The incoming velocity and its two parts.
  let sideways-tip = (particle.at(0) + sideways-speed * unit, particle.at(1))
  let velocity-tip = (sideways-tip.at(0), particle.at(1) - approach-speed * unit)
  let part-stroke = (paint: black, thickness: guide-stroke, dash: "dashed")
  line(at(particle), at(sideways-tip), stroke: part-stroke)
  line(at(sideways-tip), at(velocity-tip), stroke: part-stroke, mark: small-arrow-mark)
  coloured-arrow(at(particle), at(velocity-tip), colour: palette.liquid)
  content(at((1.6, 1.05)), anchor: "north", label[before: (3, −1) m/s])
  content(at((sideways-tip.at(0) + 0.1, particle.at(1) - 0.5)), anchor: "west", align(left, label[approach \ −1 m/s:
    \ removed]))

  // Friction takes friction x 1 m/s off the 3 m/s sideways speed, at most all of it.
  let taken = calc.min(friction * approach-speed, sideways-speed)
  let left-tip = (sideways-tip.at(0) - taken * unit, particle.at(1))
  line(at(left-tip), at(sideways-tip), stroke: 2.4 * geometry-stroke + palette.highlight.lighten(35%))
  content(at(((left-tip.at(0) + sideways-tip.at(0)) / 2, particle.at(1) + 0.12)), anchor: "south", red-label[
    friction takes #str(taken) m/s
  ])
  if taken < sideways-speed {
    coloured-arrow(
      at(particle),
      at(left-tip),
      colour: palette.particle.darken(15%),
      stroke-width: 1.4 * geometry-stroke,
    )
    content(at((particle.at(0), particle.at(1) + 0.55)), anchor: "south-west", label[after: (2.5, 0) m/s, slips])
  } else {
    content(at((particle.at(0), particle.at(1) + 0.55)), anchor: "south-west", label[after: (0, 0) m/s, the solid's
      velocity: sticks])
  }
  particle-dot(at(particle))
}

#let contact-response-figure() = figure-canvas({
  response-panel((0, 0), 0.5, [friction 0.5])
  response-panel((6.0, 0), 3, [friction 3])
})
