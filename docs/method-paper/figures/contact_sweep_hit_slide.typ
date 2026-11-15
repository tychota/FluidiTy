// Figure <fig-sweep-hit-slide>: one step of one particle against a flat solid that rises at 0.5 m/s, seen from the
// front. The straight sweep along velocity.for_this_step, the hit one tolerance above the moving surface, the
// response of both velocities, and the slide that rides up with the surface for the rest of the step. The tolerance
// is drawn far larger than it is (0.12 m) so it can be seen; every other number is the worked example of the text.

#import "../figure_style.typ": *

// Adds two 2D points.
#let offset-point(point, shift) = (point.at(0) + shift.at(0), point.at(1) + shift.at(1))

#let sweep-hit-slide-figure() = figure-canvas({
  import cetz.draw: *

  let label = text.with(size: inset-label-size)
  // The example, in metres and seconds.
  let step-duration = 1.0
  let solid-speed = 0.5  // Up, along the normal (0, 0, 1).
  let tolerance = 0.12   // Drawn, not real.
  let friction = 0.25
  let start = (0.0, 1.2)
  let for-this-step = (2.5, -1.5)
  let for-next-step = (1.5, -2.5)
  // Hit: the gap 1.2 - tol closes at 1.5 + 0.5 = 2 m/s, so at 0.54 s.
  let hit-time = (start.at(1) - tolerance) / (solid-speed - for-this-step.at(1))
  let hit = (start.at(0) + for-this-step.at(0) * hit-time, start.at(1) + for-this-step.at(1) * hit-time)
  // The response of each velocity (section text): approach removed, then friction on the sideways part.
  let respond(velocity) = {
    let approach-speed = velocity.at(1) - solid-speed
    let sideways = velocity.at(0)
    (sideways - friction * (-approach-speed), solid-speed)
  }
  let this-after = respond(for-this-step)
  let next-after = respond(for-next-step)
  let time-left = step-duration - hit-time
  let end = (hit.at(0) + this-after.at(0) * time-left, hit.at(1) + this-after.at(1) * time-left)
  let free-end = (start.at(0) + for-this-step.at(0) * step-duration, start.at(1) + for-this-step.at(1) * step-duration)

  // Drawing scales: positions 2.6 units per metre; velocities 0.45 units per m/s.
  let place(point) = (1.7 + 2.6 * point.at(0), 2.6 * point.at(1))
  let speed-scale = 0.45
  let tip(point, velocity) = {
    let (x, y) = place(point)
    (x + speed-scale * velocity.at(0), y + speed-scale * velocity.at(1))
  }
  let left-edge = 0
  let right-edge = 11.4

  // The solid at the end of the step, the surface at the start and at the hit.
  let surface-end-y = place((0, solid-speed * step-duration)).at(1)
  let surface-hit-y = place((0, solid-speed * hit-time)).at(1)
  let surface-start-y = place((0, 0)).at(1)
  let bottom-y = place((0, free-end.at(1) - 0.25)).at(1)
  // The solid as it is at the start, and lighter, the layer its surface rises through during the step.
  rect((left-edge, bottom-y), (right-edge, surface-start-y), fill: fill-solid, stroke: none)
  rect((left-edge, surface-start-y), (right-edge, surface-end-y), fill: fill-solid.lighten(55%), stroke: none)
  line((left-edge, surface-end-y), (right-edge, surface-end-y), stroke: surface-stroke)
  let guide = (paint: palette.solid, thickness: guide-stroke, dash: "dashed")
  line((left-edge, surface-hit-y), (right-edge, surface-hit-y), stroke: guide)
  line((left-edge, surface-start-y), (right-edge, surface-start-y), stroke: guide)
  content((right-edge - 0.1, surface-end-y + 0.06), anchor: "south-east", label[surface at the end])
  content((right-edge - 0.1, surface-hit-y + 0.05), anchor: "south-east", label[surface at the hit])
  content((right-edge - 0.1, surface-start-y + 0.05), anchor: "south-east", label[surface at the start])
  coloured-arrow((0.35, bottom-y + 0.1), (0.35, bottom-y + 0.1 + 2.6 * solid-speed), colour: palette.solid)
  content((0.48, bottom-y + 0.45), anchor: "west", label[solid moves \ up 0.5 m/s])

  // The sweep: dashed beyond the hit, where it would have gone without contact.
  let sweep-stroke = (paint: palette.liquid, thickness: geometry-stroke, dash: "dashed")
  line(place(start), place(hit), stroke: geometry-stroke + palette.liquid)
  line(place(hit), place(free-end), stroke: sweep-stroke)
  circle(place(free-end), radius: particle-radius, fill: white, stroke: 0.5 * geometry-stroke + palette.liquid)
  content(offset-point(place(free-end), (0.18, 0)), anchor: "west", label[end without contact])

  // The slide after the hit, riding up with the surface. It follows for_this_step, so it takes that colour, solid
  // and heavier than the sweep.
  line(place(hit), place(end), stroke: 1.6 * geometry-stroke + palette.liquid.darken(10%))
  content(
    ((place(hit).at(0) + place(end).at(0)) / 2 + 0.05, (place(hit).at(1) + place(end).at(1)) / 2 + 0.1),
    anchor: "south-east",
    label[slide],
  )

  // The rest gap at the hit.
  dimension-line((place(hit).at(0) - 0.12, surface-hit-y), (place(hit).at(0) - 0.12, place(hit).at(1) - 0.06))
  content((place(hit).at(0) - 0.22, (surface-hit-y + place(hit).at(1)) / 2), anchor: "east", label[tol])

  // Velocities: before at the start, after at the end. for_this_step in the liquid colour, for_next_step in the
  // particle colour.
  let this-colour = palette.liquid.darken(10%)
  let next-colour = palette.particle.darken(25%)
  coloured-arrow(place(start), tip(start, for-this-step), colour: this-colour)
  coloured-arrow(place(start), tip(start, for-next-step), colour: next-colour)
  content(offset-point(tip(start, for-this-step), (0.08, 0.05)), anchor: "west", text(
    size: inset-label-size,
    fill: this-colour,
  )[#code-label("for_this_step") (2.5, −1.5)])
  content(offset-point(tip(start, for-next-step), (-0.08, -0.05)), anchor: "north-east", text(
    size: inset-label-size,
    fill: next-colour,
  )[#code-label("for_next_step") \ (1.5, −2.5)])
  coloured-arrow(place(end), tip(end, this-after), colour: this-colour)
  coloured-arrow(place(end), tip(end, next-after), colour: next-colour)
  content(offset-point(tip(end, this-after), (0.08, 0.02)), anchor: "west", text(
    size: inset-label-size,
    fill: this-colour,
  )[(2, 0.5)])
  content(offset-point(tip(end, next-after), (-0.05, 0.1)), anchor: "south", text(
    size: inset-label-size,
    fill: next-colour,
  )[(0.75, 0.5)])

  // The three particle positions, drawn last.
  particle-dot(place(start))
  particle-dot(place(hit), colour: palette.highlight)
  particle-dot(place(end))
  content(offset-point(place(start), (-0.15, 0.05)), anchor: "east", label[start])
  content((place(hit).at(0) - 0.1, surface-hit-y - 0.06), anchor: "north-east", label[hit, 0.54 s])
  content(offset-point(place(end), (-0.35, 0.08)), anchor: "south", label[end])
})
