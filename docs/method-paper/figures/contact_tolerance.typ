// Figure <fig-contact-tolerance>: a zoom on a solid surface, not to scale. The band where rounding can misplace the
// surface, the rest gap at d = tol where a particle in contact sits, the end check at d = tol / 2, and on the right a
// plate 1024 tolerances thick, the thinnest feature the contact tolerance accepts, whose top face the zoom enlarges.

#import "../figure_style.typ": *

#let contact-tolerance-figure() = figure-canvas({
  import cetz.draw: *

  let label = text.with(size: inset-label-size)
  // The zoom frame and its vertical scale: one tolerance is `tol-height` units.
  let frame-low = (0, 0.2)
  let frame-high = (7.6, 4.5)
  let surface-y = 1.2
  let tol-height = 2.4
  let band-half-height = 0.12
  let y-of(distance-in-tolerances) = surface-y + distance-in-tolerances * tol-height

  // The solid below the surface and the rounding band around it.
  rect(frame-low, (frame-high.at(0), surface-y), fill: fill-solid, stroke: none)
  rect(
    (frame-low.at(0), surface-y - band-half-height),
    (frame-high.at(0), surface-y + band-half-height),
    fill: palette.highlight.lighten(70%),
    stroke: none,
  )
  line((frame-low.at(0), surface-y), (frame-high.at(0), surface-y), stroke: surface-stroke)
  content((0.15, 0.45), anchor: "west", label[solid, _d_ < 0])
  content((frame-high.at(0) - 0.12, surface-y - band-half-height - 0.06), anchor: "north-east", align(
    right,
    label[rounding band: _d_ computed with plain doubles \ is off by at most about 32 _ε_ _L_],
  ))

  // The end check and the rest gap.
  let check-y = y-of(0.5)
  let rest-y = y-of(1)
  line((1.2, check-y), (frame-high.at(0), check-y), stroke: (
    paint: palette.highlight,
    thickness: geometry-stroke,
    dash: "dashed",
  ))
  content((frame-high.at(0) - 0.12, check-y + 0.06), anchor: "south-east", text(
    size: inset-label-size,
    fill: palette.highlight,
  )[end check: every particle ends with _d_ ≥ tol / 2])
  line((1.2, rest-y), (frame-high.at(0), rest-y), stroke: guide-stroke + palette.solid)
  content((frame-high.at(0) - 0.12, rest-y - 0.06), anchor: "north-east", label[rest gap: a particle in contact sits
    at _d_ = tol])

  // Particles resting at the rest gap, and one arriving along its path, stopped at d = tol.
  coloured-arrow((6.0, 4.4), (5.27, rest-y + particle-radius + 0.02), colour: palette.liquid, small: true)
  for x in (3.4, 4.3, 5.2) {
    particle-dot((x, rest-y))
  }

  // Dimensions on the left: tol and tol / 2, measured from the surface.
  dimension-line((0.45, surface-y), (0.45, rest-y))
  content((0.55, (surface-y + rest-y) / 2 + 0.35), anchor: "west", label[tol])
  dimension-line((0.95, surface-y), (0.95, check-y))
  content((1.05, (surface-y + check-y) / 2), anchor: "west", label[tol / 2])
  content((0.12, 4.3), anchor: "north-west", label[_d_: distance to the solid])

  rect(frame-low, frame-high, stroke: guide-stroke + palette.muted.darken(25%))

  // Right: a plate 1024 tolerances thick, at a scale 1024 times smaller. The circle on its top face is the zoom.
  let plate-left = 9.0
  let plate-right = 10.1
  let plate-low = 0.5
  let plate-high = 3.6
  rect((plate-left, plate-low), (plate-right, plate-high), fill: fill-solid, stroke: surface-stroke)
  let zoom-centre = ((plate-left + plate-right) / 2, plate-high)
  circle(zoom-centre, radius: 0.16, stroke: guide-stroke + black)
  line((zoom-centre.at(0) - 0.12, zoom-centre.at(1) + 0.11), (frame-high.at(0), frame-high.at(1)), stroke: (
    guide-stroke + palette.muted.darken(25%)
  ))
  line((zoom-centre.at(0) - 0.15, zoom-centre.at(1) - 0.05), (frame-high.at(0), surface-y), stroke: (
    guide-stroke + palette.muted.darken(25%)
  ))
  dimension-line((plate-right + 0.25, plate-low), (plate-right + 0.25, plate-high))
  content((plate-right + 0.35, (plate-low + plate-high) / 2), anchor: "west", align(left, label[at least \ 1024 \ tol]))
  content(((plate-left + plate-right) / 2, plate-low - 0.1), anchor: "north", align(center, label[the thinnest \ plate
    allowed]))
})
