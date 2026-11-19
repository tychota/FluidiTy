// Figure <fig-nearby-faces>: the nearby faces of one axis (X) for one particle, in a vertical slice (X to the right, Z
// up). Left, a particle in a big cell beside smaller cells, with a plate on its other side: the candidates follow the
// rule of <proc-nearby-face-weights> in two dimensions, the faces beyond the plate are left out (hollow), and each kept
// face is a dot whose area is its scaled weight (with a smallest size, so a light face still shows). The arrow is the
// mean offset and the ellipse the weighted spread around the mean, its half axes the square roots of the spread's
// eigenvalues. Right, the complete case of one size: the mean offset is zero and the spread h^2 / 4 on every axis, a
// circle of radius h / 2. Every number is computed here from the procedure's formulas, in two dimensions, where a
// face's area over h^2 becomes its length over h.

#import "../figure_style.typ": *

// The weight curve of <eq:weight-curve>.
#let weight-curve(distance) = {
  let size = calc.abs(distance)
  if size < 0.5 { 0.75 - size * size } else if size < 1.5 { (1.5 - size) * (1.5 - size) / 2 } else { 0 }
}

// Whether two cells share a side or a corner: their closed boxes meet and they are not the same cell.
#let is-touching(first, second) = {
  let (first-x, first-y, first-width) = cell-box(first)
  let (second-x, second-y, second-width) = cell-box(second)
  let meets-x = first-x <= second-x + second-width and second-x <= first-x + first-width
  let meets-y = first-y <= second-y + second-width and second-y <= first-y + first-width
  meets-x and meets-y and first != second
}

// The candidates of the X axis for `position`, in the order of `cell-grid-face-pieces`: steps 1 and 2 of the
// procedure, with the half-open distance test -1.5 h < offset <= 1.5 h on each axis. Returns the pieces with their
// offset from the position, in widths of a size-level-0 cell, and the containing cell's width.
#let find-candidates(cells, position) = {
  let containing = cells.at(find-cell-at(cells, position))
  let (_, _, cell-width) = cell-box(containing)
  let size-level = containing.at(2)
  let near-cells = cells.filter(cell => cell == containing or is-touching(cell, containing))
  let belongs-near(piece) = (piece.cell-low, piece.cell-high)
    .filter(index => index != none)
    .any(index => near-cells.contains(cells.at(index)))
  cell-grid-face-pieces(cells)
    .filter(piece => piece.axis == 0)
    .filter(piece => calc.abs(piece.size-level - size-level) <= 1)
    .filter(belongs-near)
    .map(piece => {
      let offset = (piece.centre.at(0) - position.at(0), piece.centre.at(1) - position.at(1))
      piece + (offset: offset, cell-width: cell-width)
    })
    .filter(piece => piece.offset.all(value => -1.5 * cell-width < value and value <= 1.5 * cell-width))
}

// Steps 4 to 8 for the kept candidates: each kept piece with its scaled weight, the mean offset and the weighted
// spread (a, b, c) = ((a, b), (b, c)).
#let compute-weights(kept) = {
  let raw = kept.map(piece => {
    let distance = piece.offset.map(value => value / piece.cell-width)
    weight-curve(distance.at(0)) * weight-curve(distance.at(1)) * piece.length / piece.cell-width
  })
  let weight-sum = raw.sum()
  let weighted = kept.zip(raw).map(((piece, raw-weight)) => piece + (weight: raw-weight / weight-sum))
  let mean = (0, 1).map(axis => weighted.map(piece => piece.weight * piece.offset.at(axis)).sum())
  let spread-entry(first, second) = weighted
    .map(piece => {
      let from-mean = (piece.offset.at(0) - mean.at(0), piece.offset.at(1) - mean.at(1))
      piece.weight * from-mean.at(first) * from-mean.at(second)
    })
    .sum()
  (weighted: weighted, mean: mean, spread: (spread-entry(0, 0), spread-entry(0, 1), spread-entry(1, 1)))
}

// Draws one panel: the cells, the kept faces as filled dots of area proportional to weight, the left-out faces
// hollow, the particle, the mean offset arrow and the spread ellipse. `unit` is the width of a size-level-0 cell.
#let draw-nearby-faces(origin, unit, cells, position, is-kept) = {
  import cetz.draw: *
  cell-grid(origin, cells, unit: unit)
  let candidates = find-candidates(cells, position)
  let kept = candidates.filter(is-kept)
  let left-out = candidates.filter(piece => not is-kept(piece))
  let result = compute-weights(kept)
  let to-drawing(point) = cell-grid-point(origin, unit, point)
  for piece in left-out {
    circle(to-drawing(piece.centre), radius: 0.07, fill: white, stroke: guide-stroke + palette.liquid.darken(20%))
  }
  for piece in result.weighted {
    let dot-radius = calc.max(0.36 * calc.sqrt(piece.weight), 0.035)
    circle(to-drawing(piece.centre), radius: dot-radius, fill: palette.liquid, stroke: none)
  }
  // The spread ellipse around the mean: half axes are the square roots of the eigenvalues.
  let (spread-xx, spread-xy, spread-yy) = result.spread
  let half-sum = (spread-xx + spread-yy) / 2
  let half-difference = (spread-xx - spread-yy) / 2
  let root = calc.sqrt(half-difference * half-difference + spread-xy * spread-xy)
  let major = calc.sqrt(half-sum + root)
  let minor = calc.sqrt(calc.max(half-sum - root, 0))
  let turn = calc.atan2(half-difference, spread-xy) / 2
  let mean-point = (position.at(0) + result.mean.at(0), position.at(1) + result.mean.at(1))
  group({
    set-origin(to-drawing(mean-point))
    rotate(turn)
    circle((0, 0), radius: (major * unit, minor * unit), stroke: (
      paint: palette.highlight,
      thickness: guide-stroke,
      dash: "dashed",
    ))
  })
  let mean-length = calc.sqrt(result.mean.at(0) * result.mean.at(0) + result.mean.at(1) * result.mean.at(1))
  if mean-length > 0.02 {
    coloured-arrow(to-drawing(position), to-drawing(mean-point), colour: palette.highlight, small: true)
  }
  particle-dot(to-drawing(position))
}

#let nearby-faces-figure() = figure-canvas({
  import cetz.draw: *

  let label = text.with(size: inset-label-size)

  // Left: big cells of size level 0 on x 0 to 2, size-level-1 cells on x 2 to 4, z 0 to 3. The particle is in a big
  // cell beside the smaller cells; a plate crosses the big cells just left of it, so the faces at x = 0 and x = 1
  // are left out.
  let unit = 1.45
  let left-origin = (0, 0)
  let big-cells = range(3).map(z => range(2).map(x => (x, z, 0))).flatten().chunks(3)
  let small-cells = range(6).map(z => range(4, 8).map(x => (x, z, 1))).flatten().chunks(3)
  let cells = big-cells + small-cells
  let position = (1.34, 1.45)
  let plate-top = (1.16, 3.0)
  let plate-bottom = (1.06, 0.0)
  // A face is on the particle's side of the plate when it lies right of the plate's line at its height.
  let plate-x-at(height) = plate-bottom.at(0) + (plate-top.at(0) - plate-bottom.at(0)) * height / 3.0
  let is-kept(piece) = piece.centre.at(0) > plate-x-at(piece.centre.at(1))
  draw-nearby-faces(left-origin, unit, cells, position, is-kept)
  line(
    cell-grid-point(left-origin, unit, plate-bottom),
    cell-grid-point(left-origin, unit, plate-top),
    stroke: 2.2 * geometry-stroke + palette.solid,
  )
  content(cell-grid-point(left-origin, unit, (plate-top.at(0), 3.0)), anchor: "south", label[plate])
  content(cell-grid-point(left-origin, unit, (2.0, -0.08)), anchor: "north", label[a size change and a plate])

  // Right: the complete case, nine faces of one size, all kept.
  let right-unit = 1.0
  let right-origin = (7.2, 0.5)
  let uniform = range(3).map(z => range(3).map(x => (x, z, 0))).flatten().chunks(3)
  draw-nearby-faces(right-origin, right-unit, uniform, (1.25, 1.6), piece => true)
  content(cell-grid-point(right-origin, right-unit, (1.5, -0.08)), anchor: "north", align(center, label[the
    complete case: mean offset 0, \ spread _h_#super[2] / 4: a circle of radius _h_ / 2]))
})
