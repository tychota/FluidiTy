// Figure <fig-parallel-build>: numbering the cell blocks in parallel. A vertical slice holds six active cells of two
// sizes, numbered in input order, and the split cell they add; a dashed line marks x = 0, where block A ends and
// blocks B and C start. Below it, four lines of boxes: the block key of each
// entry in input order, the entries after the stable sort by block key, the marks (1 on the first entry of each run)
// and their exclusive sums. An arrow goes from each marked entry to the block index its sum gives.

#import "../figure_style.typ": *

#let parallel-build-figure() = figure-canvas({
  import cetz.draw: *

  let label = text.with(size: inset-label-size)

  // The slice treats its cells as the whole grid. Each entry is (x, z, size level, block, is the added split cell).
  // Blocks: A is the size-level-0 block with low corner x = -4; B the size-level-0 block at x = 0; C the
  // size-level-1 block at x = 0, which comes after B because at one low corner the bigger cells come first.
  let entries = (
    (3, 1, 1, "C", false),
    (0, 0, 0, "B", false),
    (2, 0, 1, "C", false),
    (-1, 0, 0, "A", false),
    (3, 0, 1, "C", false),
    (2, 1, 1, "C", false),
    (1, 0, 0, "B", true),
  )
  let block-names = ("A", "B", "C")
  let block-rank(entry) = block-names.position(name => name == entry.at(3))

  // The stable order by block key: ties keep their input order.
  let sorted-indexes = range(entries.len()).sorted(key: index => block-rank(entries.at(index)))
  let marks = sorted-indexes
    .enumerate()
    .map(((place, index)) => {
      let is-first = place == 0 or block-rank(entries.at(sorted-indexes.at(place - 1))) != block-rank(entries.at(index))
      if is-first { 1 } else { 0 }
    })
  let sums = range(marks.len() + 1).map(place => marks.slice(0, place).sum(default: 0))

  // Layout of the four lines of boxes.
  let first-x = 2.9
  let spacing = 1.05
  let box-width = 0.8
  let box-height = 0.5
  let input-y = 6.0
  let sorted-y = 4.4
  let mark-y = 3.5
  let sum-y = 2.75
  let block-y = 1.0
  let place-x(place) = first-x + spacing * place

  let entry-box(name, centre, body, box-fill: fill-plain, box-stroke: guide-stroke + black) = {
    let (centre-x, centre-y) = centre
    rect(
      (centre-x - box-width / 2, centre-y - box-height / 2),
      (centre-x + box-width / 2, centre-y + box-height / 2),
      name: name,
      radius: 0.1,
      fill: box-fill,
      stroke: box-stroke,
    )
    content(name + ".center", body)
  }
  let key-text(index) = [#text(weight: "bold", entries.at(index).at(3))#h(0.15em)#label[#index]]
  let added-stroke = geometry-stroke + palette.highlight

  // The slice: X to the right, Z up, a size-level-0 cell `unit` wide. Each active cell shows its input index and
  // its block; the split cell, drawn as its small cells, is outlined.
  let unit = 1.8
  let origin = (place-x(3) - unit / 2, 7.0)
  let active-cells = entries.filter(entry => not entry.at(4)).map(entry => (entry.at(0), entry.at(1), entry.at(2)))
  cell-grid(origin, active-cells, unit: unit)
  for (index, entry) in entries.enumerate() {
    let (x, z, size-level, block, is-added) = entry
    let (low-x, low-z, width) = cell-box((x, z, size-level))
    let centre = cell-grid-point(origin, unit, (low-x + width / 2, low-z + width / 2))
    if is-added {
      rect(
        cell-grid-point(origin, unit, (low-x, low-z)),
        cell-grid-point(origin, unit, (low-x + width, low-z + width)),
        stroke: added-stroke,
      )
      content(cell-grid-point(origin, unit, (low-x + width / 2, low-z + width)), anchor: "south", padding: 0.1, text(
        fill: palette.highlight,
        size: inset-label-size,
      )[split cell 6, added: block B])
    } else {
      content(centre, [#text(weight: "bold")[#index]#h(0.25em)#label[#block]])
    }
  }
  content((0, origin.at(1) + unit / 2), anchor: "west", align(left)[slice: input index \ and block])

  // The side between blocks A and B: block A ends and blocks B and C start at x = 0.
  let block-side-low = cell-grid-point(origin, unit, (0, -0.12))
  let block-side-high = cell-grid-point(origin, unit, (0, 1.12))
  line(block-side-low, block-side-high, stroke: (paint: palette.highlight, thickness: geometry-stroke, dash: "dashed"))
  content(block-side-low, anchor: "north", label[x = 0: A ends, B and C start])

  // Line 1: the block key of each entry in input order.
  content((0, input-y), anchor: "west", [input order])
  for index in range(entries.len()) {
    let is-added = entries.at(index).at(4)
    let box-stroke = if is-added { added-stroke } else { guide-stroke + black }
    entry-box("input-" + str(index), (place-x(index), input-y), key-text(index), box-stroke: box-stroke)
  }

  // Line 2: after the stable sort by block key.
  content((0, sorted-y), anchor: "west", [sorted by block])
  for (place, index) in sorted-indexes.enumerate() {
    let is-marked = marks.at(place) == 1
    let box-fill = if is-marked { fill-emphasis } else { fill-plain }
    entry-box("sorted-" + str(place), (place-x(place), sorted-y), key-text(index), box-fill: box-fill)
    coloured-arrow(
      "input-" + str(index) + ".south",
      "sorted-" + str(place) + ".north",
      stroke-width: guide-stroke,
      small: true,
    )
  }

  // Lines 3 and 4: the marks and their exclusive sums, with the total after the last sum.
  content((0, mark-y), anchor: "west", [mark])
  content((0, sum-y), anchor: "west", [exclusive sum])
  for (place, mark) in marks.enumerate() {
    let box-fill = if mark == 1 { fill-emphasis } else { fill-plain }
    entry-box("mark-" + str(place), (place-x(place), mark-y), [#mark], box-fill: box-fill)
    entry-box("sum-" + str(place), (place-x(place), sum-y), [#sums.at(place)], box-fill: box-fill)
  }
  let total-place = marks.len()
  entry-box("sum-total", (place-x(total-place), sum-y), [#sums.at(total-place)], box-fill: white)
  content((place-x(total-place), sum-y - box-height / 2 - 0.08), anchor: "north", label[total])

  // The block ids: each marked entry writes its block at the index its sum gives.
  content((0, block-y), anchor: "west", [block ids])
  for (block-index, name) in block-names.enumerate() {
    entry-box("block-" + str(block-index), (place-x(block-index), block-y), text(weight: "bold", name))
    content((place-x(block-index), block-y - box-height / 2 - 0.08), anchor: "north", label[index #block-index])
  }
  for (place, mark) in marks.enumerate() {
    if mark == 1 {
      coloured-arrow(
        "sum-" + str(place) + ".south",
        "block-" + str(sums.at(place)) + ".north",
        colour: palette.highlight,
        stroke-width: guide-stroke,
        small: true,
      )
    }
  }
})
