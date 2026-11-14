// Figure <fig-stable-order>: one counting pass of BuildStableOrder over eight items. Each item has one digit of its
// key; the pass counts the items of each digit, turns the counts into first places, and places the items front to
// back. The three items with digit 2 are drawn in the emphasis style: they leave the pass in the order they came in.

#import "../figure_style.typ": *

#let stable-order-figure() = figure-canvas({
  import cetz.draw: *

  // The digit of each item of the sequence, in sequence order. Items 0, 3 and 6 share digit 2.
  let digits = (2, 0, 3, 2, 0, 1, 2, 0)
  // The digit values that occur, with their counts and first places: the exclusive prefix sum of the counts.
  let digit-values = (0, 1, 2, 3)
  let counts = digit-values.map(value => digits.filter(digit => digit == value).len())
  let first-places = range(digit-values.len()).map(index => counts.slice(0, index).sum(default: 0))
  // The tied digit drawn in the emphasis style.
  let tied-digit = 2

  let first-x = 2.9
  let spacing = 1.1
  let box-width = 0.8
  let box-height = 0.55
  let before-y = 3.5
  let after-y = 1.0
  let place-x(place) = first-x + spacing * place

  // Row names on the left.
  content((0, before-y + 0.72), anchor: "west", [digit])
  content((0, before-y), anchor: "west", [sequence])
  content((0, after-y), anchor: "west", [next sequence])

  // The sequence before the pass: item index in the box, its digit above.
  for (item, digit) in digits.enumerate() {
    let box-fill = if digit == tied-digit { fill-emphasis } else { fill-plain }
    labelled-box("before-" + str(item), (place-x(item), before-y), box-width, box-height, box-fill: box-fill)[
      #item
    ]
    content((place-x(item), before-y + 0.72), text(weight: "bold")[#digit])
  }

  // Step 4 of the procedure: walk the sequence front to back; an item goes to the first place of its digit plus
  // the number of earlier items with the same digit.
  let next-free = first-places
  for (item, digit) in digits.enumerate() {
    let place = next-free.at(digit)
    next-free.at(digit) = place + 1
    let is-tied = digit == tied-digit
    let box-fill = if is-tied { fill-emphasis } else { fill-plain }
    labelled-box("after-" + str(place), (place-x(place), after-y), box-width, box-height, box-fill: box-fill)[
      #item
    ]
    let colour = if is-tied { palette.highlight } else { black }
    coloured-arrow(
      "before-" + str(item) + ".south",
      "after-" + str(place) + ".north",
      colour: colour,
      stroke-width: guide-stroke,
      small: true,
    )
  }

  // Under the next sequence: one bracket per digit value over the places it fills, with its count and first place.
  let table-y = after-y - 0.95
  let line-gap = 0.42
  content((0, table-y), anchor: "west", [digit])
  content((0, table-y - line-gap), anchor: "west", [count])
  content((0, table-y - 2 * line-gap), anchor: "west", [first place])
  for (index, value) in digit-values.enumerate() {
    let first = first-places.at(index)
    let last = first + counts.at(index) - 1
    let left-x = place-x(first) - box-width / 2
    let right-x = place-x(last) + box-width / 2
    let bracket-y = after-y - box-height / 2 - 0.2
    line(
      (left-x, bracket-y + 0.1),
      (left-x, bracket-y),
      (right-x, bracket-y),
      (right-x, bracket-y + 0.1),
      stroke: guide-stroke + black,
    )
    let centre-x = (place-x(first) + place-x(last)) / 2
    content((centre-x, table-y), text(weight: "bold")[#value])
    content((centre-x, table-y - line-gap))[#counts.at(index)]
    content((centre-x, table-y - 2 * line-gap))[#first]
  }
})
