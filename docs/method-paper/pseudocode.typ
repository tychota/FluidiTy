// The one pseudocode style of the method paper.
//
// A procedure is a list of numbered steps written in plain words and code names, never in paper symbols. Step N
// here is the block that starts with `// Step N:` in the code, so a reader can hold the paper beside the file.

#let procedure-font = ("Hermes Maia OT 3", "Libertinus Sans")
#let procedure-rule = 0.4pt + luma(140)

// Renders one procedure as a numbered figure of kind "procedure", referable through `label`.
//
// `title`: the function or operation, usually a code name. `label`: a label such as <proc-check-can-run-on>.
// Named arguments `inputs`, `outputs` and `fails` give the lines "Inputs", "Outputs" and "Fails when"; each is
// optional. `breakable: true` lets a procedure taller than a page continue on the next one. A named argument
// `closing` gives one unnumbered line after the last step, such as where the same procedure runs in another form.
// Every positional argument is one step, as a pair `([short name], [what the step does])`. Steps are numbered from
// 1 in the order given.
#let procedure(title, label, ..steps) = {
  let named = steps.named()
  let header-lines = (
    ("Inputs", named.at("inputs", default: none)),
    ("Outputs", named.at("outputs", default: none)),
    ("Fails when", named.at("fails", default: none)),
  ).filter(entry => entry.at(1) != none)

  let header = grid(
    columns: (auto, 1fr),
    column-gutter: 0.8em,
    row-gutter: 0.55em,
    ..header-lines
      .map(((name, body)) => (text(font: procedure-font, weight: "bold", name), body))
      .flatten(),
  )

  let step-cells = steps
    .pos()
    .enumerate()
    .map(((index, step)) => {
      let (short-name, body) = step
      (
        text(font: procedure-font, weight: "bold", number-type: "lining", str(index + 1)),
        [#text(font: procedure-font, weight: "bold", short-name). #body],
      )
    })
    .flatten()

  // A procedure floats to the top or bottom of a page, as the drawn figures do. Kept in place, a tall procedure
  // that does not fit moves to the next page whole and leaves the end of this one white. A float never breaks, so a
  // procedure taller than a page would run off it: `breakable: true` keeps such a procedure in the text instead,
  // where it continues on the next page. Its caption, header and first step then form one piece that never
  // breaks, so a page never ends with a header alone, and a float waiting for a page goes to the top of the next
  // page, before or after the procedure's parts, never between a header and its steps.
  let is-breakable = named.at("breakable", default: false)
  let kept-together-count = if is-breakable { calc.min(1, steps.pos().len()) } else { steps.pos().len() }
  let step-grid(cells) = grid(columns: (1.4em, 1fr), row-gutter: 0.65em, ..cells)

  let body = block(width: 100%, stroke: (top: procedure-rule, bottom: procedure-rule), inset: (y: 0.7em), {
    set align(left)
    set par(justify: false, first-line-indent: 0pt)
    block(breakable: not is-breakable, spacing: 0pt, {
      header
      v(0.2em)
      line(length: 100%, stroke: procedure-rule)
      v(0.2em)
      step-grid(step-cells.slice(0, 2 * kept-together-count))
    })
    if kept-together-count < steps.pos().len() {
      v(0.65em)
      block(spacing: 0pt, step-grid(step-cells.slice(2 * kept-together-count)))
    }
    let closing = named.at("closing", default: none)
    if closing != none {
      v(0.2em)
      line(length: 100%, stroke: procedure-rule)
      v(0.2em)
      closing
    }
  })

  if is-breakable {
    show figure.where(kind: "procedure"): set block(breakable: true)
    show figure.caption: set block(sticky: true)
    [#figure(body, kind: "procedure", supplement: [Procedure], caption: title)#label]
  } else {
    [#figure(body, kind: "procedure", supplement: [Procedure], caption: title, placement: auto)#label]
  }
}
