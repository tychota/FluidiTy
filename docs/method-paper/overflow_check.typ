// Fails the compile when a figure, a table, a code block or a display equation leaves the text area.
//
// A block that is wider than the text area runs into a margin, and a figure taller than the text area runs off the
// page. Neither shows in the compile, so this file measures every such block and, at the end of the document, stops
// the compile with one line for each block that does not fit:
//
//   page 31: equation (4) is 381 pt wide, and the text area is 300 pt
//
// Use it once, at the top of the document, as `#show: overflow_check.install`.
//
// What is measured. A block is too wide when its natural width, measured with no width limit, is more than the text
// area, and a limit would not make it narrower. The second condition tells a block that wraps (a paragraph in a table
// cell, a caption) from one that cannot (a drawing, a table with fixed columns, an equation, a code line without a
// space): the height of a block that wraps grows when it is measured at the text width, and the height of one that
// cannot wrap does not. A figure is measured by its body, so that its caption, which always wraps, is not read as
// width. A block with a number, such as an equation, is measured with its number.
//
// A figure is too tall when it is taller than the text area, unless it may break across pages. Floats and figures in
// the text cannot break; only a procedure kept in the text (`breakable: true`) can.
//
// What is not seen. A line of running text, such as a long code word that cannot be broken, or an inline equation
// that is wider than the rest of its line, runs into the margin without a block to measure. A block that wraps in part
// and still has a part that does not fit (a table with long cells and also a fixed column that is too wide) is read
// as one that wraps. A block inside a narrower container (a box, a column) is compared with the text area of the page,
// not with its container. A table or a code block taller than a page may break across pages, so no height is checked
// for them.

#let tolerance = 0.5pt
// A block whose width is relative to the container, such as `width: 100%`, has no natural width: measured without a
// limit it is infinite, and it fills whatever it is given.
#let is-finite(length) = length < 100000pt

#let text-area() = {
  let width = page.width - page.margin.left.length - page.margin.right.length
  let height = page.height - page.margin.top.length - page.margin.bottom.length
  (width: width, height: height)
}

// Measures `body`, which is `block` itself or the part of it that must fit, and records a finding when it does not.
// `describe` gives, inside a context, how a finding names the block. `may-break` is whether the block may be taller
// than the text area, because it can continue on the next page.
#let check-block(describe, block, body, may-break) = context {
  let area = text-area()
  let natural = measure(body)
  let wrapped = measure(width: area.width, body)
  let whole = measure(width: area.width, block)
  let is-wider = is-finite(natural.width) and natural.width > area.width + tolerance
  let can-wrap = calc.abs(natural.height - wrapped.height) >= tolerance
  let is-taller = not may-break and whole.height > area.height + tolerance
  block
  if (is-wider and not can-wrap) or is-taller {
    let page-number = block.location().page()
    let problem = if is-wider and not can-wrap {
      str(calc.round(natural.width.pt())) + " pt wide, and the text area is " + str(calc.round(area.width.pt())) + " pt"
    } else {
      str(calc.round(whole.height.pt())) + " pt tall, and the text area is " + str(calc.round(area.height.pt())) + " pt"
    }
    [#metadata("page " + str(page-number) + ": " + describe() + " is " + problem) <block-overflow>]
  }
}

#let counted(name, counter-value) = if counter-value.len() > 0 { name + " " + str(counter-value.first()) } else { name }

#let install(body) = {
  show figure: it => {
    let has-text = type(it.supplement) == content and it.supplement.has("text")
    let supplement = if has-text { it.supplement.text } else { "figure" }
    let describe() = counted(lower(supplement), it.counter.at(it.location()))
    // A procedure kept in the text can continue on the next page; every other figure is one piece.
    let may-break = it.kind == "procedure" and it.placement == none
    check-block(describe, it, it.body, may-break)
  }
  show math.equation.where(block: true): it => {
    let describe() = if it.numbering == none {
      "equation"
    } else {
      "equation (" + str(counter(math.equation).at(it.location()).first()) + ")"
    }
    check-block(describe, it, it, true)
  }
  show table: it => check-block(() => "table", it, it, true)
  show raw.where(block: true): it => check-block(() => "code block", it, it, true)
  body
  context {
    let findings = query(<block-overflow>).map(found => found.value)
    assert(findings.len() == 0, message: "a block leaves the text area:\n" + findings.join("\n"))
  }
}
