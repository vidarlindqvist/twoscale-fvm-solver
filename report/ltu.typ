// LTU report style (TVM department), after the LaTeX template by
// M. Gustafsson and A. Almqvist. Usage: see main.typ.

#let report(
  title: [],
  author: "",
  email: "",
  department: "Department of Engineering Sciences and Mathematics",
  date: datetime.today(),
  cover: none,
  logo: image("figures/ltu-logo.jpg", width: 20%),
  abstract: [],
  notation: (),
  body,
) = {
  set document(title: title, author: author)
  set page(paper: "a4", margin: 2cm, numbering: none)
  set text(font: "New Computer Modern", size: 10pt, lang: "en")
  set par(justify: true, first-line-indent: 1.5em, spacing: 0.65em)
  show math.equation: set text(font: "New Computer Modern Math")
  show raw: set text(font: "DejaVu Sans Mono", size: 0.9em)

  // Headings: 1, 1.1, 1.1.1, every section on a new page
  set heading(numbering: "1.1")
  show heading: set block(above: 1.6em, below: 1em)
  show heading.where(level: 1): it => {
    pagebreak(weak: true)
    set text(size: 14.4pt)
    it
  }
  show heading.where(level: 2): set text(size: 12pt)

  // Equations (1), figures and tables with small captions, tables captioned above
  set math.equation(numbering: "(1)")
  show ref: it => {
    let el = it.element
    if el != none and el.func() == math.equation {
      link(el.location(), numbering(el.numbering, ..counter(math.equation).at(el.location())))
    } else { it }
  }
  show figure.caption: set text(size: 9pt)
  show figure.where(kind: table): set figure.caption(position: top)
  set table(stroke: none)

  show link: set text(fill: rgb("#003865"))
  set footnote.entry(separator: line(length: 30%, stroke: 0.5pt))

  // Title page
  page(align(center)[
    #v(2cm)
    #text(size: 17.28pt, title)
    #v(1.5em)
    #if cover != none { cover; v(2em) }
    #text(size: 12pt)[
      #author \
      #raw(email) \
      #department
    ]
    #v(2em)
    #logo
    #v(1em)
    #text(size: 12pt, date.display("[month repr:long] [day padding:none], [year]"))
  ])

  // Front matter, roman page numbers
  set page(numbering: "i")
  counter(page).update(1)
  heading(numbering: none, outlined: false)[Abstract]
  abstract
  outline(depth: 3)
  if notation.len() > 0 {
    heading(numbering: none, outlined: false)[Notation]
    grid(columns: 2, column-gutter: 2em, row-gutter: 0.8em, ..notation.flatten())
  }

  // Main matter, arabic page numbers
  pagebreak()
  set page(numbering: "1")
  counter(page).update(1)
  body
}
