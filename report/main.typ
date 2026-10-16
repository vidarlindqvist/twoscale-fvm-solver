#import "ltu.typ": report

#show: report.with(
  title: [Two-Scale Finite Volume Solver \ for Mixed Lubrication],
  author: "Vidar Lindqvist",
  email: "author@student.ltu.se",
  // cover: image("figures/cover.png", width: 60%),
  abstract: [
    Problem, method, the most important results and what they mean, in
    150--200 words. No figures, references or formulas.
  ],
  notation: (
    ($h$, [Film thickness (m)]),
    ($H$, [Dimensionless film thickness, $h \/ h_"min"$]),
    ($P$, [Dimensionless pressure]),
    ($phi_x, phi_y$, [Pressure flow factors]),
    ($phi_s$, [Shear flow factor]),
    ($epsilon$, [Roughness wavelength over pad length]),
  ),
)

#include "chapters/1-introduction.typ"
#include "chapters/2-theory.typ"
#include "chapters/3-method.typ"
#include "chapters/4-results.typ"
#include "chapters/5-discussion.typ"

#bibliography("refs.bib", title: "References", style: "ieee")
