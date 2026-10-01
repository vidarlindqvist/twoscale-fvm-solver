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

= Introduction
Background, previous work @patir1978 and the aim of the report.

= Theory <sec:theory>
The dimensionless Reynolds equation for the tilted pad reads
$
  (partial) / (partial X) (H^3 (partial P) / (partial X))
  + (partial) / (partial Y) (H^3 (partial P) / (partial Y))
  = (partial H) / (partial X),
$ <eq:reynolds>
with the pressure scaled by $6 mu u_s L \/ h_"min"^2$ @hamrock2004.

== Homogenisation
For a rough surface, $H^3$ in @eq:reynolds is replaced by the flow factors
$phi_x, phi_y$ and the right-hand side by $partial phi_s \/ partial X$.

== Flow factors
The flow factors are averages of the cell problem fluxes over the unit cell.

= Method <sec:method>
The equations are discretised with a five-point finite volume scheme
@versteeg2007.

== Discretisation
== Cell problems

= Results <sec:results>
@tab:load lists the load for shrinking roughness wavelengths and
@fig:pressure shows the pressure on the pad.

#figure(
  table(
    columns: 5,
    align: center,
    table.hline(),
    [$epsilon$], [1/4], [1/8], [1/16], [1/32],
    table.hline(stroke: 0.5pt),
    [$W$], [--], [--], [--], [--],
    table.hline(),
  ),
  caption: [Load carrying capacity of the resolved rough pad.],
) <tab:load>

#figure(
  rect(width: 60%, height: 4cm, stroke: 0.5pt + gray)[figure here],
  // image("figures/pressure.pdf", width: 60%),
  caption: [Dimensionless pressure $P$ on the homogenised pad.],
) <fig:pressure>

= Discussion and Conclusions
Relate the results to the aims in the introduction, discuss sources of error
and suggest further work.

#bibliography("refs.bib", title: "References", style: "ieee")
