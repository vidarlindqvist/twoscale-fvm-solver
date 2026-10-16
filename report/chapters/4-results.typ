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
  // image("../figures/pressure.pdf", width: 60%),
  caption: [Dimensionless pressure $P$ on the homogenised pad.],
) <fig:pressure>
