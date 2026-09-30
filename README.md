# Two-scale FVM solver for mixed lubrication

Finite volume solvers for the Reynolds equation on a tilted pad bearing, in
cartesian and polar coordinates, and a homogenised cartesian solver for rough
surfaces. 

## Install

```bash
git clone https://github.com/vidarlindqvist/twoscale-fvm-solver.git
cd twoscale-fvm-solver
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Needs Python 3.10 or newer. The `[dev]` part also installs pytest.

## Layout

| file | contents |
| --- | --- |
| `film.py` | film thickness of the pads, the roughness, and `faces()` |
| `assembly.py` | FVM coefficients, source term and the five point matrix |
| `boundary.py` | which nodes are fixed (pad edges, or one node in the cell) |
| `cartesian.py`, `polar.py` | the pad solvers |
| `homogenised.py` | cell problems, flow factors and the flow factor table |
| `load.py` | load carrying capacity |
| `analytical.py` | 1D analytical solution |

Everything is dimensionless: `H = h / h_min`, pad coordinates are divided by the
pad length (polar: by `Ro`) and cell coordinates by the roughness wavelength.
The pressure scale is `6 mu u_s length / h_min**2` for the cartesian pad and
`6 mu omega Ro**2 / h_min**2` for the polar one.

## Homogenisation

The homogenised equation has the same form as the smooth one, with
`H**3` replaced by the pressure flow factors `phi_x`, `phi_y` and `H` by the
shear flow factor `phi_s`. They are computed like this:

1. `homogenised.solve_cell` solves the cell problems on the unit cell
   with periodic boundaries, for a local film `H = alpha + H_r(y1, y2)`.
2. `homogenised.flow_factors` averages the fluxes into `phi_x`, `phi_y`,
   `phi_s` and the cross terms.
3. `homogenised.table` does this for a range of separations `alpha` and returns
   cubic spline interpolants of `phi_x`, `phi_y` and `phi_s`.
4. `cartesian.solve` takes these in place of its smooth defaults.

```python
import numpy as np
from reynolds import cartesian, film, homogenised, load

phi_x, phi_y, phi_s = homogenised.table(film.bisinusoidal(0.2), alphas=np.linspace(0.9, 1.4, 26))
P_0, X_vector, Y_vector, H_taper = cartesian.solve(phi_x=phi_x, phi_y=phi_y, phi_s=phi_s)
print(load.cartesian(P_0, X_vector, Y_vector))
```

The table has to cover the film thickness on the whole pad, otherwise the
solver stops with an error. The cross terms are ignored by the five point
stencil. They are zero for roughness that is symmetric in both directions, like
the bi-sinusoidal one, and `table` warns if they are not. The roughness is
assumed to sit on the stationary surface, otherwise the cell problems become
time dependent. 

## Examples

```bash
python examples/verify_cartesian.py      # against the 1D analytical solution
python examples/verify_polar.py          # against the cartesian solver
python examples/lcc.py                   # load carrying capacity and shaft speed
python examples/verify_cell.py           # chi1 on surface 
python examples/verify_homogenised.py    # resolved rough pad -> homogenised pad
```

`verify_homogenised.py` solves the rough pad with every bump resolved for
shrinking wavelengths and compares the load with the homogenised one:

| epsilon | 1/4 | 1/8 | 1/16 | 1/32 |
| --- | --- | --- | --- | --- |
| error in load | TBD | TBD | TBD | TBD |

## Tests

```bash
pytest
```
