# Two-scale FVM solver for mixed lubrication

Finite volume solvers for the Reynolds equation on a tilted pad bearing, in
cartesian and in polar coordinates, together with the 1D analytical solution
used to verify them. Includes the load carrying capacity of a hydropower
thrust bearing.

## Getting the repo on your machine

```bash
git clone https://github.com/vidarlindqvist/twoscale-fvm-solver.git
cd twoscale-fvm-solver
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows the activation line is `.venv\Scripts\activate` instead.

Requires Python 3.10 or newer. `numpy`, `scipy` and `matplotlib` are pulled in
by the install.

Install with `-e` (editable). It puts `src/` on the import path so
`import reynolds` works from any directory, and your edits take effect without
reinstalling. A plain `pip install .` instead leaves a stale copy of the package
in a `build/` directory, which is a common source of confusion.

### Installing the development tools

Anything needed only for development is declared as an optional extra rather
than a hard dependency, so `pip install -e .` does not pull it in. Name the
extra in brackets to get it as well:

```bash
pip install -e ".[dev]"
```

That installs the package exactly as above, plus `pytest`. The name `dev` comes
from the `[project.optional-dependencies]` table in `pyproject.toml`, so adding
another tool there makes it available through the same command, and anyone who
only wants to run the solvers can keep ignoring it.

Keep the quotes around `".[dev]"`. Most shells try to expand bare brackets as a
glob pattern and the install fails with a confusing error.

## What to run

```bash
python examples/verify_cartesian.py
python examples/verify_polar.py
python examples/lcc.py
```

| script | what it does | output |
| --- | --- | --- |
| `verify_cartesian.py` | cartesian solver against the analytical solution, on a high aspect ratio pad where side leakage is negligible | aspect ratio, relative RMS error, `comparison.png` |
| `verify_polar.py` | polar solver against the cartesian one, on matched near-square domains | aspect ratio, relative RMS error, `verify_polar.png` |
| `lcc.py` | load carrying capacity of the bearing, and the shaft speed needed to carry 650 kN | dimensionless LCC, shaft speed |

## Tests

Needs the `dev` extra above. Run from the repo root:

```bash
pytest
```

`tests/test_physics.py` checks invariants that have to hold for either solver
and any geometry: a flat film generates no pressure, the boundary pressure is
exactly zero, and a converging wedge generates positive pressure.

## Using the solvers directly

The solvers themselves live in `src/reynolds/` and are importable on their own:

```python
from reynolds import analytical, cartesian, polar

pressure, X_vector, Y_vector, delta_H = cartesian.solve(l=1e-3, b=1e-3)
pressure, R_vector, theta_vector, theta2 = polar.solve(Ri=250e-3, Ro=500e-3)
```
