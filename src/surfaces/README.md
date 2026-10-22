# surfaces

Random rough surfaces for the cell problems. Used by the GP work, never by
`reynolds` itself.

| file | contents |
| --- | --- |
| `fractal_surface.py` | Python port of the Almqvist–Pérez-Ràfols surface generator |
| `fractal_surface_standalone.m` | the MATLAB original it was ported from, for reference |

```python
from surfaces import fractal_surface

s = fractal_surface(dict(Nx=256, Ny=256, plot=False, outputPrefix=""), seed=1, H=0.5)
s.z     # heights, periodic, Sq = 1 and zero mean by default
```

## Origin and AI disclosure

The generator, its algorithm and the MATLAB code are by Andreas Almqvist
(mySurfGenApp, 2023, BSD-style licence reproduced at the end of
`fractal_surface.py`; Pérez-Ràfols & Almqvist, Tribology International 131
(2019) 591–604, doi:10.1016/j.triboint.2018.11.020).

**The translation from MATLAB to Python was carried out with an agentic AI
coding assistant**, directed and checked by Vidar Lindqvist. It is a
line-by-line port with the same options, defaults, algorithm and outputs. The
copy here was taken on 2026-10-22 from the standalone surface-generator project
and is unchanged apart from the disclosure note and the import path.

## Differences from MATLAB

- **Random numbers.** `rand` is bit-identical to MATLAB's `rng(seed,'twister')`.
  `randn` is not, because MATLAB uses a proprietary Ziggurat transform. For the
  same seed, Gaussian, Bi-Gaussian and Weibull surfaces are therefore a different
  realisation with the same statistics.
- The diagnostic figure uses matplotlib/viridis instead of MATLAB/parula.
- It never touches global random state.

## Verification

- `tests/test_fractal_surface.py` covers options, reproducibility, statistics
  and the export formats.
- The port was checked against the original `.m` file run in GNU Octave 10.3,
  with the same random draws replayed. Five configurations were compared,
  covering all distributions, both spectra, both pipelines,
  rectangular/anisotropic grids and the iteration cap. Every output field
  agreed to about 1e-14, with the same iteration counts and stop reasons.
- Octave lacks `histcounts`, so that function was supplied as a shim following
  MATLAB's bin-edge algorithm. **A comparison against a real MATLAB run is still
  outstanding.** The scripts for it are in the standalone project
  (`verify/make_matlab_reference.m`, `verify/compare_with_matlab.py`).
