import os

import numpy as np
import pytest
from scipy.io import loadmat

from surfaces.fractal_surface import (MatlabRNG, ReplayRNG, SurfaceError, defaults,
                                     fractal_surface, fractal_surface_standalone)

FAST = dict(Nx=128, Ny=128, plot=False)


def test_rand_matches_matlab_twister():
    # MATLAB: rng(1,'twister'); rand  and  rng(0,'twister'); rand
    assert MatlabRNG(1).rand(1, 1)[0, 0] == 0.417022004702574
    assert MatlabRNG(0).rand(1, 1)[0, 0] == 0.8147236863931789
    # Column-major fill, like MATLAB rand(2,2)
    a = MatlabRNG(1).rand(2, 2)
    b = MatlabRNG(1).rand(4, 1).ravel()
    assert a[1, 0] == b[1] and a[0, 1] == b[2]


def test_defaults_and_option_checking():
    assert fractal_surface('defaults') == defaults()
    assert fractal_surface_standalone is fractal_surface
    with pytest.raises(SurfaceError, match='Unknown option'):
        fractal_surface(dict(FAST, Hurst=0.5))
    with pytest.raises(SurfaceError, match='even'):
        fractal_surface(dict(FAST, Nx=127))
    with pytest.raises(SurfaceError, match='shortWavelength'):
        fractal_surface(dict(FAST, shortWavelength=0.6))
    out = fractal_surface(dict(FAST, distribution='weib', spectrum='exp'))
    assert (out.options.distribution, out.options.spectrum) == ('Weibull', 'Exponential')


def test_reproducible_and_seed_dependent():
    a = fractal_surface(FAST)
    b = fractal_surface(FAST)
    c = fractal_surface(FAST, seed=1)
    assert np.array_equal(a.z, b.z)
    assert not np.allclose(a.z, c.z)


def test_replay_of_own_draws_is_identical():
    rng = MatlabRNG(7)
    draws = [rng.randn(128, 128), rng.rand(128, 128)]
    a = fractal_surface(FAST, distribution='Weibull', seed=7)
    b = fractal_surface(FAST, distribution='Weibull', rng=ReplayRNG(draws))
    assert np.array_equal(a.z, b.z)


@pytest.mark.parametrize('distribution', ['Gaussian', 'Weibull', 'Bi-Gaussian'])
@pytest.mark.parametrize('pipeline', ['app', 'band-first'])
def test_output_shape_and_normalization(distribution, pipeline):
    out = fractal_surface(Nx=128, Ny=64, Lx=2, Ly=1, Sq=0.3, meanHeight=5, plot=False,
                          distribution=distribution, pipeline=pipeline)
    assert out.z.shape == (64, 128)
    assert out.dx == 2 / 128 and out.dy == 1 / 64
    assert np.allclose(np.diff(out.x), out.dx) and out.x[0] == 0
    assert out.diagnostics.mean == pytest.approx(5)
    assert out.diagnostics.Sq == pytest.approx(0.3)
    assert out.history.stopReason in ('maximum_iterations', 'histogram_stagnation', 'tolerance')


def test_band_first_preserves_height_distribution_exactly():
    out = fractal_surface(FAST, distribution='Weibull', pipeline='band-first')
    assert out.diagnostics.sortedHeightRMSError < 1e-12


def test_self_affine_spectrum_slope():
    out = fractal_surface(Nx=256, Ny=256, H=0.8, plot=False)
    assert out.diagnostics.Hfitted == pytest.approx(0.8, abs=0.1)
    assert out.diagnostics.skewness == pytest.approx(0, abs=0.1)
    assert out.diagnostics.kurtosis == pytest.approx(3, abs=0.2)


def test_skewed_distribution_survives():
    out = fractal_surface(FAST, distribution='Weibull', weibullShape=1)
    assert out.diagnostics.skewness < -1


def test_exports(tmp_path):
    prefix = str(tmp_path / 'sub' / 'surf')
    out = fractal_surface(FAST, Ny=64, Ly=0.5, outputPrefix=prefix, exportHeight=True)
    for suffix in ('.mat', '_xyz.txt', '_height.txt', '.png'):
        assert os.path.isfile(prefix + suffix)

    xyz = np.loadtxt(prefix + '_xyz.txt', comments='%')
    assert xyz.shape == (128 * 64, 3)
    assert np.array_equal(xyz[:, 2], out.z.ravel(order='F'))  # %.17g round-trips exactly
    assert xyz[1, 1] == out.y[1] and xyz[64, 0] == out.x[1]

    with open(prefix + '_height.txt') as f:
        header = [next(f) for _ in range(5)]
    assert header == ['% Almqvist_PerezRafols\n', '% 128\n', '% 64\n', f'% {out.dx:.17g}\n', '% 1\n']
    assert np.array_equal(np.loadtxt(prefix + '_height.txt', comments='%'), out.z.ravel(order='F'))

    saved = loadmat(prefix + '.mat', simplify_cells=True)['out']
    assert np.array_equal(saved['z'], out.z)
    assert saved['history']['stopReason'] == out.history.stopReason

    with pytest.raises(SurfaceError, match='Output exists'):
        fractal_surface(FAST, outputPrefix=prefix)
    fractal_surface(FAST, outputPrefix=prefix, overwrite=True)


def test_height_export_requires_square_pixels(tmp_path):
    with pytest.raises(SurfaceError, match='dx=dy'):
        fractal_surface(FAST, Ny=64, outputPrefix=str(tmp_path / 's'), exportHeight=True)
