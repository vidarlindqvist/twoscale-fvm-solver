"""Almqvist--Perez-Rafols random rough surface generator (Python port).

Line-by-line port of ``fractal_surface_standalone.m`` (standalone refactoring
2026-09-27 of the mySurfGenApp source by Andreas Almqvist). Same options, same
defaults, same algorithm, same outputs and export formats.

AI disclosure: the translation from MATLAB to Python was carried out with an
agentic AI coding assistant, directed and checked by Vidar Lindqvist. The
algorithm and the method are Andreas Almqvist's. The port was verified against
the original as described under "Verification" in README.md next to this file.

Copied into twoscale-fvm-solver on 2026-10-22 from the standalone
surface-generator project, unchanged apart from this note and the import path.

    from surfaces.fractal_surface import defaults, fractal_surface

    cfg = defaults()
    cfg.Nx = 512
    cfg.Ny = 512
    cfg.H = 0.5
    cfg.distribution = 'Gaussian'
    cfg.outputPrefix = os.path.join(os.getcwd(), 'my_surface')

    surface = fractal_surface(cfg)      # surface.z, surface.x, surface.y, ...

Differences from the MATLAB version
-----------------------------------
* Random numbers: uniform draws (``rand``) are bit-identical to MATLAB's
  ``rng(seed,'twister')`` stream (same Mersenne Twister, same seeding, same
  column-major fill). Normal draws (``randn``) are NOT: MATLAB converts the
  stream with a proprietary Ziggurat transform. Gaussian and Bi-Gaussian
  surfaces therefore differ realization-wise from MATLAB for the same seed;
  they have the same statistics. Pass ``rng=ReplayRNG([...])`` to feed exact
  MATLAB draws (see ``verify/``); the algorithm is then deterministic and
  matches MATLAB to floating-point round-off.
* The diagnostic figure uses matplotlib with the viridis colour map (MATLAB:
  parula). Layout, contents and 160 dpi PNG export are the same.
* Only the 2nd-order finite differences of mydiffx/mydiffy are ported; the
  1st/4th-order branches are never called by the generator.
* Seeding never touches global random state (the MATLAB code saved and
  restored the caller's RNG to achieve the same thing).

Option documentation, spectral conventions and provenance: see the header of
``fractal_surface_standalone.m`` (kept next to this file); everything there
applies unchanged.

Provenance: local mySurfGenApp source, copyright Andreas Almqvist 2023;
https://www.mathworks.com/matlabcentral/fileexchange/129469
Perez-Rafols & Almqvist, Tribology International 131 (2019) 591--604,
doi:10.1016/j.triboint.2018.11.020. BSD-style license reproduced at the end of
this file.
"""

from __future__ import annotations

import ast
import math
import os
import sys

import numpy as np
import scipy.fft

__all__ = [
    'MatlabRNG',
    'ReplayRNG',
    'Struct',
    'defaults',
    'fractal_surface',
    'fractal_surface_standalone',
]

EPS = np.finfo(float).eps
REALMIN = np.finfo(float).tiny
REALMAX = np.finfo(float).max


class Struct(dict):
    """dict with attribute access, the Python stand-in for a MATLAB struct."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name) from None


class SurfaceError(ValueError):
    """Raised where the MATLAB code raises an error; ``id`` is the MATLAB identifier."""

    def __init__(self, ident, message):
        super().__init__(f'{message} ({ident})')
        self.id = ident


def _require(condition, ident, message):
    if not condition:
        raise SurfaceError(ident, message)


# --------------------------------------------------------------------------
# Random numbers
# --------------------------------------------------------------------------

class MatlabRNG:
    """Mersenne Twister stream seeded like MATLAB ``rng(seed,'twister')``.

    ``rand(M, N)`` returns exactly MATLAB's ``rand(M,N)`` (MATLAB maps seed 0
    to the reference seed 5489). ``randn`` uses numpy's normal transform, which
    differs from MATLAB's Ziggurat, so it is statistically but not bitwise
    equivalent.
    """

    def __init__(self, seed):
        self._stream = np.random.RandomState(5489 if seed == 0 else int(seed))

    def rand(self, m, n):
        return self._stream.random_sample(m * n).reshape((m, n), order='F')

    def randn(self, m, n):
        return self._stream.standard_normal(m * n).reshape((m, n), order='F')


class ReplayRNG:
    """Replays prerecorded arrays (e.g. MATLAB's own draws) in call order."""

    def __init__(self, draws):
        self._draws = [np.asarray(d, dtype=float) for d in draws]

    def _next(self, m, n):
        _require(self._draws, 'Surface:Replay', 'ReplayRNG ran out of draws.')
        d = self._draws.pop(0)
        _require(d.shape == (m, n), 'Surface:Replay',
                 f'Replay draw has shape {d.shape}, expected {(m, n)}.')
        return d

    rand = randn = _next


# --------------------------------------------------------------------------
# MATLAB numerical helpers (kept faithful so outputs match to round-off)
# --------------------------------------------------------------------------

def _fft2(a):
    return scipy.fft.fft2(a, workers=-1)


def _ifft2(a):
    return scipy.fft.ifft2(a, workers=-1)


def _linspace(d1, d2, n):
    """MATLAB linspace (its rounding differs slightly from numpy's)."""
    n = int(math.floor(n))
    if n < 1:
        return np.empty(0)
    if n == 1:
        return np.array([float(d2)])
    n1 = n - 1
    y = d1 + np.arange(n) * (d2 - d1) / n1
    y[-1] = d2
    return y


def _binpicker(xmin, xmax, nbins, raw_width):
    """matlab.internal.math.binpicker: histcounts' automatic bin edges."""
    xscale = max(abs(xmin), abs(xmax))
    xrange = xmax - xmin
    raw_width = max(raw_width, np.spacing(xscale))
    if xrange > max(math.sqrt(np.spacing(xscale)), REALMIN):
        pow10 = 10.0 ** math.floor(math.log10(raw_width))
        rel = raw_width / pow10
        if nbins is None:
            if rel < 1.5:
                width = 1 * pow10
            elif rel < 2.5:
                width = 2 * pow10
            elif rel < 4:
                width = 3 * pow10
            elif rel < 7.5:
                width = 5 * pow10
            else:
                width = 10 * pow10
            left = max(min(width * math.floor(xmin / width), xmin), -REALMAX)
            nbins_actual = max(1, math.ceil((xmax - left) / width))
            right = min(max(left + nbins_actual * width, xmax), REALMAX)
        else:
            width = pow10 * math.floor(rel)
            left = max(min(width * math.floor(xmin / width), xmin), -REALMAX)
            if nbins > 1:
                lower = (xmax - left) / nbins
                upper = (xmax - left) / (nbins - 1)
                p10 = 10.0 ** math.floor(math.log10(upper - lower))
                width = p10 * math.ceil(lower / p10)
            nbins_actual = nbins
            right = min(max(left + nbins_actual * width, xmax), REALMAX)
    else:
        if nbins is None:
            nbins = 1
        bin_range = max(1, math.ceil(nbins * np.spacing(xscale)))
        left = math.floor(2 * (xmin - bin_range / 4)) / 2
        right = math.ceil(2 * (xmax + bin_range / 4)) / 2
        width = (right - left) / nbins
        nbins_actual = nbins
    if not math.isfinite(width):
        return _linspace(left, right, nbins_actual + 1)
    return np.concatenate(([left], left + np.arange(1, nbins_actual) * width, [right]))


def _hist_edges(x, nbins=None):
    """Edges of histcounts(x) ('scott' when nbins is None, else nbins bins)."""
    x = np.ravel(x)
    finite = x[np.isfinite(x)]
    xmin, xmax = float(finite.min()), float(finite.max())
    if nbins is None:
        raw = 3.5 * np.std(x, ddof=1) / (x.size ** (1 / 3))
    else:
        raw = (xmax - xmin) / nbins
    return _binpicker(xmin, xmax, nbins, raw)


def _hist_pdf(x, edges):
    """histcounts(x, edges, 'Normalization','pdf'): count / (numel(x) * width)."""
    counts, _ = np.histogram(np.ravel(x), bins=edges)
    return counts / (np.size(x) * np.diff(edges))


def _interp1(x, v, xq):
    """interp1 linear: accepts decreasing x and returns NaN outside the range."""
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    if x[0] > x[-1]:
        x, v = x[::-1], v[::-1]
    return np.interp(xq, x, v, left=np.nan, right=np.nan)


def _trapz(x, y):
    return float(np.trapezoid(y, x)) if np.size(x) > 1 else 0.0


def _mode_indices(n):
    """[0:n/2-1, -n/2:-1], the FFT ordering of integer wave numbers."""
    return np.concatenate((np.arange(n // 2), np.arange(-(n // 2), 0))).astype(float)


def _validatestring(value, valid, name):
    """validatestring: case-insensitive, unambiguous prefixes accepted."""
    _require(isinstance(value, str), 'Surface:Option', f'{name} must be text.')
    low = value.lower()
    for v in valid:
        if v.lower() == low:
            return v
    matches = [v for v in valid if v.lower().startswith(low)] if low else []
    _require(len(matches) == 1, 'Surface:Option',
             f'{name} must match one of: {", ".join(valid)}; got {value!r}.')
    return matches[0]


# --------------------------------------------------------------------------
# 1. Configuration and validation
# --------------------------------------------------------------------------

def defaults():
    """Default options, the equivalent of ``fractal_surface_standalone('defaults')``."""
    return Struct(
        Nx=512, Ny=512, distribution='Gaussian', spectrum='Self-affine',
        H=.5, beta=.01, anisotropy=1, weibullShape=1, biGaussianCoeff=4,
        shortWavelength=1 / 64, longWavelength=.5, pipeline='app',
        Lx=1, Ly=1, Sq=1, meanHeight=0, seed=20260927,
        maxIterations=999, minSpeed=1e-3, tolerance=1e-3,
        plot=True, visible='on', outputPrefix='', exportMAT=True,
        exportXYZ=True, exportHeight=False, exportPNG=True, overwrite=False)


def _is_real_scalar(v):
    return (isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
            and math.isfinite(v))


def _validate(c):
    def scalar(name, cond=lambda v: True, what='a finite real scalar'):
        v = c[name]
        _require(_is_real_scalar(v) and cond(v), 'Surface:Option', f'{name} must be {what}.')

    for f in ('Nx', 'Ny'):
        scalar(f, lambda v: v == int(v) and v >= 8, 'an integer >= 8')
        _require(c[f] % 2 == 0, 'Surface:Grid', 'Nx and Ny must be even.')
        c[f] = int(c[f])
    for f in ('beta', 'anisotropy', 'weibullShape', 'Lx', 'Ly', 'Sq', 'longWavelength', 'tolerance'):
        scalar(f, lambda v: v > 0, 'positive')
    scalar('H', lambda v: 0 < v < 2, 'in the open interval (0, 2)')
    for f in ('shortWavelength', 'biGaussianCoeff', 'minSpeed'):
        scalar(f, lambda v: v >= 0, 'nonnegative')
    scalar('meanHeight')
    scalar('seed', lambda v: v == int(v) and 0 <= v <= 2**32 - 1, 'an integer in [0, 2^32-1]')
    scalar('maxIterations', lambda v: v == int(v) and v > 0, 'a positive integer')
    c.seed = int(c.seed)
    c.maxIterations = int(c.maxIterations)
    _require(c.longWavelength <= 1 and c.shortWavelength < c.longWavelength, 'Surface:Band',
             'Require 0 <= shortWavelength < longWavelength <= 1.')
    c.distribution = _validatestring(c.distribution, ('Gaussian', 'Weibull', 'Bi-Gaussian'), 'distribution')
    c.spectrum = _validatestring(c.spectrum, ('Self-affine', 'Exponential'), 'spectrum')
    c.pipeline = _validatestring(c.pipeline, ('app', 'band-first'), 'pipeline')
    c.visible = _validatestring(c.visible, ('on', 'off'), 'visible')
    for f in ('plot', 'exportMAT', 'exportXYZ', 'exportHeight', 'exportPNG', 'overwrite'):
        _require(isinstance(c[f], (bool, np.bool_, int, np.integer, float)) and c[f] in (0, 1),
                 'Surface:Option', f'{f} must be true/false.')
        c[f] = bool(c[f])
    _require(isinstance(c.outputPrefix, (str, os.PathLike)), 'Surface:Prefix', 'outputPrefix must be text.')
    c.outputPrefix = os.fspath(c.outputPrefix)
    if c.exportHeight:
        dx, dy = c.Lx / c.Nx, c.Ly / c.Ny
        _require(abs(dx - dy) <= 1e-12 * max(dx, dy), 'Surface:SquarePixels',
                 'Five-line height format requires dx=dy; use XYZ or MAT otherwise.')
    # Test band before calling the original helper (which normalizes its output).
    kx, ky = np.meshgrid(_mode_indices(c.Nx), _mode_indices(c.Ny))
    k = np.hypot(c.anisotropy * kx, ky / c.anisotropy)
    mask = k >= 1 / c.longWavelength
    if c.pipeline == 'band-first' and c.shortWavelength > 0:
        mask &= k <= 1 / c.shortWavelength
    _require(mask.any(), 'Surface:EmptySpectrum', 'No supported modes in requested band.')
    if c.pipeline == 'app' and c.shortWavelength > 0:
        _require((mask & (np.hypot(kx, ky) <= 1 / c.shortWavelength)).any(),
                 'Surface:EmptySpectrum', 'Post-filter removes every target mode.')
    return c


def _output_paths(c):
    paths = []
    if c.exportMAT:
        paths.append(c.outputPrefix + '.mat')
    if c.exportXYZ:
        paths.append(c.outputPrefix + '_xyz.txt')
    if c.exportHeight:
        paths.append(c.outputPrefix + '_height.txt')
    if c.exportPNG:
        paths.append(c.outputPrefix + '.png')
    return paths


def _check_outputs(c):
    if not c.outputPrefix or c.overwrite:
        return
    for p in _output_paths(c):
        _require(not os.path.isfile(p), 'Surface:Exists', f'Output exists: {p}')


def _normalize_height(z, sq, mu):
    z = np.real(z)
    z = z - z.mean()
    s = math.sqrt(np.mean(z**2))
    _require(math.isfinite(s) and s > 0, 'Surface:ZeroVariance',
             'Surface has no finite nonzero variance.')
    return z / s * sq + mu


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def fractal_surface(options=None, *, rng=None, **overrides):
    """Generate a surface; the equivalent of ``fractal_surface_standalone(cfg)``.

    ``options`` is a dict/Struct of options (unknown names are rejected), or the
    string ``'defaults'`` to return the default options. Keyword arguments are
    applied on top of ``options``. ``rng`` overrides the random source (an
    object with MATLAB-style ``rand(M, N)`` / ``randn(M, N)``); by default it
    is ``MatlabRNG(cfg.seed)``.

    Returns a Struct with z, x, y, dx, dy, zHeight, zSpectrum, options,
    history, diagnostics, parameters, heightPDF, bearingCurve, provenance.
    """
    cfg = defaults()
    if isinstance(options, str):
        _require(options.lower() == 'defaults', 'Surface:Command',
                 "Only the command 'defaults' is supported.")
        return cfg
    given = dict(options or {})
    given.update(overrides)
    for name, value in given.items():
        _require(name in cfg, 'Surface:UnknownOption', f'Unknown option: {name}')
        cfg[name] = value
    cfg = _validate(cfg)
    _check_outputs(cfg)
    if rng is None:
        rng = MatlabRNG(cfg.seed)

    # 2. Original app spectral construction
    N, M = cfg.Nx, cfg.Ny
    parameter = cfg.beta if cfg.spectrum == 'Exponential' else cfg.H
    short = cfg.shortWavelength if cfg.pipeline == 'band-first' else 0
    ZG, C = _fractal_surf_hu_tonder_anisotropic(
        parameter, cfg.longWavelength, short, 1, 1, N / M, M, N, cfg.anisotropy, cfg.spectrum, rng)
    PS = np.abs(_fft2(ZG))
    PS[C == 0] = 0
    _require((PS > 0).any(), 'Surface:EmptySpectrum', 'Selected band contains no nonzero Fourier modes.')
    _require(np.isfinite(PS).all(), 'Surface:Spectrum', 'Nonfinite spectrum: check cutoffs and parameters.')

    # 3. Sample the requested height distribution
    if cfg.distribution == 'Gaussian':
        target = rng.randn(M, N)
    elif cfg.distribution == 'Weibull':
        b = cfg.weibullShape
        # Shape is sufficient: later normalization sets scale and mean.
        target = -((-np.log(np.maximum(rng.rand(M, N), REALMIN))) ** (1 / b))
    else:  # Bi-Gaussian
        c = cfg.biGaussianCoeff
        first = c * rng.randn(M, N) - c
        target = np.minimum(first, rng.randn(M, N))
    target = target - target.mean()
    _require(np.isfinite(target).all() and np.std(target, ddof=1) > 0, 'Surface:Distribution',
             'Degenerate or nonfinite height sample.')
    # Preserve original app sample scaling within iteration; normalize at output.

    # 4. Alternating spectrum and rank projections
    zh, zs, history = _fractal_surf_generator(PS, target, cfg.maxIterations + 1, cfg.minSpeed, cfg.tolerance)
    z = _fftbandfilter(zh, 1, cfg.shortWavelength, 1) if cfg.pipeline == 'app' else zh
    z = _normalize_height(z, cfg.Sq, cfg.meanHeight)
    zh = _normalize_height(zh, cfg.Sq, cfg.meanHeight)
    zs = _normalize_height(zs, cfg.Sq, cfg.meanHeight)

    # 5. Coordinates and independent diagnostics
    x = np.arange(N) * cfg.Lx / N
    y = np.arange(M) * cfg.Ly / M
    X, Y = np.meshgrid(x, y)
    out = Struct(z=z, x=x, y=y, dx=cfg.Lx / N, dy=cfg.Ly / M,
                 zHeight=zh, zSpectrum=zs, options=cfg, history=history)
    out.diagnostics = _diagnostics(z, PS, C, target, cfg)
    out.parameters = _my_surf_param(X, Y, z)  # original app's descriptive parameters
    edges = _hist_edges(z, 80)
    out.heightPDF = Struct(height=(edges[:-1] + edges[1:]) / 2, density=_hist_pdf(z, edges))
    q = _linspace(0, 1, min(z.size, 1001))
    v = np.sort(z.ravel(order='F'))[::-1]
    out.bearingCurve = Struct(materialRatio=q, height=_interp1(_linspace(0, 1, v.size), v, q))
    out.provenance = 'Almqvist-Perez-Rafols local mySurfGenApp; standalone refactoring 2026-09-27'

    # 6. Optional plots and exports (no files by default)
    fig = None
    if cfg.plot or (cfg.outputPrefix and cfg.exportPNG):
        fig = _plot_surface(out, cfg.visible if cfg.plot else 'off')
    if cfg.outputPrefix:
        _export(out, X, Y, fig)
    if cfg.plot and cfg.visible == 'on':
        import matplotlib.pyplot as plt
        plt.show()
    return out


fractal_surface_standalone = fractal_surface


def _export(out, X, Y, fig):
    cfg = out.options
    prefix = cfg.outputPrefix
    parent = os.path.dirname(prefix)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if cfg.exportMAT:
        from scipy.io import savemat
        savemat(prefix + '.mat', {'out': _to_mat(out)}, do_compression=True)
    if cfg.exportXYZ:
        with open(prefix + '_xyz.txt', 'w') as fid:
            fid.write('% x y z; periodic samples, no duplicated endpoint\n')
            xyz = np.column_stack((X.ravel(order='F'), Y.ravel(order='F'), out.z.ravel(order='F')))
            np.savetxt(fid, xyz, fmt='%.17g')
    if cfg.exportHeight:
        with open(prefix + '_height.txt', 'w') as fid:
            fid.write('%% Almqvist_PerezRafols\n%% %d\n%% %d\n%% %.17g\n%% 1\n' % (cfg.Nx, cfg.Ny, out.dx))
            np.savetxt(fid, out.z.ravel(order='F'), fmt='%.17g')
    if cfg.exportPNG:
        fig.savefig(prefix + '.png', dpi=160, bbox_inches='tight')


def _to_mat(value):
    """Convert to types scipy.io.savemat writes as the matching MATLAB types."""
    if isinstance(value, dict):
        return {k: _to_mat(v) for k, v in value.items()}
    if isinstance(value, (bool, np.bool_)):
        return np.array(value, dtype=bool)
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    return value


# --------------------------------------------------------------------------
# Diagnostics and plotting
# --------------------------------------------------------------------------

def _diagnostics(z, PS, C, target, c):
    u = (z - z.mean()) / c.Sq
    F = _fft2(u)
    A = PS / math.sqrt(np.sum(PS**2)) * PS.size
    zc = z - z.mean()
    d = Struct(mean=z.mean(), Sq=math.sqrt(np.mean(zc**2)),
               skewness=np.mean(u**3), kurtosis=np.mean(u**4), min=z.min(), max=z.max())
    pos = A > 0
    d.spectrumRelativeAmplitudeError = math.sqrt(np.mean((np.abs(F[pos]) / A[pos] - 1) ** 2))
    power_all = np.abs(F) ** 2
    d.outsideIterationBandVariance = power_all[C == 0].sum() / power_all.sum()
    t = _normalize_height(target, 1, 0)
    d.sortedHeightRMSError = math.sqrt(np.mean((np.sort(u.ravel()) - np.sort(t.ravel())) ** 2))
    kx, ky = np.meshgrid(_mode_indices(c.Nx), _mode_indices(c.Ny))
    k = np.hypot(kx, ky)
    P = np.abs(F / F.size) ** 2
    bins = 10.0 ** _linspace(0, math.log10(k.max() + EPS), 41)
    cent = np.full(40, np.nan)
    power = np.full(40, np.nan)
    for j in range(40):
        m = (k >= bins[j]) & (k < bins[j + 1])
        if m.any():
            cent[j] = math.exp(np.mean(np.log(k[m])))
            power[j] = np.mean(P[m])
    d.radialMode = cent
    d.radialMeanPower = power
    d.Hfitted = np.nan
    m = _fit_mask(cent, power, c)
    if c.spectrum == 'Self-affine' and c.anisotropy == 1 and np.count_nonzero(m) >= 4:
        fit = np.polyfit(np.log(cent[m]), np.log(power[m]), 1)
        d.Hfitted = -fit[0] / 2 - 1
    d.fitMeaning = 'H fit is reported only for isotropic self-affine targets; fit is a realization diagnostic.'
    return d


def _fit_mask(cent, power, c):
    hi = min(c.Nx, c.Ny) / 4
    if c.shortWavelength > 0:
        hi = min(hi, .75 / c.shortWavelength)
    with np.errstate(invalid='ignore'):
        return (cent >= 2 / c.longWavelength) & (cent <= hi) & (power > 0) & np.isfinite(power)


def _plot_surface(s, visible):
    """Three-panel verification layout; normalization affects display only."""
    if visible == 'on':
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(15, 4.8))
    else:
        from matplotlib.figure import Figure
        fig = Figure(figsize=(15, 4.8))
    from matplotlib.ticker import NullFormatter
    c, d = s.options, s.diagnostics
    u = (s.z - d.mean) / d.Sq
    fig.patch.set_facecolor('w')
    ax1, ax2, ax3 = fig.subplots(1, 3, gridspec_kw={'wspace': 0.3})

    x, y = s.x / c.Lx, s.y / c.Ly
    extent = (x[0], x[-1] + (x[1] - x[0]), y[0], y[-1] + (y[1] - y[0]))
    im = ax1.imshow(u, origin='lower', extent=extent, cmap='viridis', interpolation='nearest')
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_xlabel('$x/L_x$')
    ax1.set_ylabel('$y/L_y$')
    if c.spectrum == 'Self-affine':
        ax1.set_title(f'{c.distribution} surface, H = {c.H:.3g}')
    else:
        ax1.set_title(f'{c.distribution} surface, Exponential spectrum')

    edges = _hist_edges(u, 80)
    ax2.hist(u.ravel(), bins=edges, density=True, color=(.3, .65, .8), edgecolor='k', linewidth=0.3)
    if c.distribution == 'Gaussian':
        t = np.linspace(min(-5, u.min()), max(5, u.max()), 400)
        ax2.plot(t, np.exp(-t**2 / 2) / math.sqrt(2 * math.pi), 'k--', linewidth=1.5)
    ax2.set_xlabel('(z - mean(z)) / $S_q$')
    ax2.set_ylabel('Probability density')
    ax2.set_title(f'Skew {d.skewness:.3f}; kurtosis {d.kurtosis:.3f}')

    k, power = d.radialMode, d.radialMeanPower
    limit = min(c.Nx, c.Ny) / 2
    if c.shortWavelength > 0:
        limit = min(limit, max(c.anisotropy, 1 / c.anisotropy) / c.shortWavelength)
    lower = min(c.anisotropy, 1 / c.anisotropy) / c.longWavelength
    with np.errstate(invalid='ignore'):
        show = np.isfinite(k) & (power > 0) & (k >= lower) & (k <= limit)
    ax3.loglog(k[show], power[show], 'o-', linewidth=1, markerfacecolor='none')
    if np.isfinite(d.Hfitted):
        m = _fit_mask(k, power, c)
        fit = np.polyfit(np.log(k[m]), np.log(power[m]), 1)
        ax3.loglog(k[m], np.exp(np.polyval(fit, np.log(k[m]))), 'k--', linewidth=1.3)
        ax3.set_title(f'Measured H = {d.Hfitted:.4f}')
    else:
        ax3.set_title('Radial spectrum (H fit not applicable)')
    ax3.xaxis.set_minor_formatter(NullFormatter())
    ax3.grid(True, which='both', alpha=0.3)
    ax3.set_xlabel('Radial Fourier mode')
    ax3.set_ylabel('Mean power per mode')
    return fig


# --------------------------------------------------------------------------
# Embedded original algorithms and original descriptive parameters
# --------------------------------------------------------------------------

def _fractal_surf_generator(PS, Zh, max_it, min_speed, min_err):
    M, N = PS.shape
    # Scale and center the power spectrum appropiatelly
    PS = PS.copy()
    PS[0, 0] = 0
    PS = PS / math.sqrt(np.sum(PS**2)) * M * N
    # Scale and center the height distribution
    Zh_mean = Zh.mean()
    Zh = Zh - Zh_mean
    Sq = math.sqrt(np.mean(Zh**2))
    Zh = Zh / Sq

    Zh_order = np.sort(Zh.ravel(order='F'))
    bin_edges = _hist_edges(Zh_order)
    n0 = _hist_pdf(Zh_order, bin_edges)
    widths = np.diff(bin_edges)
    pos, zero = PS > 0, PS == 0
    PS_pos = PS[pos]

    # Initialize errors and Zh
    err_H, err_PS, err_PS0 = [1.0], [1.0], [1.0]
    zh = _fft2(Zh)
    while True:
        # Step 1: Fix the power spectrum by FFT filter
        phase = np.angle(zh)
        Zs = np.real(_ifft2(PS * np.exp(1j * phase)))
        # Error in height distribution
        n = _hist_pdf(Zs, bin_edges)
        err_H.append(float(np.sum(np.abs(n - n0) * widths)))

        # Step 2: Fix the height distribution rank ordering
        flat = Zs.ravel(order='F')
        ind = np.argsort(flat, kind='stable')
        ranked = np.empty_like(flat)
        ranked[ind] = Zh_order
        Zh = ranked.reshape((M, N), order='F')
        zh = _fft2(Zh)
        # Error in power spectrum
        Zh_abs = np.abs(zh)
        err_PS.append(math.sqrt(np.mean((1 - Zh_abs[pos] / PS_pos) ** 2)))
        err_PS0.append(math.sqrt(np.mean(Zh_abs[zero] ** 2)) / PS_pos.mean())

        if len(err_H) == max_it:
            stop = 'maximum_iterations'
            break
        if abs(err_H[-1] - err_H[-2]) / max(err_H[-1], EPS) < min_speed:
            stop = 'histogram_stagnation'
            break
        if err_H[-1] < min_err or (err_PS[-1] < min_err and err_PS0[-1] < min_err):
            stop = 'tolerance'
            break

    # Rescale and output
    history = Struct(H=np.array(err_H), PS=np.array(err_PS), PS0=np.array(err_PS0), stopReason=stop)
    return Zh * Sq + Zh_mean, Zs * Sq + Zh_mean, history


def _fractal_surf_hu_tonder_anisotropic(H, Llong, Lshort, Sq, Lx, Ly, M, N, af, PS, rng):
    # Random surface with power spectral density C(q) ~ q^(-2*H-2) by the
    # digital-filter technique of Hu and Tonder (1992), Int. J. Mach. Tools
    # Manufact. 32(1/2) 83-90: filter a random sample to impose the spectrum.
    # M, N are nodes in y and x. Returns the filtered sample and the spectrum
    # in FFT order (the original's X, Y outputs are unused and omitted).

    # Transform the cut-off's to wavenumber
    minq = Lx / Llong                            # Long wavelengths cutoff
    maxq = Lx / Lshort if Lshort else np.inf     # Short wavelengths cutoff

    # Matrix form of the wavenumbers
    qx, qy = np.meshgrid(np.arange(-(N // 2), N // 2 + 1, dtype=float),
                         np.arange(-(M // 2), M // 2 + 1, dtype=float))

    # Account for anisotropy when defining the 1D wavenumber
    q = np.sqrt((af * qx) ** 2 + (qy / af) ** 2)

    # Define the power spectral density
    with np.errstate(divide='ignore'):
        if PS == 'Self-affine':
            C = q ** (-2 * (H + 1))
        else:  # Exponential
            C = H / (2 * math.pi * (q**2 + H**2) ** (3 / 2))
    # Apply the cut-off's
    C[q < minq] = 0
    C[q > maxq] = 0
    # Rearange C to match the FFT ordering
    C = np.fft.fftshift(C[:-1, :-1])

    # Define the filter function as a function of C
    Hf = np.sqrt(C)

    # Gaussian random numbers generation
    eta = rng.randn(M, N)

    # Filter eta to obtain a surface with the defined C(q). The spectrum is
    # conjugate symmetric, for which MATLAB's ifft2 returns a real array.
    Z = np.real(_ifft2(Hf * _fft2(eta)))

    # Scale Z
    Z = Z - Z.mean()
    Z = Z / math.sqrt(np.mean(Z**2))
    return Z * Sq, C


def _fftbandfilter(z, Llong, Lshort, Lx):
    M, N = z.shape

    # Define low- and high frequency cutoff
    lowq = Lx / Llong                           # <=> Long wavelength cutoff (Relative)
    highq = Lx / Lshort if Lshort else np.inf   # <=> Short wavelength cutoff (Relative)

    # Frequency vector
    qx, qy = np.meshgrid(np.arange(-(N // 2), N // 2 + 1, dtype=float),
                         np.arange(-(M // 2), M // 2 + 1, dtype=float))
    q = np.sqrt(qx**2 + qy**2)

    # Set up the transfer function
    H = np.ones_like(q)
    H[q < lowq] = 0   # Apply long wavelength cutoff
    H[q > highq] = 0  # Apply short wavelength cutoff
    H = np.fft.fftshift(H[:-1, :-1])

    # Filter surface (and truncate imaginary noise)
    return np.real(_ifft2(_fft2(z) * H))


def _my_surf_param(X, Y, Z):
    # Adjusting the mean value to zero
    Z = Z - Z.mean()
    z = Z.ravel(order='F')
    P = Struct()

    # Height parameters
    P.a = np.mean(np.abs(z))           # Average roughness
    P.q = math.sqrt(np.mean(z**2))     # rms roughness
    P.sk = np.mean(z**3) / P.q**3      # Skewness
    P.ku = np.mean(z**4) / P.q**4      # Kurtosis

    P.p = z.max()
    P.v = z.min()
    P.z = P.p - P.v

    # Average "max" and "min" for 5 highest and lowest values.
    zsort = np.sort(z)
    P.p5 = zsort[-5:].mean()
    P.v5 = zsort[:5].mean()
    P.z5 = zsort[-5:].mean() - zsort[:5].mean()

    # Hybrid parameter(s)
    dzdy = _mydiffy(Y, Z)
    dzdx = _mydiffx(X, Z)
    P.dq = math.sqrt(np.mean(dzdy**2 + dzdx**2))  # rms slope

    # Abbot curve parameters
    #   Obtain cumulative function
    AB = np.sort(z)[::-1]
    AB = np.interp(_linspace(1, AB.size, 1e3), np.arange(1, AB.size + 1), AB)
    AB = np.unique(AB)[::-1]
    nbins = AB.size
    XA = _linspace(0, 100, nbins)
    #   Fit line with minimum slope taking 40 % of the data
    N40 = math.floor(0.4 * nbins)
    slope, inter = np.inf, np.nan
    for jj in range(nbins - N40):
        # fit slope
        y = AB[jj:jj + N40 + 1]
        x = XA[jj:jj + N40 + 1]
        s = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
        n = y.mean() - s * x.mean()
        # Save value if minimum
        if abs(s) < abs(slope):
            slope, inter = s, n

    # Find parameters
    #   Heights at intercepts
    H1 = inter
    H2 = inter + slope * 100
    #   Percentage at intercepts height (MATLAB max/min ignore NaN)
    P.Mr1 = float(np.fmax(0.11, _interp1(AB, XA, H1)))
    P.Mr2 = float(np.fmin(_interp1(AB, XA, H2), 100 - 0.11))
    #   Areas of peaks and valeys on the curve
    m1, m2 = XA < P.Mr1, XA > P.Mr2
    A1 = _trapz(XA[m1], AB[m1] - H1)
    A2 = _trapz(XA[m2], H2 - AB[m2])
    #   Core Sk parameters
    P.Sk = H1 - H2
    P.Spk = 2 * A1 / P.Mr1
    P.Svk = 2 * A2 / (100 - P.Mr2)
    #   Volumes
    H10 = float(np.interp(10, XA, AB))
    H80 = float(np.interp(80, XA, AB))
    P.Vmp = _trapz(XA[XA < 10], AB[XA < 10] - H10) / 100
    P.Vmc = _trapz(XA[XA < 80], AB[XA < 80] - H80) / 100 - P.Vmp
    P.Vvv = _trapz(XA[XA > 80], H80 - AB[XA > 80]) / 100
    P.Vvc = _trapz(XA[XA > 10], H10 - AB[XA > 10]) / 100 - P.Vvv
    # Sk (core height): difference of heights at material ratio 0 % and 100 %
    # on the equivalent line. Spk / Svk: average height of the reduced peaks /
    # depth of the reduced valleys; Mr1 / Mr2: material ratios separating the
    # reduced peaks / valleys from the core surface.
    return P


def _mydiffx(x, F):
    """Second-order finite-difference derivative of F (M x N) along columns.

    Author(s): A. Almqvist, Copyright 2009- Andreas Almqvist (mydiffx, order 2).
    """
    N = x.shape[1]
    return np.hstack((
        (-3 * F[:, [0]] + 4 * F[:, [1]] - F[:, [2]]),
        F[:, 2:] - F[:, :-2],
        F[:, [N - 3]] - 4 * F[:, [N - 2]] + 3 * F[:, [N - 1]],
    )) / 2 / (x[0, 1] - x[0, 0])


def _mydiffy(y, F):
    return _mydiffx(y.T, F.T).T


# --------------------------------------------------------------------------
# Command line: python fractal_surface.py Nx=512 H=0.5 outputPrefix=my_surface
# --------------------------------------------------------------------------

def _main(argv):
    opts = {}
    for arg in argv:
        name, sep, text = arg.partition('=')
        if not sep:
            sys.exit(f'usage: {sys.argv[0]} [option=value ...]   e.g. Nx=256 H=0.8 distribution=Weibull')
        try:
            opts[name] = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            opts[name] = {'true': True, 'false': False}.get(text.lower(), text)
    out = fractal_surface(opts)
    d, h = out.diagnostics, out.history
    print(f'{out.options.Nx}x{out.options.Ny} {out.options.distribution} surface: '
          f'{h.H.size - 1} iterations ({h.stopReason}), Sq={d.Sq:.4g}, '
          f'skew={d.skewness:.4f}, kurtosis={d.kurtosis:.4f}, H fit={d.Hfitted:.4f}')


if __name__ == '__main__':
    _main(sys.argv[1:])


# Original source license
# Copyright (c) 2023, Andreas Almqvist
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution
#
# * Neither the name of  nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
