# pykwave — JAX implementation of k-Wave pseudospectral acoustic simulation.
# Based on the k-Wave MATLAB Toolbox (http://www.k-wave.org) by B.E. Treeby & B.T. Cox.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""
MATLAB parity tests. Fixtures are .npz files in tests/fixtures/ generated
by running tests/fixtures/generate_fixtures.m in MATLAB k-Wave.
Skip gracefully if fixture files are absent.
"""
import os
import numpy as np
import pytest
from pykwave import KWaveGrid, Medium, Source, Sensor, kspace_first_order_2d

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')


def _load(name):
    """Load fixture .npz file, skip test if not found."""
    path = os.path.join(FIXTURE_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"Fixture {name} not found — run generate_fixtures.m first")
    return np.load(path)


def _make_base_setup(alpha_coeff=None, alpha_power=None):
    """Create baseline grid, medium, source, and sensor for parity tests."""
    Nx, Ny, dx = 64, 64, 1e-3
    g = KWaveGrid(Nx, dx, Ny, dx)
    g.make_time(1500.0, cfl=0.3)
    m = Medium(sound_speed=1500.0, density=1000.0,
               alpha_coeff=alpha_coeff, alpha_power=alpha_power)
    p0 = np.zeros((Nx, Ny), dtype=np.float32)
    p0[Nx // 2, Ny // 2] = 1000.0
    src = Source(p0=p0)
    mask = np.zeros((Nx, Ny))
    mask[Nx // 4, :] = 1.0
    sen = Sensor(mask=mask)
    return g, m, src, sen


def test_parity_lossless():
    """Test parity with MATLAB k-Wave for lossless homogeneous medium."""
    f = _load('fixture_lossless.npz')
    ref = f['sensor_data']   # (N_sensors, Nt)
    g, m, src, sen = _make_base_setup()
    data = np.array(kspace_first_order_2d(g, m, src, sen))
    rel_err = np.max(np.abs(data - ref)) / (np.max(np.abs(ref)) + 1e-30)
    assert rel_err < 1e-3, f"Lossless parity rel_err = {rel_err:.2e}"


def test_parity_absorbing():
    """Test parity with MATLAB k-Wave for absorbing medium."""
    f = _load('fixture_absorbing.npz')
    ref = f['sensor_data']
    g, m, src, sen = _make_base_setup(alpha_coeff=0.5, alpha_power=1.5)
    data = np.array(kspace_first_order_2d(g, m, src, sen))
    rel_err = np.max(np.abs(data - ref)) / (np.max(np.abs(ref)) + 1e-30)
    assert rel_err < 1e-3, f"Absorbing parity rel_err = {rel_err:.2e}"
