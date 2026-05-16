import numpy as np
import pytest
from pykwave.grid import KWaveGrid


def test_dim():
    g = KWaveGrid(64, 1e-3, 48, 1e-3)
    assert g.dim == 2


def test_grid_shape():
    g = KWaveGrid(64, 1e-3, 48, 1e-3)
    assert g.x.shape == (64, 48)
    assert g.y.shape == (64, 48)


def test_x_vec_centered():
    g = KWaveGrid(64, 1e-3, 64, 1e-3)
    x = np.array(g.x_vec)
    assert abs(x.mean()) < 1e-9          # centered at 0
    assert abs(x[0] + x[-1]) < 1e-9     # symmetric


def test_kx_vec_sorted_order():
    """kx_vec must be in fftshift (sorted) order: DC at center."""
    g = KWaveGrid(64, 1e-3, 64, 1e-3)
    kx = np.array(g.kx_vec)
    # DC (k=0) must be near the middle, not at index 0
    assert np.argmin(np.abs(kx)) == len(kx) // 2


def test_k_min_at_center():
    """k grid (fftshift order) has its minimum at the center element."""
    g = KWaveGrid(64, 1e-3, 64, 1e-3)
    k = np.array(g.k)
    assert k[32, 32] == pytest.approx(0.0, abs=1e-6)


def test_make_time_cfl():
    g = KWaveGrid(64, 1e-3, 64, 1e-3)
    g.make_time(1500.0, cfl=0.3)
    assert g.dt == pytest.approx(0.3 * 1e-3 / 1500.0, rel=1e-6)
    assert g.Nt > 0
    assert g.t_array.shape == (g.Nt,)


def test_set_time():
    g = KWaveGrid(64, 1e-3, 64, 1e-3)
    g.set_time(100, 2e-7)
    assert g.Nt == 100
    assert g.dt == pytest.approx(2e-7)


def test_total_grid_points():
    g = KWaveGrid(64, 1e-3, 48, 1e-3)
    assert g.total_grid_points == 64 * 48
