# pykwave — JAX implementation of k-Wave pseudospectral acoustic simulation.
# Based on the k-Wave MATLAB Toolbox (http://www.k-wave.org) by B.E. Treeby & B.T. Cox.
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import pytest
from pykwave.utils import db2neper, smooth, interp_rho_staggered

def test_db2neper_zero():
    assert db2neper(0.0, 1.5) == pytest.approx(0.0)

def test_db2neper_known():
    # alpha=0.5 dB/(MHz^1.5 cm), y=1.5
    # alpha_np = 0.5 * (1e3)^1.5 / (100 * 20 * log10(e))
    import math
    expected = 0.5 * (1e3) ** 1.5 / (100 * 20 * math.log10(math.e))
    assert db2neper(0.5, 1.5) == pytest.approx(expected, rel=1e-5)

def test_smooth_reduces_discontinuity():
    img = np.zeros((64, 64), dtype=np.float32)
    img[32, 32] = 1.0
    out = smooth(img)
    # Smoothed image must spread energy around (32,32)
    assert out[32, 32] < 1.0
    assert out.sum() == pytest.approx(img.sum(), rel=0.01)

def test_smooth_shape():
    img = np.random.randn(32, 48).astype(np.float32)
    assert smooth(img).shape == (32, 48)

def test_interp_rho_staggered_homogeneous():
    rho = np.ones((32, 32), dtype=np.float32) * 1000.0
    out = interp_rho_staggered(rho, axis=0)
    np.testing.assert_allclose(out, 1000.0, rtol=1e-5)

def test_interp_rho_staggered_shape():
    rho = np.random.rand(32, 48).astype(np.float32)
    assert interp_rho_staggered(rho, axis=0).shape == (32, 48)
    assert interp_rho_staggered(rho, axis=1).shape == (32, 48)
