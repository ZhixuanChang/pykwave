# pykwave — JAX implementation of k-Wave pseudospectral acoustic simulation.
# Based on the k-Wave MATLAB Toolbox (http://www.k-wave.org) by B.E. Treeby & B.T. Cox.
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import pytest
from pykwave.pml import get_pml

def test_pml_shape_axis0():
    pml = get_pml(64, 1e-3, 2e-7, 1500.0, 20, 2.0, staggered=False, axis=0)
    assert pml.shape == (64, 1)

def test_pml_shape_axis1():
    pml = get_pml(64, 1e-3, 2e-7, 1500.0, 20, 2.0, staggered=False, axis=1)
    assert pml.shape == (1, 64)

def test_pml_interior_is_one():
    pml = get_pml(64, 1e-3, 2e-7, 1500.0, 20, 2.0, staggered=False, axis=0)
    interior = np.array(pml[20:-20, 0])
    np.testing.assert_allclose(interior, 1.0, atol=1e-6)

def test_pml_boundary_less_than_one():
    pml = get_pml(64, 1e-3, 2e-7, 1500.0, 20, 2.0, staggered=False, axis=0)
    arr = np.array(pml[:, 0])
    assert arr[0] < 1.0       # attenuation at outermost point
    assert arr[63] < 1.0
    assert arr[20] == pytest.approx(1.0, abs=1e-6)  # first interior point

def test_pml_zero_alpha_is_ones():
    pml = get_pml(64, 1e-3, 2e-7, 1500.0, 20, 0.0, staggered=False, axis=0)
    np.testing.assert_allclose(np.array(pml), 1.0, atol=1e-6)

def test_staggered_differs_from_non_staggered():
    p1 = get_pml(64, 1e-3, 2e-7, 1500.0, 20, 2.0, staggered=False, axis=0)
    p2 = get_pml(64, 1e-3, 2e-7, 1500.0, 20, 2.0, staggered=True,  axis=0)
    assert not np.allclose(np.array(p1), np.array(p2))
