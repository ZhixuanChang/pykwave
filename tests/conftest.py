# pykwave — JAX implementation of k-Wave pseudospectral acoustic simulation.
# Based on the k-Wave MATLAB Toolbox (http://www.k-wave.org) by B.E. Treeby & B.T. Cox.
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import pytest

@pytest.fixture
def small_grid_params():
    """64x64 grid, 1mm spacing."""
    return dict(Nx=64, dx=1e-3, Ny=64, dy=1e-3)

@pytest.fixture
def homogeneous_medium_params():
    return dict(sound_speed=1500.0, density=1000.0)
