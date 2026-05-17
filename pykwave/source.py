# pykwave — JAX implementation of k-Wave pseudospectral acoustic simulation.
# Based on the k-Wave MATLAB Toolbox (http://www.k-wave.org) by B.E. Treeby & B.T. Cox.
# SPDX-License-Identifier: LGPL-3.0-or-later

from dataclasses import dataclass
from typing import Any

@dataclass
class Source:
    p0: Any = None        # (Nx, Ny) initial pressure [Pa]
    p: Any = None         # (n_sig, Nt) or (1, Nt) time-varying pressure
    p_mask: Any = None    # (Nx, Ny) binary
    p_mode: str = 'additive'   # 'additive' | 'dirichlet'
    ux: Any = None        # (n_sig, Nt) x-velocity source
    uy: Any = None        # (n_sig, Nt) y-velocity source
    u_mask: Any = None    # (Nx, Ny) binary
    u_mode: str = 'additive'
