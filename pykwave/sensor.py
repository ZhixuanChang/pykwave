# pykwave — JAX implementation of k-Wave pseudospectral acoustic simulation.
# Based on the k-Wave MATLAB Toolbox (http://www.k-wave.org) by B.E. Treeby & B.T. Cox.
# SPDX-License-Identifier: LGPL-3.0-or-later

from dataclasses import dataclass
from typing import Any

VALID_RECORD_FIELDS = frozenset([
    'p', 'p_max', 'p_min', 'p_rms', 'p_final',
    'p_max_all', 'p_min_all',
    'u', 'u_max', 'u_min', 'u_rms', 'u_final',
    'u_max_all', 'u_min_all', 'u_non_staggered',
    'I', 'I_avg',
])

@dataclass
class Sensor:
    mask: Any                              # binary (Nx,Ny) or Cartesian (2,N)
    record: list[str] = None
    record_start_index: int = 1           # 1-based, matches MATLAB
    time_reversal_boundary_data: Any = None  # (N_sensors, Nt)
    frequency_response: tuple = None      # (center_hz, bandwidth_pct)
    directivity_angle: Any = None         # (N_sensors,) [rad]
    directivity_size: float = None
    directivity_pattern: str = None       # 'pressure' | 'gradient'


@dataclass
class SensorData:
    p: Any = None
    p_max: Any = None
    p_min: Any = None
    p_rms: Any = None
    p_final: Any = None
    p_max_all: Any = None
    p_min_all: Any = None
    ux: Any = None
    uy: Any = None
    ux_max: Any = None
    uy_max: Any = None
    ux_min: Any = None
    uy_min: Any = None
    ux_rms: Any = None
    uy_rms: Any = None
    ux_final: Any = None
    uy_final: Any = None
    ux_max_all: Any = None
    uy_max_all: Any = None
    ux_min_all: Any = None
    uy_min_all: Any = None
    ux_non_staggered: Any = None
    uy_non_staggered: Any = None
    Ix: Any = None
    Iy: Any = None
    Ix_avg: Any = None
    Iy_avg: Any = None
