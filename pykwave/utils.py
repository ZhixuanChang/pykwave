# pykwave — JAX implementation of k-Wave pseudospectral acoustic simulation.
# Based on the k-Wave MATLAB Toolbox (http://www.k-wave.org) by B.E. Treeby & B.T. Cox.
# SPDX-License-Identifier: LGPL-3.0-or-later

import math
import numpy as np
from scipy.ndimage import map_coordinates

def db2neper(alpha: float, y: float) -> float:
    """Convert absorption from dB/(MHz^y cm) to Np/(m (rad/s)^y).

    Matches MATLAB db2neper.m: alpha_np = alpha * (1e-3)^(-y) / (100*20*log10(e))
    """
    return alpha * (1e-3) ** (-y) / (100.0 * 20.0 * math.log10(math.e))


def smooth(img: np.ndarray, restore_max: bool = False) -> np.ndarray:
    """Apply Hann-window smoothing in frequency domain (matches MATLAB smooth.m)."""
    Nx, Ny = img.shape
    win = np.outer(np.hanning(Nx), np.hanning(Ny)).astype(np.float64)
    win /= win.max()

    max_val = float(np.max(np.abs(img))) if restore_max else None

    img_f  = np.fft.fftshift(np.fft.fft2(img.astype(np.float64)))
    result = np.real(np.fft.ifft2(np.fft.ifftshift(img_f * win)))

    if restore_max and max_val and max_val > 0:
        cur = float(np.max(np.abs(result)))
        if cur > 0:
            result *= max_val / cur

    return result.astype(img.dtype)


def interp_rho_staggered(rho0: np.ndarray, axis: int) -> np.ndarray:
    """Interpolate rho0 at staggered grid locations (+0.5 grid pts along axis).

    Uses linear interpolation with nearest-neighbour extrapolation at boundaries,
    matching MATLAB interpn with '*linear' and NaN→original value replacement.
    """
    coords = np.mgrid[:rho0.shape[0], :rho0.shape[1]].astype(np.float64)
    coords[axis] += 0.5
    result = map_coordinates(rho0.astype(np.float64), coords, order=1, mode='nearest')
    return result.astype(rho0.dtype)
