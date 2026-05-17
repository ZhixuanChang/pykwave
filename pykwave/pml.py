# pykwave — JAX implementation of k-Wave pseudospectral acoustic simulation.
# Based on the k-Wave MATLAB Toolbox (http://www.k-wave.org) by B.E. Treeby & B.T. Cox.
# SPDX-License-Identifier: LGPL-3.0-or-later

import jax.numpy as jnp
import numpy as np

def get_pml(
    N: int, d: float, dt: float, c: float,
    pml_size: int, pml_alpha: float,
    staggered: bool, axis: int,
) -> jnp.ndarray:
    """Return PML multiplier array shaped for broadcasting along `axis`.

    Matches MATLAB getPML.m: quartic profile, exp(-profile * dt/2).
    """
    x = np.arange(1, pml_size + 1, dtype=np.float64)  # 1..pml_size

    if staggered:
        left  = pml_alpha * (c / d) * (((x + 0.5) - pml_size - 1) / (-pml_size)) ** 4
        right = pml_alpha * (c / d) * ((x + 0.5) / pml_size) ** 4
    else:
        left  = pml_alpha * (c / d) * ((x - pml_size - 1) / (-pml_size)) ** 4
        right = pml_alpha * (c / d) * (x / pml_size) ** 4

    left  = np.exp(-left  * dt / 2)
    right = np.exp(-right * dt / 2)

    pml = np.ones(N, dtype=np.float32)
    pml[:pml_size]       = left.astype(np.float32)
    pml[N - pml_size:]   = right.astype(np.float32)

    pml_jax = jnp.array(pml)
    return pml_jax[:, None] if axis == 0 else pml_jax[None, :]
