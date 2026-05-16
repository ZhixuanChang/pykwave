# pykwave/kspace_first_order_2d.py
from __future__ import annotations
from functools import partial
import jax
import jax.numpy as jnp

from pykwave.sensor import VALID_RECORD_FIELDS, SensorData
from pykwave._precompute import build_flags_and_ops
from pykwave._scan_body import make_init_state, scan_step
from pykwave._sensor_ops import build_sensor_data


def kspace_first_order_2d(kgrid, medium, source, sensor=None, **options):
    """2D k-space first-order acoustic simulation (JAX implementation).

    Matches MATLAB kspaceFirstOrder2D API. Returns sensor_data as a plain
    JAX array if sensor.record is not set, otherwise a SensorData dataclass.
    """
    _validate(kgrid, medium, source, sensor)

    flags, ops = build_flags_and_ops(kgrid, medium, source, sensor, options)

    init_state = make_init_state(ops)

    Nt = ops['Nt']
    xs_stacked = {
        'p_src':   ops['xs_p'],
        'ux_src':  ops['xs_ux'],
        'uy_src':  ops['xs_uy'],
        't_index': jnp.arange(Nt, dtype=jnp.int32),
        'tr_bc':   ops['tr_bc'],
    }

    # For time reversal, run Nt-1 steps
    if flags['time_rev']:
        xs_stacked = {k: v[:Nt - 1] for k, v in xs_stacked.items()}

    step_fn = partial(scan_step, ops=ops, flags=flags)

    @jax.jit
    def _run(state, xs):
        return jax.lax.scan(step_fn, state, xs)

    final_state, stacked_out = _run(init_state, xs_stacked)

    sd = build_sensor_data(final_state, stacked_out, sensor, ops, flags)

    # Return plain array if no sensor.record specified (MATLAB default behaviour)
    if not flags['time_rev'] and (sensor is None or sensor.record is None):
        return sd.p if sd.p is not None else jnp.array([])

    # Time reversal returns p_final
    if flags['time_rev']:
        return sd.p_final

    return sd


def _validate(kgrid, medium, source, sensor):
    if kgrid.dim != 2:
        raise ValueError(f"kgrid.dim must be 2, got {kgrid.dim}")
    if kgrid.Nt is None or kgrid.dt is None:
        raise ValueError("kgrid time must be set via make_time() or set_time() before simulation.")
    if medium.sound_speed is None:
        raise ValueError("medium.sound_speed is required.")
    if (medium.alpha_coeff is None) != (medium.alpha_power is None):
        raise ValueError("medium.alpha_coeff and medium.alpha_power must both be set or both be None.")
    if source is not None and source.p is not None and source.p_mask is None:
        raise ValueError("source.p_mask is required when source.p is set.")
    if source is not None and (source.ux is not None or source.uy is not None) and source.u_mask is None:
        raise ValueError("source.u_mask is required when source.ux or source.uy is set.")
    if sensor is not None and sensor.record is not None:
        unknown = set(sensor.record) - VALID_RECORD_FIELDS
        if unknown:
            raise ValueError(f"Unknown sensor.record fields: {unknown}")
