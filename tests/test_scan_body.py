import numpy as np
import jax
import jax.numpy as jnp
import pytest
from pykwave import KWaveGrid, Medium, Source, Sensor
from pykwave._precompute import build_flags_and_ops
from pykwave._scan_body import SimState, make_init_state, scan_step

def _make_lossless_setup():
    g = KWaveGrid(32, 1e-3, 32, 1e-3)
    g.make_time(1500.0, cfl=0.3)
    m = Medium(sound_speed=1500.0, density=1000.0)
    p0 = np.zeros((32, 32), dtype=np.float32)
    p0[16, 16] = 1000.0
    src = Source(p0=p0)
    mask = np.zeros((32, 32)); mask[8, :] = 1
    sen = Sensor(mask=mask)
    flags, ops = build_flags_and_ops(g, m, src, sen, {})
    return flags, ops, g.Nt

def test_init_state_shapes():
    flags, ops, Nt = _make_lossless_setup()
    state = make_init_state(ops)
    assert state.p.shape == (32, 32)
    assert state.p_k.shape == (32, 32)
    assert state.rhox.shape == (32, 32)

def test_single_step_runs():
    flags, ops, Nt = _make_lossless_setup()
    state = make_init_state(ops)
    from functools import partial
    step = partial(scan_step, ops=ops, flags=flags)
    xs0 = {
        'p_src': ops['xs_p'][0],
        'ux_src': ops['xs_ux'][0],
        'uy_src': ops['xs_uy'][0],
        't_index': jnp.array(0, dtype=jnp.int32),
        'tr_bc':   ops['tr_bc'][0],
    }
    new_state, out = step(state, xs0)
    assert new_state.p.shape == (32, 32)
    # After t=0 with p0, pressure field must be non-zero
    assert float(jnp.max(jnp.abs(new_state.p))) > 0.0

def test_pressure_conserved_over_scan():
    """Max pressure must not grow unboundedly for 10 steps."""
    flags, ops, Nt = _make_lossless_setup()
    state = make_init_state(ops)
    from functools import partial
    step = partial(scan_step, ops=ops, flags=flags)
    xs_stacked = {
        'p_src':    ops['xs_p'][:10],
        'ux_src':   ops['xs_ux'][:10],
        'uy_src':   ops['xs_uy'][:10],
        't_index':  jnp.arange(10, dtype=jnp.int32),
        'tr_bc':    ops['tr_bc'][:10],
    }
    import jax.lax as lax
    final, outputs = lax.scan(step, state, xs_stacked)
    p_max = float(jnp.max(jnp.abs(final.p)))
    assert p_max < 5000.0   # must not blow up
    assert p_max > 0.0
