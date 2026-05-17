# tests/test_kspace_first_order_2d.py
import numpy as np
import pytest
from pykwave import KWaveGrid, Medium, Source, Sensor, kspace_first_order_2d


def test_point_source_wave_speed():
    """
    Point source at grid centre in homogeneous water-like medium.
    Sensor ring at radius r. Expected arrival time: t_arrival = r / c0.
    Check that peak pressure at the ring arrives within 2 time steps of expected.
    """
    Nx, Ny = 128, 128
    dx = 1e-3   # 1 mm
    c0 = 1500.0  # m/s
    r_pts = 20   # sensor ring radius in grid points

    g = KWaveGrid(Nx, dx, Ny, dx)
    g.make_time(c0, cfl=0.3)

    m = Medium(sound_speed=c0, density=1000.0)

    p0 = np.zeros((Nx, Ny), dtype=np.float32)
    p0[Nx // 2, Ny // 2] = 1.0
    src = Source(p0=p0)

    # Circular sensor ring
    cx, cy = Nx // 2, Ny // 2
    mask = np.zeros((Nx, Ny), dtype=np.float32)
    for angle in np.linspace(0, 2 * np.pi, 360, endpoint=False):
        xi = int(round(cx + r_pts * np.cos(angle)))
        yi = int(round(cy + r_pts * np.sin(angle)))
        if 0 <= xi < Nx and 0 <= yi < Ny:
            mask[xi, yi] = 1.0
    sen = Sensor(mask=mask)

    data = kspace_first_order_2d(g, m, src, sen)  # (N_sensors, Nt)

    # Expected arrival: r in metres / c0
    r_m = r_pts * dx
    t_expected = r_m / c0
    t_index_expected = int(round(t_expected / g.dt))

    # Peak of mean pressure over sensor ring
    mean_p = np.array(data).mean(axis=0)   # (Nt,)
    t_peak = int(np.argmax(np.abs(mean_p)))

    assert abs(t_peak - t_index_expected) <= 2, (
        f"Wave arrived at t_index={t_peak}, expected ~{t_index_expected} "
        f"(error = {abs(t_peak - t_index_expected)} steps)"
    )


def test_lossless_homogeneous_no_source_returns_empty():
    g = KWaveGrid(32, 1e-3, 32, 1e-3)
    g.make_time(1500.0)
    m = Medium(sound_speed=1500.0, density=1000.0)
    src = Source()   # no source
    mask = np.zeros((32, 32)); mask[8, 8] = 1.0
    sen = Sensor(mask=mask)
    data = kspace_first_order_2d(g, m, src, sen)
    assert float(data.max()) == pytest.approx(0.0, abs=1e-8)


def test_sensor_record_p_max():
    Nx, Ny = 32, 32
    g = KWaveGrid(Nx, 1e-3, Ny, 1e-3)
    g.make_time(1500.0)
    m = Medium(sound_speed=1500.0, density=1000.0)
    p0 = np.zeros((Nx, Ny), dtype=np.float32); p0[16, 16] = 500.0
    src = Source(p0=p0)
    mask = np.zeros((Nx, Ny)); mask[8, :] = 1.0
    sen = Sensor(mask=mask, record=['p', 'p_max', 'p_min'])
    sd = kspace_first_order_2d(g, m, src, sen)
    assert sd.p is not None
    assert sd.p_max is not None
    assert sd.p_min is not None
    # p_max must be >= p_min everywhere
    assert bool((sd.p_max >= sd.p_min).all())


def test_absorbing_medium_decays():
    """Absorbing medium must attenuate pressure more than lossless.

    The fractional-power absorbing EOS (alpha_power != 2) has a numerical
    stability constraint: the absorb_eta coefficient times the maximum
    spectral weight (k_max^alpha_power) times c0^2 * dt must be < 1.
    For dx=1mm, CFL=0.3, c0=1500, alpha_power=1.5 this limits alpha_coeff
    to roughly 1e-9.  We use alpha_coeff=1e-10 which is well inside the
    stable regime and still produces measurable attenuation.
    """
    Nx, Ny = 64, 64
    dx = 1e-3
    g = KWaveGrid(Nx, dx, Ny, dx)
    g.make_time(1500.0, cfl=0.3)
    p0 = np.zeros((Nx, Ny), dtype=np.float32); p0[32, 32] = 1000.0
    mask = np.zeros((Nx, Ny)); mask[16, :] = 1.0
    sen = Sensor(mask=mask)

    m_lossless  = Medium(sound_speed=1500.0, density=1000.0)
    m_absorbing = Medium(sound_speed=1500.0, density=1000.0,
                         alpha_coeff=1e-10, alpha_power=1.5)

    d_loss = kspace_first_order_2d(g, m_lossless,  Source(p0=p0.copy()), sen)
    d_abs  = kspace_first_order_2d(g, m_absorbing, Source(p0=p0.copy()), sen)

    assert float(np.abs(d_abs).max()) < float(np.abs(d_loss).max())


def test_validation_missing_time():
    g = KWaveGrid(32, 1e-3, 32, 1e-3)
    m = Medium(sound_speed=1500.0, density=1000.0)
    with pytest.raises(ValueError, match="time must be set"):
        kspace_first_order_2d(g, m, Source(), None)


def test_validation_p_without_mask():
    g = KWaveGrid(32, 1e-3, 32, 1e-3); g.make_time(1500.0)
    m = Medium(sound_speed=1500.0, density=1000.0)
    import jax.numpy as jnp
    with pytest.raises(ValueError, match="p_mask"):
        kspace_first_order_2d(g, m, Source(p=jnp.zeros((1, g.Nt))), None)
