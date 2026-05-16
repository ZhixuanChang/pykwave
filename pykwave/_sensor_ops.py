# pykwave/_sensor_ops.py
import jax.numpy as jnp
from pykwave.sensor import SensorData


def build_sensor_data(final_state, stacked_out, sensor, ops, flags) -> SensorData:
    """Convert scan outputs + final carry into SensorData."""
    rs  = ops['record_start_index']   # 0-based first recorded step
    n_rec = int(final_state.n_recorded)
    x1, x2, y1, y2 = ops['x1'], ops['x2'], ops['y1'], ops['y2']

    # stacked_out.p_at_sensor has shape (Nt, N_sensors)
    # Slice off pre-record steps and transpose to (N_sensors, Nt_record)
    def _slice(arr):
        return arr[rs:].T   # (N_sensors, Nt_record)

    sd = SensorData()

    if flags['record_p']:
        sd.p = _slice(stacked_out.p_at_sensor)
        if sensor is not None and sensor.frequency_response is not None:
            sd.p = _gaussian_filter(sd.p, 1.0 / ops['dt'],
                                    sensor.frequency_response[0],
                                    sensor.frequency_response[1])

    if flags['record_p_max']:
        sd.p_max = final_state.p_max_s

    if flags['record_p_min']:
        sd.p_min = final_state.p_min_s

    if flags['record_p_rms']:
        sd.p_rms = jnp.sqrt(final_state.p_rms_sq / jnp.maximum(n_rec, 1))

    if flags['record_p_final'] or flags['time_rev']:
        sd.p_final = final_state.p[x1:x2, y1:y2]

    if flags['record_p_max_all']:
        sd.p_max_all = final_state.p_max_all[x1:x2, y1:y2]

    if flags['record_p_min_all']:
        sd.p_min_all = final_state.p_min_all[x1:x2, y1:y2]

    if flags['record_u']:
        sd.ux = _slice(stacked_out.ux_at_sensor)
        sd.uy = _slice(stacked_out.uy_at_sensor)

    if flags['record_u_max']:
        sd.ux_max = final_state.ux_max_s
        sd.uy_max = final_state.uy_max_s

    if flags['record_u_min']:
        sd.ux_min = final_state.ux_min_s
        sd.uy_min = final_state.uy_min_s

    if flags['record_u_rms']:
        sd.ux_rms = jnp.sqrt(final_state.ux_rms_sq / jnp.maximum(n_rec, 1))
        sd.uy_rms = jnp.sqrt(final_state.uy_rms_sq / jnp.maximum(n_rec, 1))

    if flags['record_u_final']:
        sd.ux_final = final_state.ux_sgx[x1:x2, y1:y2]
        sd.uy_final = final_state.uy_sgy[x1:x2, y1:y2]

    if flags['record_u_max_all']:
        sd.ux_max_all = final_state.ux_max_all[x1:x2, y1:y2]
        sd.uy_max_all = final_state.uy_max_all[x1:x2, y1:y2]

    if flags['record_u_min_all']:
        sd.ux_min_all = final_state.ux_min_all[x1:x2, y1:y2]
        sd.uy_min_all = final_state.uy_min_all[x1:x2, y1:y2]

    if flags['record_u_non_staggered']:
        sd.ux_non_staggered = _slice(stacked_out.ux_ns_at_sensor)
        sd.uy_non_staggered = _slice(stacked_out.uy_ns_at_sensor)

    if flags['record_I'] or flags['record_I_avg']:
        p_slice = _slice(stacked_out.p_at_sensor)
        ux_ns = _slice(stacked_out.ux_ns_at_sensor)
        uy_ns = _slice(stacked_out.uy_ns_at_sensor)
        if flags['record_I']:
            sd.Ix = p_slice * ux_ns
            sd.Iy = p_slice * uy_ns
        if flags['record_I_avg']:
            sd.Ix_avg = jnp.mean(p_slice * ux_ns, axis=1)
            sd.Iy_avg = jnp.mean(p_slice * uy_ns, axis=1)

    return sd


def _gaussian_filter(data, fs, f0, bw_pct):
    """Apply Gaussian frequency-domain filter to sensor_data.p (N_sensors, Nt)."""
    Nt = data.shape[1]
    freqs = jnp.fft.fftfreq(Nt, d=1.0 / fs)
    sigma = f0 * bw_pct / 100.0 / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0)))
    H = jnp.exp(-0.5 * ((freqs - f0) / sigma) ** 2) + \
        jnp.exp(-0.5 * ((freqs + f0) / sigma) ** 2)
    return jnp.real(jnp.fft.ifft(jnp.fft.fft(data, axis=1) * H[None, :], axis=1))
