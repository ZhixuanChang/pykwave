# pykwave — JAX implementation of k-Wave pseudospectral acoustic simulation.
# Based on the k-Wave MATLAB Toolbox (http://www.k-wave.org) by B.E. Treeby & B.T. Cox.
# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import annotations
import numpy as np
import jax.numpy as jnp
from pykwave.pml import get_pml
from pykwave.utils import db2neper, smooth, interp_rho_staggered
from pykwave.sensor import VALID_RECORD_FIELDS


def build_flags_and_ops(kgrid, medium, source, sensor, options: dict) -> tuple[dict, dict]:
    """Return (flags, ops) for use in the scan body."""
    Nx, Ny = kgrid.Nx, kgrid.Ny
    dt = kgrid.dt
    dx, dy = kgrid.dx, kgrid.dy

    flags = _build_flags(medium, source, sensor, options)
    ops   = _build_ops(kgrid, medium, source, sensor, options, flags)
    return flags, ops


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

def _build_flags(medium, source, sensor, options) -> dict:
    record = set(sensor.record) if sensor is not None and sensor.record else {'p'}

    # EOS selection
    has_absorption = (medium.alpha_coeff is not None and medium.alpha_power is not None)
    if has_absorption and medium.alpha_power == 2.0:
        eos = 'stokes'
    elif has_absorption:
        eos = 'absorbing'
    else:
        eos = 'lossless'

    return {
        'eos': eos,
        'nonlinear': medium.BonA is not None,
        'use_sg': True,   # staggered grid always enabled
        'source_p0': source is not None and source.p0 is not None,
        'source_p':  source is not None and source.p is not None,
        'source_ux': source is not None and source.ux is not None,
        'source_uy': source is not None and source.uy is not None,
        'p_mode': getattr(source, 'p_mode', 'additive') if source else 'additive',
        'u_mode': getattr(source, 'u_mode', 'additive') if source else 'additive',
        'time_rev': sensor is not None and sensor.time_reversal_boundary_data is not None,
        'use_sensor': sensor is not None,
        'binary_sensor_mask': _is_binary_mask(sensor),
        'record_p':          'p'           in record,
        'record_p_max':      'p_max'       in record,
        'record_p_min':      'p_min'       in record,
        'record_p_rms':      'p_rms'       in record,
        'record_p_final':    'p_final'     in record,
        'record_p_max_all':  'p_max_all'   in record,
        'record_p_min_all':  'p_min_all'   in record,
        'record_u':          'u'           in record,
        'record_u_max':      'u_max'       in record,
        'record_u_min':      'u_min'       in record,
        'record_u_rms':      'u_rms'       in record,
        'record_u_final':    'u_final'     in record,
        'record_u_max_all':  'u_max_all'   in record,
        'record_u_min_all':  'u_min_all'   in record,
        'record_u_non_staggered': 'u_non_staggered' in record,
        'record_I':          'I'           in record,
        'record_I_avg':      'I_avg'       in record,
    }

def _is_binary_mask(sensor) -> bool:
    if sensor is None:
        return True
    mask = np.asarray(sensor.mask)
    return mask.ndim == 2   # 2D array → binary; (2, N) → Cartesian


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------

def _build_ops(kgrid, medium, source, sensor, options, flags) -> dict:
    Nx, Ny = kgrid.Nx, kgrid.Ny
    dt = float(kgrid.dt)
    dx, dy = float(kgrid.dx), float(kgrid.dy)

    # ---- PML settings ----
    pml_size  = options.get('pml_size', 20)
    pml_alpha = options.get('pml_alpha', 2.0)
    pml_x_size  = pml_size[0] if hasattr(pml_size,  '__len__') else pml_size
    pml_y_size  = pml_size[1] if hasattr(pml_size,  '__len__') else pml_size
    pml_x_alpha = pml_alpha[0] if hasattr(pml_alpha, '__len__') else pml_alpha
    pml_y_alpha = pml_alpha[1] if hasattr(pml_alpha, '__len__') else pml_alpha

    # ---- Reference sound speed ----
    c0_arr = np.asarray(medium.sound_speed, dtype=np.float32)
    if medium.sound_speed_ref is not None:
        c_ref = float(medium.sound_speed_ref)
    else:
        c_ref = float(c0_arr.max())
    c0_jax = jnp.array(np.broadcast_to(c0_arr, (Nx, Ny)))

    # ---- Density on staggered grids ----
    rho0_arr = np.asarray(medium.density, dtype=np.float32) if medium.density is not None \
               else np.ones((Nx, Ny), dtype=np.float32)
    rho0_arr = np.broadcast_to(rho0_arr, (Nx, Ny)).copy()
    if flags['use_sg'] and rho0_arr.ndim == 2:
        rho0_sgx = interp_rho_staggered(rho0_arr, axis=0)
        rho0_sgy = interp_rho_staggered(rho0_arr, axis=1)
    else:
        rho0_sgx = rho0_sgy = rho0_arr

    # ---- PML arrays ----
    pml_x     = get_pml(Nx, dx, dt, c_ref, pml_x_size, pml_x_alpha, False, 0)
    pml_x_sgx = get_pml(Nx, dx, dt, c_ref, pml_x_size, pml_x_alpha, True,  0)
    pml_y     = get_pml(Ny, dy, dt, c_ref, pml_y_size, pml_y_alpha, False, 1)
    pml_y_sgy = get_pml(Ny, dy, dt, c_ref, pml_y_size, pml_y_alpha, True,  1)

    # ---- Spectral derivative + stagger-shift operators ----
    kx = np.array(kgrid.kx_vec, dtype=np.float64)[:, None]  # (Nx,1) fftshift order
    ky = np.array(kgrid.ky_vec, dtype=np.float64)[None, :]  # (1,Ny)

    ddx_k_shift_pos = jnp.array(np.fft.ifftshift(1j * kx * np.exp( 1j * kx * dx / 2)), dtype=jnp.complex64)
    ddx_k_shift_neg = jnp.array(np.fft.ifftshift(1j * kx * np.exp(-1j * kx * dx / 2)), dtype=jnp.complex64)
    ddy_k_shift_pos = jnp.array(np.fft.ifftshift(1j * ky * np.exp( 1j * ky * dy / 2)), dtype=jnp.complex64)
    ddy_k_shift_neg = jnp.array(np.fft.ifftshift(1j * ky * np.exp(-1j * ky * dy / 2)), dtype=jnp.complex64)

    # ---- k-space correction ----
    k_arr = np.array(np.fft.ifftshift(np.sqrt(kx**2 + ky**2)))  # FFT order
    kappa        = jnp.array(np.sinc(c_ref * k_arr * dt / 2 / np.pi), dtype=jnp.float32)
    source_kappa = jnp.array(np.cos(c_ref * k_arr * dt / 2), dtype=jnp.float32)

    # ---- Absorption variables ----
    absorb_nabla1 = absorb_nabla2 = absorb_tau = absorb_eta = None
    if flags['eos'] in ('absorbing', 'stokes'):
        alpha_np = db2neper(medium.alpha_coeff, medium.alpha_power)
        c0_src   = c_ref  # use reference for operators
        absorb_tau = float(-2.0 * alpha_np * c0_src ** (medium.alpha_power - 1))
        if flags['eos'] == 'absorbing':
            k_fs = np.array(np.fft.ifftshift(np.sqrt(kx**2 + ky**2)))  # FFT order, same as k_arr
            with np.errstate(divide='ignore', invalid='ignore'):
                n1 = k_fs ** (medium.alpha_power - 2)
                n2 = k_fs ** (medium.alpha_power - 1)
            n1[~np.isfinite(n1)] = 0.0
            n2[~np.isfinite(n2)] = 0.0
            absorb_nabla1 = jnp.array(n1, dtype=jnp.float32)
            absorb_nabla2 = jnp.array(n2, dtype=jnp.float32)
            absorb_eta = float(2.0 * alpha_np * c0_src ** medium.alpha_power
                               * np.tan(np.pi * medium.alpha_power / 2))
        # Apply alpha_sign if set
        if medium.alpha_sign is not None:
            absorb_tau *= np.sign(medium.alpha_sign[0])
            if absorb_eta is not None:
                absorb_eta *= np.sign(medium.alpha_sign[1])

    # ---- Non-staggered velocity shift operators ----
    x_shift_neg = jnp.array(np.fft.ifftshift(np.exp(-1j * kx * dx / 2)), dtype=jnp.complex64)
    y_shift_neg = jnp.array(np.fft.ifftshift(np.exp(-1j * ky * dy / 2)), dtype=jnp.complex64)

    # ---- Grid interior bounds (PML inside) ----
    pml_inside = options.get('pml_inside', True)
    if pml_inside:
        x1, x2 = pml_x_size, Nx - pml_x_size
        y1, y2 = pml_y_size, Ny - pml_y_size
    else:
        x1, x2, y1, y2 = 0, Nx, 0, Ny

    # ---- Source indices and pre-scaled signals ----
    p_src_flat_idx = np.array([], dtype=np.int32)
    xs_p  = np.zeros((int(kgrid.Nt), 1), dtype=np.float32)
    xs_ux = np.zeros((int(kgrid.Nt), 1), dtype=np.float32)
    xs_uy = np.zeros((int(kgrid.Nt), 1), dtype=np.float32)
    u_src_flat_idx = np.array([], dtype=np.int32)

    if flags['source_p']:
        pmask = np.asarray(source.p_mask, dtype=bool)
        p_src_flat_idx = np.flatnonzero(pmask).astype(np.int32)
        N_psrc = len(p_src_flat_idx)
        sig = np.asarray(source.p, dtype=np.float32)
        if sig.shape[0] == 1:
            sig = np.tile(sig, (N_psrc, 1))
        Nt = int(kgrid.Nt)
        sig_padded = np.zeros((N_psrc, Nt), dtype=np.float32)
        sig_padded[:, :sig.shape[1]] = sig[:, :Nt]
        # Pre-scale
        c0_at_src = float(c0_arr.ravel()[p_src_flat_idx[0]]) if c0_arr.ndim == 2 else float(c0_arr)
        if flags['p_mode'] == 'dirichlet':
            sig_padded /= (2.0 * c0_at_src ** 2)
        else:
            sig_padded *= (2.0 * dt / (2.0 * c0_at_src * dx))
        xs_p = sig_padded.T  # (Nt, N_psrc)

    if flags['source_ux'] or flags['source_uy']:
        umask = np.asarray(source.u_mask, dtype=bool)
        u_src_flat_idx = np.flatnonzero(umask).astype(np.int32)
        N_usrc = len(u_src_flat_idx)
        Nt = int(kgrid.Nt)
        c0_at_src = float(c0_arr.ravel()[u_src_flat_idx[0]]) if c0_arr.ndim == 2 else float(c0_arr)

        if flags['source_ux']:
            sig = np.asarray(source.ux, dtype=np.float32)
            if sig.shape[0] == 1:
                sig = np.tile(sig, (N_usrc, 1))
            pad = np.zeros((N_usrc, Nt), dtype=np.float32)
            pad[:, :sig.shape[1]] = sig[:, :Nt]
            if flags['u_mode'] == 'additive':
                pad *= (2.0 * c0_at_src * dt / dx)
            xs_ux = pad.T  # (Nt, N_usrc)

        if flags['source_uy']:
            sig = np.asarray(source.uy, dtype=np.float32)
            if sig.shape[0] == 1:
                sig = np.tile(sig, (N_usrc, 1))
            pad = np.zeros((N_usrc, Nt), dtype=np.float32)
            pad[:, :sig.shape[1]] = sig[:, :Nt]
            if flags['u_mode'] == 'additive':
                pad *= (2.0 * c0_at_src * dt / dy)
            xs_uy = pad.T  # (Nt, N_usrc)

    # ---- Sensor indices ----
    sensor_flat_idx = np.array([0], dtype=np.int32)
    sensor_tri = np.zeros((1, 3), dtype=np.int32)
    sensor_bc  = np.ones((1, 3), dtype=np.float32) / 3.0
    N_sensors  = 1

    if flags['use_sensor'] and sensor is not None:
        mask_arr = np.asarray(sensor.mask)
        if flags['binary_sensor_mask']:
            sensor_flat_idx = np.flatnonzero(mask_arr).astype(np.int32)
            N_sensors = len(sensor_flat_idx)
        else:
            # Cartesian: (2, N_pts) array of [x; y] coordinates
            from scipy.spatial import Delaunay
            pts = mask_arr.T  # (N_pts, 2)
            xg = np.arange(Nx) * dx
            yg = np.arange(Ny) * dy
            grid_pts = np.column_stack([
                np.repeat(xg, Ny), np.tile(yg, Nx)
            ])
            tri_obj = Delaunay(grid_pts)
            simplex = tri_obj.find_simplex(pts)
            sensor_tri = tri_obj.simplices[simplex].astype(np.int32)
            b = tri_obj.transform[simplex, :2] @ (pts - tri_obj.transform[simplex, 2])[:,  :, None]
            bc3 = np.concatenate([b[:, :, 0], 1 - b[:, :, 0].sum(axis=1, keepdims=True)], axis=1)
            sensor_bc = bc3.astype(np.float32)
            N_sensors = len(pts)

    # ---- Time-reversal boundary data ----
    tr_bc_arr = np.zeros((int(kgrid.Nt), N_sensors), dtype=np.float32)
    if flags['time_rev'] and sensor is not None:
        tr = np.asarray(sensor.time_reversal_boundary_data, dtype=np.float32)
        tr_bc_arr[:tr.shape[1], :] = tr[:, :int(kgrid.Nt)].T
        tr_bc_arr = tr_bc_arr[::-1]  # reverse for time-reversal loop

    # ---- Initial pressure ----
    p0 = None
    if flags['source_p0']:
        p0_arr = np.asarray(source.p0, dtype=np.float32)
        if options.get('smooth', True) and isinstance(options.get('smooth'), bool):
            from pykwave.utils import smooth as _smooth
            p0_arr = _smooth(p0_arr)
        p0 = jnp.array(p0_arr)

    return {
        # PML
        'pml_x': pml_x, 'pml_x_sgx': pml_x_sgx,
        'pml_y': pml_y, 'pml_y_sgy': pml_y_sgy,
        # Derivative operators
        'ddx_k_shift_pos': ddx_k_shift_pos, 'ddx_k_shift_neg': ddx_k_shift_neg,
        'ddy_k_shift_pos': ddy_k_shift_pos, 'ddy_k_shift_neg': ddy_k_shift_neg,
        # k-space correction
        'kappa': kappa, 'source_kappa': source_kappa,
        # Medium
        'c0': c0_jax, 'rho0': jnp.array(rho0_arr),
        'rho0_sgx_inv': jnp.array(1.0 / rho0_sgx),
        'rho0_sgy_inv': jnp.array(1.0 / rho0_sgy),
        # Absorption
        'absorb_nabla1': absorb_nabla1, 'absorb_nabla2': absorb_nabla2,
        'absorb_tau': absorb_tau, 'absorb_eta': absorb_eta,
        # Nonlinear
        'BonA': jnp.array(np.broadcast_to(np.asarray(medium.BonA, dtype=np.float32), (Nx, Ny))) if medium.BonA is not None else None,
        # Non-staggered shift
        'x_shift_neg': x_shift_neg, 'y_shift_neg': y_shift_neg,
        # Source
        'p0': p0,
        'p_src_flat_idx': jnp.array(p_src_flat_idx),
        'u_src_flat_idx': jnp.array(u_src_flat_idx),
        'xs_p': jnp.array(xs_p),
        'xs_ux': jnp.array(xs_ux),
        'xs_uy': jnp.array(xs_uy),
        # Sensor
        'sensor_flat_idx': jnp.array(sensor_flat_idx),
        'sensor_tri': jnp.array(sensor_tri),
        'sensor_bc':  jnp.array(sensor_bc),
        'N_sensors': N_sensors,
        # Time reversal
        'tr_bc': jnp.array(tr_bc_arr),
        # Grid scalars
        'Nx': Nx, 'Ny': Ny, 'dt': dt, 'dx': dx, 'dy': dy,
        'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2,
        'record_start_index': int(sensor.record_start_index) - 1 if sensor else 0,  # 0-based
        'Nt': int(kgrid.Nt),
    }
