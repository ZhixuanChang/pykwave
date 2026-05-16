# pykwave/_scan_body.py
from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp


class SimState(NamedTuple):
    """Mutable simulation state carried through lax.scan."""
    p:        jax.Array   # (Nx, Ny)
    p_k:      jax.Array   # (Nx, Ny) complex64
    rhox:     jax.Array   # (Nx, Ny)
    rhoy:     jax.Array   # (Nx, Ny)
    ux_sgx:   jax.Array   # (Nx, Ny)
    uy_sgy:   jax.Array   # (Nx, Ny)
    duxdx:    jax.Array   # (Nx, Ny)
    duydy:    jax.Array   # (Nx, Ny)
    # Running stats at sensor locations — shape (N_sensors,)
    p_max_s:  jax.Array
    p_min_s:  jax.Array
    p_rms_sq: jax.Array
    ux_max_s: jax.Array
    uy_max_s: jax.Array
    ux_min_s: jax.Array
    uy_min_s: jax.Array
    ux_rms_sq: jax.Array
    uy_rms_sq: jax.Array
    # Running stats over all grid points — shape (Nx, Ny)
    p_max_all:  jax.Array
    p_min_all:  jax.Array
    ux_max_all: jax.Array
    uy_max_all: jax.Array
    ux_min_all: jax.Array
    uy_min_all: jax.Array
    n_recorded: jax.Array  # scalar int32


class StepOutput(NamedTuple):
    """Per-step outputs stacked by lax.scan into (Nt, N_sensors)."""
    p_at_sensor:     jax.Array   # (N_sensors,)
    ux_at_sensor:    jax.Array
    uy_at_sensor:    jax.Array
    ux_ns_at_sensor: jax.Array   # non-staggered
    uy_ns_at_sensor: jax.Array


def make_init_state(ops: dict) -> SimState:
    Nx, Ny, N_sensors = ops['Nx'], ops['Ny'], ops['N_sensors']
    z2d  = jnp.zeros((Nx, Ny), dtype=jnp.float32)
    z2dc = jnp.zeros((Nx, Ny), dtype=jnp.complex64)
    zs   = jnp.zeros((N_sensors,), dtype=jnp.float32)
    ninf = jnp.full((N_sensors,), -jnp.inf, dtype=jnp.float32)
    pinf = jnp.full((N_sensors,),  jnp.inf, dtype=jnp.float32)
    ninf2d = jnp.full((Nx, Ny), -jnp.inf, dtype=jnp.float32)
    pinf2d = jnp.full((Nx, Ny),  jnp.inf, dtype=jnp.float32)
    return SimState(
        p=z2d, p_k=z2dc, rhox=z2d, rhoy=z2d,
        ux_sgx=z2d, uy_sgy=z2d, duxdx=z2d, duydy=z2d,
        p_max_s=ninf, p_min_s=pinf, p_rms_sq=zs,
        ux_max_s=ninf, uy_max_s=ninf,
        ux_min_s=pinf, uy_min_s=pinf,
        ux_rms_sq=zs, uy_rms_sq=zs,
        p_max_all=ninf2d, p_min_all=pinf2d,
        ux_max_all=ninf2d, uy_max_all=ninf2d,
        ux_min_all=pinf2d, uy_min_all=pinf2d,
        n_recorded=jnp.array(0, dtype=jnp.int32),
    )


def scan_step(carry: SimState, xs: dict, *, ops: dict, flags: dict) -> tuple[SimState, StepOutput]:
    """One time-step of the k-space first-order 2D solver."""
    t_index = xs['t_index']
    p_k = carry.p_k

    # ------------------------------------------------------------------ #
    # Time reversal: enforce Dirichlet BC on pressure before velocity step
    # ------------------------------------------------------------------ #
    if flags['time_rev']:
        p_bc = xs['tr_bc']                              # (N_sensors,)
        tr_grid = jnp.zeros((ops['Nx'], ops['Ny']), dtype=jnp.float32)
        tr_grid = tr_grid.at[ops['sensor_flat_idx']].set(p_bc)
        p_k = jnp.fft.fft2(tr_grid).astype(jnp.complex64)

    # ------------------------------------------------------------------ #
    # Step 1: update velocity (momentum equation)
    # ------------------------------------------------------------------ #
    ux_sgx = ops['pml_x_sgx'] * (
        ops['pml_x_sgx'] * carry.ux_sgx
        - ops['dt'] * ops['rho0_sgx_inv'] * jnp.real(
            jnp.fft.ifft2(ops['ddx_k_shift_pos'] * ops['kappa'] * p_k)
        )
    )
    uy_sgy = ops['pml_y_sgy'] * (
        ops['pml_y_sgy'] * carry.uy_sgy
        - ops['dt'] * ops['rho0_sgy_inv'] * jnp.real(
            jnp.fft.ifft2(ops['ddy_k_shift_pos'] * ops['kappa'] * p_k)
        )
    )

    # ------------------------------------------------------------------ #
    # Step 2: velocity source injection
    # ------------------------------------------------------------------ #
    if flags['source_ux']:
        ux_sig = xs['ux_src']                          # (N_u_src,)
        if flags['u_mode'] == 'dirichlet':
            ux_sgx = ux_sgx.at[ops['u_src_flat_idx']].set(ux_sig)
        else:
            src_mat = jnp.zeros((ops['Nx'], ops['Ny']), dtype=jnp.float32)
            src_mat = src_mat.at[ops['u_src_flat_idx']].add(ux_sig)
            src_mat = jnp.real(jnp.fft.ifft2(ops['source_kappa'] * jnp.fft.fft2(src_mat)))
            ux_sgx = ux_sgx + src_mat

    if flags['source_uy']:
        uy_sig = xs['uy_src']
        if flags['u_mode'] == 'dirichlet':
            uy_sgy = uy_sgy.at[ops['u_src_flat_idx']].set(uy_sig)
        else:
            src_mat = jnp.zeros((ops['Nx'], ops['Ny']), dtype=jnp.float32)
            src_mat = src_mat.at[ops['u_src_flat_idx']].add(uy_sig)
            src_mat = jnp.real(jnp.fft.ifft2(ops['source_kappa'] * jnp.fft.fft2(src_mat)))
            uy_sgy = uy_sgy + src_mat

    # ------------------------------------------------------------------ #
    # Step 3: spectral divergence
    # ------------------------------------------------------------------ #
    duxdx = jnp.real(jnp.fft.ifft2(ops['ddx_k_shift_neg'] * ops['kappa'] * jnp.fft.fft2(ux_sgx)))
    duydy = jnp.real(jnp.fft.ifft2(ops['ddy_k_shift_neg'] * ops['kappa'] * jnp.fft.fft2(uy_sgy)))

    # ------------------------------------------------------------------ #
    # Step 4: continuity equation → update density
    # ------------------------------------------------------------------ #
    if flags['nonlinear']:
        rho_total = 2.0 * (carry.rhox + carry.rhoy) + ops['rho0']
        rhox = ops['pml_x'] * (ops['pml_x'] * carry.rhox - ops['dt'] * rho_total * duxdx)
        rhoy = ops['pml_y'] * (ops['pml_y'] * carry.rhoy - ops['dt'] * rho_total * duydy)
    else:
        rhox = ops['pml_x'] * (ops['pml_x'] * carry.rhox - ops['dt'] * ops['rho0'] * duxdx)
        rhoy = ops['pml_y'] * (ops['pml_y'] * carry.rhoy - ops['dt'] * ops['rho0'] * duydy)

    # ------------------------------------------------------------------ #
    # Step 5: pressure source injection
    # ------------------------------------------------------------------ #
    if flags['source_p']:
        p_sig = xs['p_src']                            # (N_p_src,)
        if flags['p_mode'] == 'dirichlet':
            rhox = rhox.at[ops['p_src_flat_idx']].set(p_sig)
            rhoy = rhoy.at[ops['p_src_flat_idx']].set(p_sig)
        else:
            src_mat = jnp.zeros((ops['Nx'], ops['Ny']), dtype=jnp.float32)
            src_mat = src_mat.at[ops['p_src_flat_idx']].add(p_sig)
            src_mat = jnp.real(jnp.fft.ifft2(ops['source_kappa'] * jnp.fft.fft2(src_mat)))
            rhox = rhox + src_mat
            rhoy = rhoy + src_mat

    # ------------------------------------------------------------------ #
    # Step 6: equation of state → pressure
    # ------------------------------------------------------------------ #
    rho_sum = rhox + rhoy
    c0sq = ops['c0'] ** 2

    if flags['eos'] == 'lossless':
        p = c0sq * rho_sum
    elif flags['eos'] == 'stokes':
        p = c0sq * (rho_sum + ops['absorb_tau'] * ops['rho0'] * (duxdx + duydy))
    else:  # absorbing
        p = c0sq * (
            rho_sum
            + ops['absorb_tau'] * jnp.real(jnp.fft.ifft2(
                ops['absorb_nabla1'] * jnp.fft.fft2(ops['rho0'] * (duxdx + duydy))
            ))
            - ops['absorb_eta'] * jnp.real(jnp.fft.ifft2(
                ops['absorb_nabla2'] * jnp.fft.fft2(rho_sum)
            ))
        )

    if flags['nonlinear']:
        p = p + c0sq * ops['BonA'] * rho_sum ** 2 / (2.0 * ops['rho0'])

    # ------------------------------------------------------------------ #
    # Step 7: enforce p0 initial condition at t=0
    # ------------------------------------------------------------------ #
    if flags['source_p0']:
        p0 = ops['p0']
        is_first = (t_index == 0)
        p0_k = jnp.fft.fft2(p0).astype(jnp.complex64)
        ux_p0 = (ops['dt'] * ops['rho0_sgx_inv']
                 * jnp.real(jnp.fft.ifft2(ops['ddx_k_shift_pos'] * ops['kappa'] * p0_k)) / 2.0)
        uy_p0 = (ops['dt'] * ops['rho0_sgy_inv']
                 * jnp.real(jnp.fft.ifft2(ops['ddy_k_shift_pos'] * ops['kappa'] * p0_k)) / 2.0)
        p      = jnp.where(is_first, p0,              p)
        rhox   = jnp.where(is_first, p0 / (2.0 * c0sq), rhox)
        rhoy   = jnp.where(is_first, p0 / (2.0 * c0sq), rhoy)
        ux_sgx = jnp.where(is_first, ux_p0,           ux_sgx)
        uy_sgy = jnp.where(is_first, uy_p0,           uy_sgy)

    # ------------------------------------------------------------------ #
    # Step 8: update p_k
    # ------------------------------------------------------------------ #
    p_k_new = jnp.fft.fft2(p).astype(jnp.complex64)

    # ------------------------------------------------------------------ #
    # Step 9: sensor extraction
    # ------------------------------------------------------------------ #
    active = (t_index >= ops['record_start_index']).astype(jnp.float32)

    # Non-staggered velocity (needed for u_non_staggered / I)
    if flags['record_u_non_staggered'] or flags['record_I'] or flags['record_I_avg']:
        ux_ns = jnp.real(jnp.fft.ifft(
            ops['x_shift_neg'] * jnp.fft.fft(ux_sgx, axis=0), axis=0))
        uy_ns = jnp.real(jnp.fft.ifft(
            ops['y_shift_neg'] * jnp.fft.fft(uy_sgy, axis=1), axis=1))
    else:
        ux_ns = ux_sgx
        uy_ns = uy_sgy

    if flags['binary_sensor_mask']:
        idx = ops['sensor_flat_idx']
        p_s  = p.ravel()[idx]   * active
        ux_s = ux_sgx.ravel()[idx] * active
        uy_s = uy_sgy.ravel()[idx] * active
        ux_ns_s = ux_ns.ravel()[idx] * active
        uy_ns_s = uy_ns.ravel()[idx] * active
    else:
        tri, bc = ops['sensor_tri'], ops['sensor_bc']
        p_s  = jnp.sum(p.ravel()[tri]     * bc, axis=-1) * active
        ux_s = jnp.sum(ux_sgx.ravel()[tri] * bc, axis=-1) * active
        uy_s = jnp.sum(uy_sgy.ravel()[tri] * bc, axis=-1) * active
        ux_ns_s = jnp.sum(ux_ns.ravel()[tri] * bc, axis=-1) * active
        uy_ns_s = jnp.sum(uy_ns.ravel()[tri] * bc, axis=-1) * active

    # Update running stats
    n_rec = carry.n_recorded + active.astype(jnp.int32)

    if flags['record_p_max']:
        p_max_s = jnp.maximum(carry.p_max_s, p_s)
    else:
        p_max_s = carry.p_max_s

    if flags['record_p_min']:
        p_min_s = jnp.minimum(carry.p_min_s, p_s)
    else:
        p_min_s = carry.p_min_s

    if flags['record_p_rms']:
        p_rms_sq = carry.p_rms_sq + p_s ** 2
    else:
        p_rms_sq = carry.p_rms_sq

    if flags['record_u_max']:
        ux_max_s = jnp.maximum(carry.ux_max_s, ux_s)
        uy_max_s = jnp.maximum(carry.uy_max_s, uy_s)
    else:
        ux_max_s, uy_max_s = carry.ux_max_s, carry.uy_max_s

    if flags['record_u_min']:
        ux_min_s = jnp.minimum(carry.ux_min_s, ux_s)
        uy_min_s = jnp.minimum(carry.uy_min_s, uy_s)
    else:
        ux_min_s, uy_min_s = carry.ux_min_s, carry.uy_min_s

    if flags['record_u_rms']:
        ux_rms_sq = carry.ux_rms_sq + ux_s ** 2
        uy_rms_sq = carry.uy_rms_sq + uy_s ** 2
    else:
        ux_rms_sq, uy_rms_sq = carry.ux_rms_sq, carry.uy_rms_sq

    # All-grid running stats
    if flags['record_p_max_all']:
        p_max_all = jnp.maximum(carry.p_max_all, p)
    else:
        p_max_all = carry.p_max_all

    if flags['record_p_min_all']:
        p_min_all = jnp.minimum(carry.p_min_all, p)
    else:
        p_min_all = carry.p_min_all

    if flags['record_u_max_all']:
        ux_max_all = jnp.maximum(carry.ux_max_all, ux_sgx)
        uy_max_all = jnp.maximum(carry.uy_max_all, uy_sgy)
    else:
        ux_max_all, uy_max_all = carry.ux_max_all, carry.uy_max_all

    if flags['record_u_min_all']:
        ux_min_all = jnp.minimum(carry.ux_min_all, ux_sgx)
        uy_min_all = jnp.minimum(carry.uy_min_all, uy_sgy)
    else:
        ux_min_all, uy_min_all = carry.ux_min_all, carry.uy_min_all

    new_carry = SimState(
        p=p, p_k=p_k_new, rhox=rhox, rhoy=rhoy,
        ux_sgx=ux_sgx, uy_sgy=uy_sgy, duxdx=duxdx, duydy=duydy,
        p_max_s=p_max_s, p_min_s=p_min_s, p_rms_sq=p_rms_sq,
        ux_max_s=ux_max_s, uy_max_s=uy_max_s,
        ux_min_s=ux_min_s, uy_min_s=uy_min_s,
        ux_rms_sq=ux_rms_sq, uy_rms_sq=uy_rms_sq,
        p_max_all=p_max_all, p_min_all=p_min_all,
        ux_max_all=ux_max_all, uy_max_all=uy_max_all,
        ux_min_all=ux_min_all, uy_min_all=uy_min_all,
        n_recorded=n_rec,
    )
    out = StepOutput(
        p_at_sensor=p_s, ux_at_sensor=ux_s, uy_at_sensor=uy_s,
        ux_ns_at_sensor=ux_ns_s, uy_ns_at_sensor=uy_ns_s,
    )
    return new_carry, out
