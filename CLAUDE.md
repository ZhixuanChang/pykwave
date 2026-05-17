# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable)
pip install -e .

# Run all tests
/home/chang/.conda/envs/ml/bin/python -m pytest tests/ -v

# Run a single test
/home/chang/.conda/envs/ml/bin/python -m pytest tests/test_kspace_first_order_2d.py::test_point_source_wave_speed -v

# Run without slow integration tests
/home/chang/.conda/envs/ml/bin/python -m pytest tests/ -v --ignore=tests/test_kspace_first_order_2d.py
```

**Python interpreter:** always use `/home/chang/.conda/envs/ml/bin/python` (JAX 0.10.0).  
**k-Wave MATLAB reference:** `/home/chang/k-Wave/` — consult for numerical parity questions.  
**Expected test result:** 29 passed, 2 skipped (parity tests skip when MATLAB `.npz` fixtures are absent in `tests/fixtures/`).

## Architecture

Execution has three phases:

### 1. Precompute (`_precompute.py`)
`build_flags_and_ops()` runs once before the time loop. It returns:
- **`flags` dict** — Python booleans resolved from input configuration (EOS type, which sensor fields to record, source types, etc.). These drive `if flags[...]` branches inside the scan body that get resolved at JAX trace time, not at runtime.
- **`ops` dict** — JAX arrays (operators, PML multipliers, pre-scaled source signals, sensor indices, etc.) captured by `functools.partial` before `lax.scan`.

### 2. Time loop (`_scan_body.py`)
`scan_step(carry, xs, *, ops, flags)` is a pure function executed `Nt` times by `jax.lax.scan`. It takes `SimState` (a 24-field NamedTuple) as carry and returns an updated carry plus `StepOutput` (5 per-step sensor values stacked into `(Nt, N_sensors)` by scan).

The per-step pipeline: update velocity → inject velocity source → spectral divergence → update density → inject pressure source → equation of state → enforce p0 IC → update p_k → gather sensor data.

### 3. Post-process (`_sensor_ops.py`)
`build_sensor_data()` converts scan outputs and final carry into a `SensorData` dataclass. Running accumulators (RMS, max/min) live in the carry to avoid O(Nx·Ny·Nt) stacked arrays; they are finalized here.

### Orchestration (`kspace_first_order_2d.py`)
Calls `build_flags_and_ops` → `make_init_state` → builds `xs_stacked` dict → `partial(scan_step, ops=ops, flags=flags)` → `jax.jit(jax.lax.scan)` → `build_sensor_data`. Validation runs eagerly before JAX tracing.

## Critical Numerical Conventions

**Wavenumber storage order:** `kgrid.kx_vec` / `ky_vec` are stored in **fftshift order** (sorted, DC at center). All spectral operators in `_precompute.py` apply `np.fft.ifftshift()` to convert to FFT output order before storing as JAX arrays. Never skip this conversion.

**Stagger-shift operators:**
```python
ddx_k_shift_pos = ifftshift(1j * kx * exp(+1j * kx * dx/2))   # forward half-step
ddx_k_shift_neg = ifftshift(1j * kx * exp(-1j * kx * dx/2))   # backward half-step
```

**k-space sinc:** `np.sinc(c_ref * k * dt / 2 / np.pi)` — NumPy's normalized sinc divided by π gives the correct MATLAB-matching formula.

**Source pre-scaling (verified against MATLAB `kspaceFirstOrder_scaleSourceTerms.m`):**
- Pressure additive: `sig *= 2*dt / (2 * c0 * dx)` = `dt/(c0*dx)` for N=2 components
- Pressure Dirichlet: `sig /= 2 * c0²`
- Velocity additive: `sig *= 2 * c0 * dt / dx`

**Absorbing EOS sign (verified against MATLAB line 866):**
```python
p = c0sq * (rho_sum
    + absorb_tau * real(ifft2(absorb_nabla1 * fft2(rho0*(duxdx+duydy))))
    - absorb_eta * real(ifft2(absorb_nabla2 * fft2(rho_sum))))   # minus sign is correct
```

**RMS accumulation:** `p_rms_sq` accumulates `p_s**2` where `p_s` already has `* active` applied (zeroed before `record_start_index`). Don't multiply by `active` again.

## Known Limitations (do not accidentally "fix")

- **Absorbing EOS stability:** `alpha_coeff=0.5, alpha_power=1.5` (realistic tissue) causes NaN on 1mm grids. Integration tests use `alpha_coeff=1e-10` intentionally. This is a real numerical instability, not a bug.
- **`sensor.directivity_angle`** is stored in the `Sensor` dataclass but not yet applied in `_sensor_ops.py`.
- **Source scaling for heterogeneous media** uses `c0` at the first source grid point only — a known approximation.

## License

LGPLv3. Based on k-Wave MATLAB Toolbox by B.E. Treeby & B.T. Cox. Every source file must carry the 3-line SPDX header:
```python
# pykwave — JAX implementation of k-Wave pseudospectral acoustic simulation.
# Based on the k-Wave MATLAB Toolbox (http://www.k-wave.org) by B.E. Treeby & B.T. Cox.
# SPDX-License-Identifier: LGPL-3.0-or-later
```
