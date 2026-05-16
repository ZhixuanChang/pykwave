# pykwave Design Spec — kspaceFirstOrder2D JAX Implementation

**Date:** 2026-05-17  
**Author:** Claude Code (brainstorming session)  
**Status:** Approved by user

---

## 1. Overview

pykwave is a JAX implementation of the k-Wave acoustic simulation toolbox, beginning
with `kspaceFirstOrder2D`. The function simulates time-domain 2D compressional wave
propagation in homogeneous or heterogeneous acoustic media using a first-order
pseudospectral k-space method with a staggered leap-frog time-stepping scheme and
a perfectly matched layer (PML) absorbing boundary.

**Scope of first implementation:** full feature parity with MATLAB k-Wave v1.3
`kspaceFirstOrder2D`, including:
- Linear and nonlinear (BonA) equations of state
- Lossless, power-law absorbing, and Stokes absorbing equations of state
- Initial pressure source (`p0`) and time-varying pressure/velocity sources
- Binary and Cartesian sensor masks
- All `sensor.record` fields: `p`, `p_max`, `p_min`, `p_rms`, `p_final`,
  `p_max_all`, `p_min_all`, `u`, `u_max`, `u_min`, `u_rms`, `u_final`,
  `u_max_all`, `u_min_all`, `u_non_staggered`, `I`, `I_avg`
- Time reversal image reconstruction
- Directional sensor response
- Frequency-domain Gaussian sensor filter

**Out of scope:** visualisation (PlotSim, RecordMovie), HDF5 disk I/O (SaveToDisk),
C++/GPU backend interop, `kspaceFirstOrder1D/3D`, elastic media solvers.

**Reference implementation:** `/home/chang/k-Wave/kspaceFirstOrder2D.m` and its
private subscripts in `/home/chang/k-Wave/private/`.

---

## 2. Technology Stack

| Item | Choice |
|---|---|
| Language | Python 3.10+ |
| Accelerator framework | JAX 0.10.0 |
| Default dtype | float32 (configurable via `data_cast` kwarg) |
| Time loop | `jax.lax.scan` + `@jax.jit` |
| Python interpreter | `/home/chang/.conda/envs/ml/bin/python` |
| Pre-scan interpolation | `scipy.ndimage` / `scipy.spatial` (NumPy-land only) |

---

## 3. Public API

```python
from pykwave import kspace_first_order_2d, KWaveGrid, Medium, Source, Sensor

kgrid = KWaveGrid(Nx, dx, Ny, dy)
kgrid.make_time(sound_speed, cfl=0.3, t_end=None)

sensor_data = kspace_first_order_2d(
    kgrid, medium, source, sensor,
    # optional kwargs (matching MATLAB names, snake_case):
    pml_size=20,          # int or (int, int) for x and y independently
    pml_alpha=2.0,        # float or (float, float)
    pml_inside=True,
    smooth=True,          # bool or (bool, bool, bool) for p0/c0/rho0
    cart_interp='linear', # 'linear' | 'nearest'
    data_cast='float32',  # dtype string
)
```

If `sensor.record` is not set, `sensor_data` is a plain JAX array of shape
`(N_sensors, Nt_record)` (pressure time series). Otherwise `sensor_data` is a
`SensorData` dataclass with named fields.

---

## 4. Project Layout

```
pykwave/
  __init__.py                   # re-exports public symbols
  grid.py                       # KWaveGrid class
  medium.py                     # Medium dataclass
  source.py                     # Source dataclass
  sensor.py                     # Sensor dataclass + SensorData dataclass
  pml.py                        # get_pml()
  utils.py                      # db2neper, smooth, scale_si, interp2d helpers
  kspace_first_order_2d.py      # public entry point; orchestrates precompute + scan
  _precompute.py                # builds operators, PML, absorption vars, sensor indices
  _scan_body.py                 # pure lax.scan body (the per-step kernel)
  _sensor_ops.py                # sensor gather/accumulate helpers called from scan body
tests/
  fixtures/                     # .npz reference outputs generated from MATLAB k-Wave
  test_grid.py
  test_pml.py
  test_utils.py
  test_kspace_first_order_2d.py
  test_parity.py
docs/
  superpowers/specs/
    2026-05-17-pykwave-design.md  (this file)
```

---

## 5. Core Data Structures

### 5.1 KWaveGrid

Regular Python class (not frozen — `dt`/`Nt`/`t_array` are mutable).

```python
kgrid = KWaveGrid(Nx, dx, Ny, dy)
```

**Computed on construction (JAX arrays):**
- `x_vec` (Nx,), `y_vec` (Ny,)
- `x` (Nx, Ny), `y` (Nx, Ny) — meshgrids
- `kx_vec` (Nx,), `ky_vec` (Ny,) — wavenumber vectors
- `k` (Nx, Ny) — wavenumber magnitude `sqrt(kx² + ky²)`
- `k_max`, `dim`, `total_grid_points`

**Mutable time properties:**
- `Nt`, `dt`, `t_array`
- `make_time(sound_speed, cfl=0.3, t_end=None)` — CFL-based automatic time stepping
- `set_time(Nt, dt)` — manual override

Wavenumber vectors follow the MATLAB convention:
`kx_vec = (2π/Nx*dx) * ifftshift(0, 1, …, Nx/2-1, -Nx/2, …, -1)` (for even Nx).

### 5.2 Medium

```python
@dataclass
class Medium:
    sound_speed: ArrayLike           # scalar or (Nx, Ny) [m/s]
    density:     ArrayLike = None    # scalar or (Nx, Ny) [kg/m³]; required for velocity I/O
    sound_speed_ref: float = None    # overrides auto-computed reference speed
    BonA:        ArrayLike = None    # nonlinearity; presence triggers nonlinear EOS
    alpha_coeff: float     = None    # [dB/(MHz^y cm)]; presence (with alpha_power) triggers absorbing EOS
    alpha_power: float     = None
    alpha_mode:  str       = None    # 'no_absorption' | 'no_dispersion'
    alpha_filter: ArrayLike = None   # (Nx, Ny) frequency-domain filter
    alpha_sign:  tuple     = None    # (sign_absorption, sign_dispersion)
```

Equation of state selection (resolved before scan):
- No `alpha_coeff`/`alpha_power` → **lossless**
- `alpha_power == 2` → **Stokes** (simplified absorbing)
- Otherwise → **absorbing** (full fractional Laplacian)
- `BonA` present → additionally **nonlinear**

### 5.3 Source

```python
@dataclass
class Source:
    p0:     ArrayLike = None   # (Nx, Ny) initial pressure
    p:      ArrayLike = None   # (n_signals, Nt) or (1, Nt) time-varying pressure
    p_mask: ArrayLike = None   # (Nx, Ny) binary mask for p
    p_mode: str       = 'additive'   # 'additive' | 'dirichlet'
    ux:     ArrayLike = None   # (n_signals, Nt) x-velocity source
    uy:     ArrayLike = None   # (n_signals, Nt) y-velocity source
    u_mask: ArrayLike = None   # (Nx, Ny) binary mask for u
    u_mode: str       = 'additive'
```

### 5.4 Sensor

```python
@dataclass
class Sensor:
    mask: ArrayLike                          # binary (Nx, Ny) or Cartesian (2, N_pts)
    record: list[str]            = None      # subset of known field names; None → ['p']
    record_start_index: int      = 1
    time_reversal_boundary_data: ArrayLike = None  # (N_sensors, Nt)
    frequency_response: tuple    = None     # (center_hz, bandwidth_pct)
    directivity_angle:  ArrayLike = None    # (N_sensors,) [rad]
    directivity_size:   float    = None     # [m]
    directivity_pattern: str     = None     # 'pressure' | 'gradient'
```

### 5.5 SensorData

```python
@dataclass
class SensorData:
    p:               Array | None   # (N_sensors, Nt_record)
    p_max:           Array | None   # (N_sensors,)
    p_min:           Array | None
    p_rms:           Array | None
    p_final:         Array | None   # (Nx_inner, Ny_inner)
    p_max_all:       Array | None   # (Nx_inner, Ny_inner)
    p_min_all:       Array | None
    ux:              Array | None   # (N_sensors, Nt_record)
    uy:              Array | None
    ux_max:          Array | None   # (N_sensors,)
    uy_max:          Array | None
    ux_min:          Array | None
    uy_min:          Array | None
    ux_rms:          Array | None
    uy_rms:          Array | None
    ux_final:        Array | None   # (Nx_inner, Ny_inner)
    uy_final:        Array | None
    ux_max_all:      Array | None   # (Nx_inner, Ny_inner)
    uy_max_all:      Array | None
    ux_min_all:      Array | None
    uy_min_all:      Array | None
    ux_non_staggered: Array | None  # (N_sensors, Nt_record)
    uy_non_staggered: Array | None
    Ix:              Array | None   # (N_sensors, Nt_record)
    Iy:              Array | None
    Ix_avg:          Array | None   # (N_sensors,)
    Iy_avg:          Array | None
```

---

## 6. Numerical Algorithm

### 6.1 Precomputed Operators (outside scan)

All of the following are computed once in `_precompute.py` before `lax.scan` is
called. They are passed as static `operators` dict captured by `functools.partial`.

```
# PML — shaped for broadcasting (axis 0 → (Nx,1), axis 1 → (1,Ny))
pml_x,     pml_x_sgx  = get_pml(..., axis=0, staggered=False/True)
pml_y,     pml_y_sgy  = get_pml(..., axis=1, staggered=False/True)

# Spectral derivative + stagger-shift operators
ddx_k_shift_pos = ifftshift(1j * kx_vec * exp( 1j * kx_vec * dx/2))  # (Nx, 1)
ddx_k_shift_neg = ifftshift(1j * kx_vec * exp(-1j * kx_vec * dx/2))
ddy_k_shift_pos = ifftshift(1j * ky_vec * exp( 1j * ky_vec * dy/2))  # (1, Ny)
ddy_k_shift_neg = ifftshift(1j * ky_vec * exp(-1j * ky_vec * dy/2))

# k-space correction operators
kappa        = ifftshift(sinc(c_ref * k * dt / 2))        # (Nx, Ny)
source_kappa = ifftshift(cos( c_ref * k * dt / 2))

# Absorption operators (absorbing EOS only)
absorb_nabla1 = ifftshift(k^(alpha_power - 2))            # (Nx, Ny); inf→0
absorb_nabla2 = ifftshift(k^(alpha_power - 1))
absorb_tau    = -2 * alpha_coeff_nepers * c0^(alpha_power - 1)
absorb_eta    =  2 * alpha_coeff_nepers * c0^alpha_power * tan(π*alpha_power/2)

# Staggered-grid density (via scipy.ndimage before entering JAX)
rho0_sgx_inv = 1 / interp_rho_shifted(rho0, +dx/2, axis=0)  # (Nx, Ny)
rho0_sgy_inv = 1 / interp_rho_shifted(rho0, +dy/2, axis=1)

# Non-staggered velocity shift vectors (for u_non_staggered / I)
x_shift_neg = ifftshift(exp(-1j * kx_vec * dx/2))         # (Nx, 1)
y_shift_neg = ifftshift(exp(-1j * ky_vec * dy/2))         # (1, Ny)

# Reference sound speed (max of sound_speed unless overridden)
c_ref = max(sound_speed)  or  medium.sound_speed_ref
```

### 6.2 Per-Step Kernel (lax.scan body)

Executed `Nt` times. All branches resolved at trace time; no Python conditionals on
array values inside this function.

```
Step 1 — Update particle velocity (momentum equation):
  ux_sgx = pml_x_sgx * (pml_x_sgx * ux_sgx
           - dt * rho0_sgx_inv * real(ifft2(ddx_k_shift_pos * kappa * p_k)))
  uy_sgy = pml_y_sgy * (pml_y_sgy * uy_sgy
           - dt * rho0_sgy_inv * real(ifft2(ddy_k_shift_pos * kappa * p_k)))

Step 2 — Inject velocity source:
  dirichlet: ux_sgx = ux_sgx.at[u_src_idx].set(ux_sig[t])
  additive:  ux_sgx += real(ifft2(source_kappa * fft2(scatter(ux_sig[t], u_src_idx))))

Step 3 — Spectral divergence:
  duxdx = real(ifft2(ddx_k_shift_neg * kappa * fft2(ux_sgx)))
  duydy = real(ifft2(ddy_k_shift_neg * kappa * fft2(uy_sgy)))

Step 4 — Update density (continuity equation):
  linear:    rhox = pml_x * (pml_x * rhox - dt * rho0 * duxdx)
  nonlinear: rhox = pml_x * (pml_x * rhox - dt * (2*(rhox+rhoy) + rho0) * duxdx)
  (same for rhoy with pml_y and duydy)

Step 5 — Inject pressure source:
  dirichlet: rhox = rhox.at[p_src_idx].set(p_sig[t])
             rhoy = rhoy.at[p_src_idx].set(p_sig[t])
  additive:  rhox += real(ifft2(source_kappa * fft2(scatter(p_sig[t], p_src_idx))))

Step 6 — Equation of state → pressure:
  lossless:  p = c0² * (rhox + rhoy)
  absorbing: p = c0² * ((rhox+rhoy)
                 + absorb_tau * real(ifft2(absorb_nabla1 * fft2(rho0*(duxdx+duydy))))
                 - absorb_eta * real(ifft2(absorb_nabla2 * fft2(rhox+rhoy))))
  Stokes:    p = c0² * ((rhox+rhoy) + absorb_tau * rho0 * (duxdx+duydy))
  nonlinear adds: + BonA * (rhox+rhoy)² / (2*rho0)

Step 7 — Enforce p0 initial condition (t_index == 0):
  p    = jnp.where(is_first_step, p0, p)
  rhox = jnp.where(is_first_step, p0 / (2*c0²), rhox)
  rhoy = jnp.where(is_first_step, p0 / (2*c0²), rhoy)
  ux_sgx = jnp.where(is_first_step,
               dt * rho0_sgx_inv * real(ifft2(ddx_k_shift_pos * kappa * fft2(p0))) / 2,
               ux_sgx)
  uy_sgy = jnp.where(is_first_step, ..., uy_sgy)

Step 8 — Update p_k:
  p_k = fft2(p)

Step 9 — Extract sensor data (see Section 7).
```

### 6.3 Time Reversal

Before calling `lax.scan`:
- Source signals `xs['p_source']` are reversed along the time axis.
- Loop runs for `Nt - 1` steps (stop one step before end, matching MATLAB).
- At step 0, `p` is enforced as a Dirichlet BC from
  `sensor.time_reversal_boundary_data` (reversed).
- On return, `sensor_data.p_final` is the reconstructed initial pressure.

---

## 7. Time Loop Architecture

### 7.1 lax.scan Inputs

```
xs (per-step arrays, shape (Nt, ...)):
  p_source:   (Nt, N_p_sources)   pre-indexed pressure source signals; 0 outside window
  ux_source:  (Nt, N_u_sources)
  uy_source:  (Nt, N_u_sources)
  t_index:    (Nt,) int32          0-based step counter for p0 initial-condition guard
  tr_bc:      (Nt, N_sensors)      time-reversal boundary data (zeros if not used)
```

Pre-materialising source signals into `xs` avoids dynamic indexing inside the
compiled kernel. `source.p0` is part of the static `operators` dict (not `xs`)
because it is a full `(Nx, Ny)` array used only at `t_index == 0`.

### 7.2 Carry (SimState)

```
p, p_k, rhox, rhoy, ux_sgx, uy_sgy, duxdx, duydy  — field arrays (Nx, Ny)

# Running accumulators (O(Nx·Ny) each, avoids stacking Nt frames):
p_max_all, p_min_all        — (Nx, Ny)
ux_max_all, uy_max_all      — (Nx, Ny)
ux_min_all, uy_min_all      — (Nx, Ny)
p_rms_sq_sum                — (N_sensors,)  running sum of p²
ux_rms_sq_sum, uy_rms_sq_sum
n_recorded                  — scalar int32, number of steps recorded so far
```

### 7.3 Per-Step Output (StepOutput)

Stacked by `lax.scan` into shape `(Nt, ...)`:

```
p_at_sensor:      (N_sensors,)   — zeroed for t < record_start_index
ux_at_sensor:     (N_sensors,)
uy_at_sensor:     (N_sensors,)
ux_ns_at_sensor:  (N_sensors,)   — non-staggered; present only if requested
uy_ns_at_sensor:  (N_sensors,)
```

After scan, these are transposed to `(N_sensors, Nt)` and sliced to `Nt_record` steps.

### 7.4 Postprocessing (after scan)

The following are computed outside JAX or as simple JAX array ops, not inside scan:

- `p_rms = sqrt(p_rms_sq_sum / n_recorded)` — from carried accumulators
- `Ix`, `Iy`, `Ix_avg`, `Iy_avg` — from stacked `p` and `u_non_staggered`
- Directional response — applied to stacked `p_at_sensor`
- Gaussian frequency filter — `sensor.frequency_response`
- `p_final`, `ux_final`, `uy_final` — sliced from `final_carry` field arrays
- `p_max_all`, `p_min_all`, etc. — taken from `final_carry` accumulators

---

## 8. Sensor Data Collection

### 8.1 Binary Sensor Mask

Precomputed flat index array `sensor_flat_idx: (N_sensors,) int32`.

Inside scan body (pure gather, XLA-friendly):
```python
p_at_sensor = p.ravel()[sensor_flat_idx] * active_mask(t_index)
```

`active_mask(t)` is `jnp.where(t >= record_start_index - 1, 1.0, 0.0)`.

### 8.2 Cartesian Sensor Mask

Resolved to a triangulation in Python before the scan (using `scipy.spatial.Delaunay`),
producing `tri: (N_sensors, 3) int32` and `bc: (N_sensors, 3) float32`.

Inside scan body:
```python
p_at_sensor = jnp.sum(p.ravel()[tri] * bc, axis=-1) * active_mask(t_index)
```

`'nearest'` interpolation precomputes a single nearest-neighbour index per sensor
point, reducing to the binary mask case.

### 8.3 Non-Staggered Velocity

Computed inside scan body when `'u_non_staggered'`, `'I'`, or `'I_avg'` is in
`sensor.record`:
```python
ux_ns = jnp.real(jnp.fft.ifft(x_shift_neg * jnp.fft.fft(ux_sgx, axis=0), axis=0))
uy_ns = jnp.real(jnp.fft.ifft(y_shift_neg * jnp.fft.fft(uy_sgy, axis=1), axis=1))
```

### 8.4 Directional Response

Applied in postprocessing (outside scan) using a per-sensor directivity weight
matrix precomputed from `sensor.directivity_angle`, `sensor.directivity_size`,
and `sensor.directivity_pattern` over the `kx`/`ky` grid.

---

## 9. PML Implementation

`get_pml(N, d, dt, c, pml_size, pml_alpha, staggered, axis)` returns a 1D array
of shape `(N,)` then reshaped to `(N, 1)` for axis=0 or `(1, N)` for axis=1.

Profile (quartic, matching MATLAB):
```
pml_left[i]  = pml_alpha * (c/d) * ((i + 0.5*staggered - pml_size - 1) / (-pml_size))^4
pml_right[i] = pml_alpha * (c/d) * ((i + 0.5*staggered) / pml_size)^4
pml[i]       = exp(-profile[i] * dt / 2)
```

`pml_inside=False` causes the grid to be expanded by `pml_size` in each direction
before the simulation, then the PML region is stripped from outputs. This expansion
is done in precomputation by padding all input arrays.

---

## 10. Validation and Error Handling

All validation is eager Python, before JAX tracing:

| Check | Error |
|---|---|
| `kgrid.dim != 2` | `ValueError` |
| `kgrid.t_array` not set | `ValueError` |
| `source.p` given without `source.p_mask` | `ValueError` |
| `alpha_coeff` given without `alpha_power` or vice versa | `ValueError` |
| `sensor.record` contains unknown field name | `ValueError` |
| `sensor.mask` not binary or 2×N float | `ValueError` |
| `medium.sound_speed` / `medium.density` wrong shape | `ValueError` |

---

## 11. Testing

### 11.1 Unit Tests

- `test_grid.py` — `KWaveGrid` construction, wavenumber vectors, `make_time` CFL
- `test_pml.py` — PML profile shape, boundary values, staggered vs non-staggered
- `test_utils.py` — `db2neper`, `smooth`, `scale_si`

### 11.2 Algorithm Regression Test

Homogeneous lossless medium, point source at grid centre, binary sensor ring mask.
At `t_end`, compare simulated pressure wave-front radius and peak amplitude against
the analytical circular wave solution. Target: `< 1%` relative error in peak pressure.

### 11.3 MATLAB Parity Tests

Pre-recorded `.npz` fixtures in `tests/fixtures/` generated from MATLAB k-Wave
examples. Five scenarios:
1. Homogeneous lossless (initial pressure)
2. Heterogeneous lossless
3. Absorbing medium (power-law)
4. Nonlinear medium (BonA)
5. Time reversal reconstruction

Assertion: `max(|p_pykwave - p_matlab|) / max(|p_matlab|) < 1e-3`.

---

## 12. Implementation Order

1. `grid.py` — `KWaveGrid`
2. `pml.py` — `get_pml`
3. `utils.py` — `db2neper`, `smooth`, `interp2d`
4. `medium.py`, `source.py`, `sensor.py` — dataclasses
5. `_precompute.py` — all precomputed operators
6. `_sensor_ops.py` — gather helpers
7. `_scan_body.py` — lossless linear case first, then absorbing, then nonlinear
8. `kspace_first_order_2d.py` — validation + orchestration + postprocessing
9. `sensor.py` — `SensorData` + full sensor record logic
10. Unit tests, then parity tests
