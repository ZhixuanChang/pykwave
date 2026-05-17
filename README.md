# pykwave

A JAX implementation of the [k-Wave](http://www.k-wave.org) pseudospectral acoustic simulator, beginning with `kspaceFirstOrder2D`. Designed for GPU-accelerated medical ultrasound research with full feature parity to MATLAB k-Wave v1.3.

## Features

- **Pseudospectral k-space method** — FFT-based spatial derivatives with k-space correction, matching MATLAB k-Wave numerics
- **JIT-compiled time loop** — `jax.lax.scan` + `@jax.jit` for efficient XLA compilation; no Python overhead per time step
- **Equations of state** — lossless, power-law absorbing (fractional Laplacian), Stokes absorbing, and nonlinear (BonA)
- **Source types** — initial pressure `p0`, time-varying additive/Dirichlet pressure and velocity sources
- **Sensor recording** — all 17 k-Wave record fields: pressure/velocity time series, running max/min/RMS, final fields, acoustic intensity
- **Sensor masks** — binary (grid-aligned) and Cartesian (arbitrary coordinates, Delaunay interpolation)
- **Time reversal** — image reconstruction from boundary pressure data
- **PML** — quartic perfectly matched layer absorbing boundary, matching MATLAB `getPML.m`
- **Gaussian frequency filter** — `sensor.frequency_response` post-processing

---

## Architecture

```
pykwave/
├── pykwave/
│   ├── __init__.py                  # Public API re-exports
│   ├── grid.py                      # KWaveGrid — spatial/temporal grid
│   ├── medium.py                    # Medium dataclass (sound speed, density, absorption)
│   ├── source.py                    # Source dataclass (p0, p, ux, uy)
│   ├── sensor.py                    # Sensor + SensorData dataclasses
│   ├── pml.py                       # PML absorbing boundary (quartic profile)
│   ├── utils.py                     # db2neper, smooth, interp_rho_staggered
│   ├── kspace_first_order_2d.py     # Public solver entry point
│   ├── _precompute.py               # Pre-scan operator construction
│   ├── _scan_body.py                # lax.scan kernel (SimState + scan_step)
│   └── _sensor_ops.py               # Post-scan sensor data assembly
└── tests/
    ├── fixtures/                    # MATLAB-generated .npz parity fixtures
    │   ├── generate_fixtures.m      # Run in MATLAB to generate fixtures
    │   └── generate_fixtures.py     # Python wrapper
    ├── conftest.py
    ├── test_grid.py
    ├── test_pml.py
    ├── test_utils.py
    ├── test_scan_body.py
    ├── test_kspace_first_order_2d.py
    └── test_parity.py               # MATLAB parity tests (skipped without fixtures)
```

### How it works

**1. Precompute** (`_precompute.py`) — Before the time loop, all operators are built as JAX arrays and Python flags are resolved:

- Spectral derivative operators (`ddx_k_shift_pos/neg`, `ddy_k_shift_pos/neg`) with stagger-shift phase factors
- k-space correction (`kappa = sinc(c_ref·k·dt/2)`, `source_kappa = cos(c_ref·k·dt/2)`)
- PML arrays for pressure and staggered velocity grids
- Absorption operators (`absorb_nabla1 = k^(α-2)`, `absorb_nabla2 = k^(α-1)`) for power-law media
- Staggered-grid density interpolation (`rho0_sgx`, `rho0_sgy` at ±0.5 grid offsets)
- Pre-scaled source signals indexed into per-step scan inputs

**2. Time loop** (`_scan_body.py`) — A single pure function stepped `Nt` times by `jax.lax.scan`:

```
Step 1: Update velocity (momentum equation, PML-damped)
Step 2: Inject velocity source
Step 3: Spectral divergence (∂ux/∂x + ∂uy/∂y)
Step 4: Update density (continuity equation, PML-damped)
Step 5: Inject pressure source
Step 6: Equation of state → pressure (lossless / Stokes / absorbing / nonlinear)
Step 7: Enforce p0 initial condition (t=0 only, via jnp.where)
Step 8: Update p_k = fft2(p)
Step 9: Gather sensor data; update running statistics
```

All Python `if flags[...]` branches are resolved at trace time — the compiled XLA graph contains only the physics paths actually needed.

**3. Post-process** (`_sensor_ops.py`) — After the scan, running accumulators are converted to final fields (RMS, max/min, intensity) and the Gaussian frequency filter is applied.

---

## Installation

```bash
git clone https://github.com/ZhixuanChang/pykwave.git
cd pykwave
pip install -e .
```

**Requirements:** Python 3.10+, JAX ≥ 0.10.0, NumPy ≥ 1.24, SciPy ≥ 1.10.

For GPU acceleration, install JAX with CUDA support first — see the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

---

## Quick Start

### Point source, homogeneous medium

```python
import numpy as np
from pykwave import KWaveGrid, Medium, Source, Sensor, kspace_first_order_2d

# 1. Create a 128×128 grid with 1 mm spacing
kgrid = KWaveGrid(128, 1e-3, 128, 1e-3)
kgrid.make_time(1500.0, cfl=0.3)   # sets dt and Nt automatically

# 2. Define the medium
medium = Medium(sound_speed=1500.0, density=1000.0)

# 3. Define an initial pressure source (Gaussian blob at grid centre)
p0 = np.zeros((128, 128), dtype=np.float32)
p0[64, 64] = 1.0
source = Source(p0=p0)

# 4. Define a sensor ring
mask = np.zeros((128, 128), dtype=np.float32)
for angle in np.linspace(0, 2 * np.pi, 360, endpoint=False):
    xi = int(round(64 + 20 * np.cos(angle)))
    yi = int(round(64 + 20 * np.sin(angle)))
    if 0 <= xi < 128 and 0 <= yi < 128:
        mask[xi, yi] = 1.0
sensor = Sensor(mask=mask)

# 5. Run the simulation
# Returns a JAX array of shape (N_sensors, Nt)
data = kspace_first_order_2d(kgrid, medium, source, sensor)
print(data.shape)   # e.g. (360, 1050)
```

### Absorbing medium with multiple sensor.record fields

```python
from pykwave import Medium, Sensor

medium = Medium(
    sound_speed=1500.0,
    density=1000.0,
    alpha_coeff=0.5,   # dB/(MHz^y cm) — triggers power-law absorbing EOS
    alpha_power=1.5,
)

sensor = Sensor(
    mask=mask,
    record=['p', 'p_max', 'p_rms', 'p_final'],
)

sd = kspace_first_order_2d(kgrid, medium, source, sensor)
print(sd.p.shape)       # (N_sensors, Nt_record)
print(sd.p_max.shape)   # (N_sensors,)
print(sd.p_final.shape) # (Nx_inner, Ny_inner)
```

### Time-varying pressure source

```python
import numpy as np
from pykwave import Source, Sensor

# Sinusoidal tone burst at 500 kHz
t = np.arange(kgrid.Nt) * kgrid.dt
freq = 500e3
sig = np.sin(2 * np.pi * freq * t) * np.exp(-((t - 2e-6) / 0.5e-6) ** 2)

mask_src = np.zeros((128, 128), dtype=np.float32)
mask_src[64, 64] = 1.0

source = Source(p=sig[None, :], p_mask=mask_src)   # (1, Nt) broadcasts to all source points
```

### Heterogeneous medium

```python
import numpy as np
from pykwave import Medium

c0 = np.full((128, 128), 1500.0, dtype=np.float32)
c0[60:70, :] = 1800.0   # faster layer

rho = np.full((128, 128), 1000.0, dtype=np.float32)
rho[60:70, :] = 1200.0

medium = Medium(sound_speed=c0, density=rho)
```

---

## API Reference

### `KWaveGrid(Nx, dx, Ny, dy)`

| Method / Property | Description |
|---|---|
| `make_time(sound_speed, cfl=0.3, t_end=None)` | Auto-compute `dt` and `Nt` from CFL condition |
| `set_time(Nt, dt)` | Manual time stepping |
| `kx_vec`, `ky_vec` | Wavenumber vectors (fftshift order, DC at centre) |
| `k` | Wavenumber magnitude grid `(Nx, Ny)` |
| `t_array` | Time array `(Nt,)` |

### `kspace_first_order_2d(kgrid, medium, source, sensor, **options)`

| Option | Default | Description |
|---|---|---|
| `pml_size` | `20` | PML thickness in grid points; `int` or `(int, int)` for x/y |
| `pml_alpha` | `2.0` | PML absorption coefficient; `float` or `(float, float)` |
| `pml_inside` | `True` | PML region inside the grid (True) or add to grid (False) |
| `smooth` | `True` | Hann-window smooth `p0` before simulation |

**Returns:** `SensorData` dataclass if `sensor.record` is set; plain JAX array `(N_sensors, Nt)` otherwise.

### `sensor.record` fields

| Field | Shape | Description |
|---|---|---|
| `p` | `(N_sensors, Nt)` | Pressure time series |
| `p_max`, `p_min` | `(N_sensors,)` | Peak positive/negative pressure |
| `p_rms` | `(N_sensors,)` | RMS pressure |
| `p_final` | `(Nx, Ny)` | Pressure at final time step |
| `p_max_all`, `p_min_all` | `(Nx, Ny)` | Peak pressure over all grid points |
| `u` | `(N_sensors, Nt)` | Staggered velocity `(ux, uy)` |
| `u_max`, `u_min`, `u_rms` | `(N_sensors,)` | Velocity statistics |
| `u_final` | `(Nx, Ny)` | Velocity at final time step |
| `u_max_all`, `u_min_all` | `(Nx, Ny)` | Peak velocity over all grid points |
| `u_non_staggered` | `(N_sensors, Nt)` | Velocity interpolated to pressure grid |
| `I`, `I_avg` | `(N_sensors, Nt)`, `(N_sensors,)` | Acoustic intensity and time average |

---

## Testing

```bash
pip install pytest
pytest tests/ -v
```

Expected result: **29 passed, 2 skipped** (parity tests skip when MATLAB fixtures are absent).

### Generating MATLAB parity fixtures

The 2 skipped tests compare pykwave output against MATLAB k-Wave reference data. To run them:

1. Open MATLAB with k-Wave on the path
2. Run `tests/fixtures/generate_fixtures.m`
3. Re-run pytest — the 2 parity tests will now execute

---

## Known Limitations

- **Absorbing EOS stability** — The fractional Laplacian absorbing EOS (`alpha_power ≠ 2`) has a stability constraint. For realistic tissue values (`alpha_coeff = 0.5 dB/(MHz^1.5 cm)`) with 1 mm grids the simulation goes unstable; use much smaller `alpha_coeff` or the Stokes approximation (`alpha_power = 2.0`) for production runs until a stabilisation scheme is added.
- **MATLAB parity** — Numerical agreement with MATLAB k-Wave is expected but not yet verified (pending fixture generation).
- **Directional sensor response** — `sensor.directivity_angle` is accepted but not yet applied in post-processing.
- **Heterogeneous source scaling** — Source pre-scaling uses `c0` at the first source point; this is an approximation for strongly heterogeneous media.
- **2D only** — `kspaceFirstOrder1D` and `kspaceFirstOrder3D` are not implemented.

---

## License

pykwave is released under the [GNU Lesser General Public License v3](LICENSE).

This library is based on the [k-Wave MATLAB Toolbox](http://www.k-wave.org) by B.E. Treeby and B.T. Cox, which is also licensed under LGPLv3. The algorithms implemented here are a JAX/Python reimplementation of the k-Wave pseudospectral method; no MATLAB source code is reproduced verbatim.

> Treeby, B.E. and Cox, B.T., "k-Wave: MATLAB toolbox for the simulation and reconstruction of photoacoustic wave fields," *Journal of Biomedical Optics*, 15(2), 021314, 2010. https://doi.org/10.1117/1.3360308
