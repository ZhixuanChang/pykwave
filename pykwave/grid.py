from functools import cached_property
import jax.numpy as jnp
import numpy as np


class KWaveGrid:
    def __init__(self, Nx: int, dx: float, Ny: int, dy: float):
        self._Nx = int(Nx)
        self._dx = float(dx)
        self._Ny = int(Ny)
        self._dy = float(dy)
        self._Nt: int | None = None
        self._dt: float | None = None

    @property
    def Nx(self):
        return self._Nx

    @property
    def dx(self):
        return self._dx

    @property
    def Ny(self):
        return self._Ny

    @property
    def dy(self):
        return self._dy

    @property
    def dim(self):
        return 2

    @property
    def total_grid_points(self):
        return self._Nx * self._Ny

    @property
    def x_size(self):
        return self._Nx * self._dx

    @property
    def y_size(self):
        return self._Ny * self._dy

    @cached_property
    def x_vec(self):
        return (jnp.arange(self._Nx, dtype=jnp.float32) - (self._Nx - 1) / 2) * self._dx

    @cached_property
    def y_vec(self):
        return (jnp.arange(self._Ny, dtype=jnp.float32) - (self._Ny - 1) / 2) * self._dy

    @cached_property
    def x(self):
        return jnp.broadcast_to(self.x_vec[:, None], (self._Nx, self._Ny))

    @cached_property
    def y(self):
        return jnp.broadcast_to(self.y_vec[None, :], (self._Nx, self._Ny))

    @cached_property
    def kx_vec(self):
        # fftshift order: sorted, DC at center
        return jnp.fft.fftshift(
            jnp.fft.fftfreq(self._Nx, d=self._dx) * 2 * jnp.pi
        )

    @cached_property
    def ky_vec(self):
        return jnp.fft.fftshift(
            jnp.fft.fftfreq(self._Ny, d=self._dy) * 2 * jnp.pi
        )

    @cached_property
    def kx(self):
        return jnp.broadcast_to(self.kx_vec[:, None], (self._Nx, self._Ny))

    @cached_property
    def ky(self):
        return jnp.broadcast_to(self.ky_vec[None, :], (self._Nx, self._Ny))

    @cached_property
    def k(self):
        return jnp.sqrt(self.kx ** 2 + self.ky ** 2)

    @cached_property
    def k_max(self):
        return float(jnp.max(self.k))

    @property
    def Nt(self):
        return self._Nt

    @property
    def dt(self):
        return self._dt

    @property
    def t_array(self):
        if self._Nt is None:
            return "auto"
        return jnp.arange(self._Nt, dtype=jnp.float32) * self._dt

    def set_time(self, Nt: int, dt: float):
        self._Nt = int(Nt)
        self._dt = float(dt)

    def make_time(self, sound_speed, cfl: float = 0.3, t_end: float | None = None):
        c = np.asarray(sound_speed, dtype=np.float64)
        c_max = float(c.max())
        c_min = float(c.min())
        dt = cfl * min(self._dx, self._dy) / c_max
        if t_end is None:
            diag = np.sqrt((self._Nx * self._dx) ** 2 + (self._Ny * self._dy) ** 2)
            t_end = diag / c_min
        self._Nt = int(np.ceil(t_end / dt))
        self._dt = dt
