from dataclasses import dataclass
from typing import Any

@dataclass
class Medium:
    sound_speed: Any              # scalar or (Nx, Ny) [m/s]
    density: Any = None           # scalar or (Nx, Ny) [kg/m³]
    sound_speed_ref: float = None
    BonA: Any = None              # nonlinearity parameter
    alpha_coeff: float = None     # [dB/(MHz^y cm)]
    alpha_power: float = None
    alpha_mode: str = None        # 'no_absorption' | 'no_dispersion'
    alpha_filter: Any = None      # (Nx, Ny) frequency-domain filter
    alpha_sign: tuple = None      # (sign_abs, sign_disp)
