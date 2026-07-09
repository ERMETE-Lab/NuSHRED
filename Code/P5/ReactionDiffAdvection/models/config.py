"""Shared dataset configuration (no FEniCS/dolfinx dependency)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PARAM_NAMES = ("linear_scale", "source_t_peak", "source_pulse_width")
DEFAULT_VELOCITY_SCALE = 5.0


@dataclass
class DatasetConfig:
    N: int = 50
    num_species: int = 6
    dt: float = 0.005
    T_final: float = 10.0
    save_every: int = 10
    direct_solver: bool = False
    velocity_scale: float = DEFAULT_VELOCITY_SCALE


def case_label(params_row: np.ndarray, case_idx: int) -> str:
    """Short label for dropdown menus."""
    ls, tp, w = params_row
    return f"case {case_idx}: linear={ls:g}, t_peak={tp:g}, width={w:g}"
