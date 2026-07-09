"""Parametric HF/LF reaction–diffusion dataset generation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from dolfinx import fem

from models.config import PARAM_NAMES, DatasetConfig
from models.solvers import AdvReactDiff_HF, AdvReactDiff_ODE, build_quadratic_reaction_tensor

LINEAR_SCALED_COLUMNS = (0, 1, 2)  # scale kinetics of species 0, 1, 2 (preserves column mass balance)
SOURCE_SPECIES = (0, 2, 4)

# --- Fixed problem setup -----------------------------------------------------------------------

BASE_DIFFUSION = [0.002, 0.001, 0.004, 0.002, 0.003, 0.0015]

# Column j sums to ~0 for j=0..4 (mass redistributed among active species).
# Column 5 is the terminal pool (species 5); net imbalance is allowed there.
BASE_LINEAR_REACTION = np.array(
    [
        [-0.40, 0.10, 0.05, 0.00, 0.00, 0.00],
        [0.20, -0.35, 0.10, 0.05, 0.00, 0.00],
        [0.15, 0.10, -0.40, 0.10, 0.05, 0.00],
        [0.00, 0.15, 0.10, -0.30, 0.10, 0.08],
        [0.05, 0.00, 0.15, 0.10, -0.20, 0.10],
        [0.00, 0.00, 0.00, 0.05, 0.05, -0.15],
    ]
)

QUADRATIC_REACTIONS = [
    {"reactants": [0, 0], "products": [1], "rate": 1.0},
    {"reactants": [1, 2], "products": [0], "rate": 2.0},
    {"reactants": [0, 2], "products": [1], "rate": 1.5},
    {"reactants": [3, 3], "products": [5], "rate": 1.5},
    {"reactants": [5, 5], "products": [1], "rate": 0.8},
    {"reactants": [5, 2], "products": [3], "rate": 1.8},
    {"reactants": [1, 3], "products": [4], "rate": 0.6},
    {"reactants": [4, 4], "products": [2], "rate": 0.5},
    {"reactants": [2, 4], "products": [5], "rate": 0.7},  # link even-indexed sources to terminal species
]

# Spatial source profiles S_i(x): active for species 0, 2, 4 only ([frequency, x0, y0]).
SINUSOIDAL_SOURCE_TERMS = [
    [2.0, 0.15, 0.50],
    [0.0, 0.00, 0.00],
    [1.5, 0.75, 0.25],
    [0.0, 0.00, 0.00],
    [1.0, 0.50, 0.75],
    [0.0, 0.00, 0.00],
]

INITIAL_CONDITIONS: list[Callable] = [
    lambda x: 0.5 * (np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])) ** 2,
    lambda x: 0.2 * (np.sin(np.pi * x[0]) * np.cos(np.pi * x[1])) ** 2,
    lambda x: 0.4 * (np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])) ** 2,
    lambda x: 0.6 * (np.cos(np.pi * x[0]) * np.cos(np.pi * x[1])) ** 2,
    lambda x: 0.2 * (np.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / (2 * 0.1**2))) ** 2,
    lambda x: 0.3 * (np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])) ** 2,
]


def build_param_grid(
    linear_scales: Sequence[float],
    source_t_peaks: Sequence[float],
    source_pulse_widths: Sequence[float],
) -> np.ndarray:
    """Build (Nparams, 3) array of [linear_scale, source_t_peak, source_pulse_width]."""
    arrays = [
        np.asarray(linear_scales, dtype=float),
        np.asarray(source_t_peaks, dtype=float),
        np.asarray(source_pulse_widths, dtype=float),
    ]
    grids = np.meshgrid(*arrays, indexing="ij")
    return np.column_stack([g.ravel() for g in grids])


def build_linear_reaction(num_species: int, linear_scale: float = 1.0) -> np.ndarray:
    """Return linear rate matrix with species 0–2 kinetics scaled (columns 0–2)."""
    k = BASE_LINEAR_REACTION[:num_species, :num_species].copy()
    for col in LINEAR_SCALED_COLUMNS:
        if col < num_species:
            k[:, col] *= linear_scale
    return k


def build_st_lambda(
    source_t_peak: float,
    source_pulse_width: float,
    num_species: int = 6,
) -> list[Callable[[float], float]]:
    """Temporal source amplitudes for species 0, 2, 4 (distinct shapes)."""
    if num_species != 6:
        raise ValueError("St_lambda is only defined for num_species=6")

    width = max(float(source_pulse_width), 0.05)
    t_peak = float(source_t_peak)

    def gaussian_pulse(t, tp=t_peak, w=width):
        return float(np.exp(-0.5 * ((t - tp) / w) ** 2))

    def sine_bump(t, tp=t_peak*1.5, w=width):
        t0, t1 = tp - w, tp + w
        if t < t0 or t > t1:
            return 0.0
        return float(np.sin(0.5 * np.pi * (t - t0) / (2.0 * w)) ** 2)

    def skewed_pulse(t, tp=t_peak*2.5, w=width):
        if t < tp - w:
            return 0.0
        if t <= tp:
            return float(((t - (tp - w)) / w) ** 2)
        return float(np.exp(-(t - tp) / w))

    st = [lambda t: 0.0 for _ in range(num_species)]
    st[0] = gaussian_pulse
    st[2] = sine_bump
    st[4] = skewed_pulse
    return st


def st_lambda_profiles(
    source_t_peak: float,
    source_pulse_width: float,
    times: np.ndarray,
    num_species: int = 6,
) -> np.ndarray:
    """Evaluate all St_lambda curves on a time grid; shape (num_species, Nt)."""
    st = build_st_lambda(source_t_peak, source_pulse_width, num_species)
    return np.array([[fn(t) for t in times] for fn in st])


def build_physical_params(num_species: int, linear_scale: float = 1.0) -> dict:
    base_quad = build_quadratic_reaction_tensor(num_species, QUADRATIC_REACTIONS)
    return {
        "num_species": num_species,
        "diffusion": BASE_DIFFUSION[:num_species],
        "linear_reaction": build_linear_reaction(num_species, linear_scale),
        "quadratic_reaction": base_quad[:num_species, :num_species, :num_species].copy(),
    }


def get_mesh_nodes(hf_model: AdvReactDiff_HF) -> np.ndarray:
    return hf_model.domain.geometry.x[:, :2].copy()


def compute_si_avg(hf_model: AdvReactDiff_HF) -> np.ndarray:
    return np.array(
        [fem.assemble_scalar(fem.form(hf_model.Si[i] * hf_model.dx)) for i in range(hf_model.num_species)]
    )


def compute_ic_avg(hf_model: AdvReactDiff_HF) -> np.ndarray:
    return np.array(
        [
            fem.assemble_scalar(fem.form(hf_model.u_old.sub(i).collapse() * hf_model.dx))
            for i in range(hf_model.num_species)
        ]
    )


def run_hf_trajectory(
    hf_model: AdvReactDiff_HF,
    times: np.ndarray,
    st_lambda: Sequence[Callable[[float], float]],
    store_indices: np.ndarray,
) -> np.ndarray:
    """Integrate HF model; return array (num_species, Nt, Nspace)."""
    n_store = len(store_indices)
    n_nodes = hf_model.u_old.sub(0).collapse().x.array.shape[0]
    out = np.zeros((hf_model.num_species, n_store, n_nodes))

    store_set = set(store_indices)
    j = 0
    for step, t in enumerate(times):
        snap = hf_model.advance(float(t), st_lambda)
        if step in store_set:
            for i in range(hf_model.num_species):
                out[i, j, :] = snap[i].x.array.copy()
            j += 1
    return out


def generate_dataset(
    config: DatasetConfig,
    params: np.ndarray,
    progress: bool = True,
) -> dict:
    """Generate paired HF spatial fields and LF lumped trajectories.

    Parameters
    ----------
    params : (Nparams, 3) array
        Columns are [linear_scale, source_t_peak, source_pulse_width].
    """
    params = np.asarray(params, dtype=float)
    if params.ndim != 2 or params.shape[1] != 3:
        raise ValueError(
            "params must have shape (Nparams, 3) with "
            "[linear_scale, source_t_peak, source_pulse_width]"
        )

    times = np.arange(0.0, config.T_final + 0.5 * config.dt, config.dt)
    store_indices = np.arange(0, len(times), config.save_every)
    stored_times = times[store_indices]
    n_params = params.shape[0]

    ref_params = build_physical_params(config.num_species, linear_scale=1.0)
    ref_model = AdvReactDiff_HF(config.N, ref_params)
    ref_model.assign_initial_conditions(INITIAL_CONDITIONS[: config.num_species])
    ref_model.assign_sinusoidal_source_terms(SINUSOIDAL_SOURCE_TERMS[: config.num_species])
    nodes = get_mesh_nodes(ref_model)
    si_avg = compute_si_avg(ref_model)
    ic_avg = compute_ic_avg(ref_model)

    n_nodes = nodes.shape[0]
    n_store = len(store_indices)
    hf = np.zeros((n_params, config.num_species, n_store, n_nodes))
    lf = np.zeros((n_params, n_store, config.num_species))
    timing_hf = np.zeros(n_params)
    timing_lf = np.zeros(n_params)

    iterator = range(n_params)
    if progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="Parameter cases", unit="case")

    for param_idx in iterator:
        linear_scale, source_t_peak, source_pulse_width = params[param_idx]
        physical_params = build_physical_params(config.num_species, linear_scale=linear_scale)
        st_lambda = build_st_lambda(source_t_peak, source_pulse_width, config.num_species)

        hf_model = AdvReactDiff_HF(config.N, physical_params)
        hf_model.assign_initial_conditions(INITIAL_CONDITIONS[: config.num_species])
        hf_model.assign_velocity(scale=config.velocity_scale)
        hf_model.assign_sinusoidal_source_terms(SINUSOIDAL_SOURCE_TERMS[: config.num_species])
        hf_model.assemble_form(dt=config.dt, direct_solver=config.direct_solver)

        t0_hf = time.perf_counter()
        hf[param_idx] = run_hf_trajectory(hf_model, times, st_lambda, store_indices)
        timing_hf[param_idx] = time.perf_counter() - t0_hf

        ode_model = AdvReactDiff_ODE(physical_params, si_avg)
        t0_lf = time.perf_counter()
        ode_sol = ode_model.solve((0.0, times[-1]), ic_avg, st_lambda)
        lf[param_idx] = ode_sol.sol(stored_times).T
        timing_lf[param_idx] = time.perf_counter() - t0_lf

    return {
        "params": params,
        "param_names": np.array(PARAM_NAMES),
        "times": stored_times,
        "nodes": nodes,
        "si_avg": si_avg,
        "ic_avg": ic_avg,
        "hf": hf,
        "lf": lf,
        "timing_hf": timing_hf,
        "timing_lf": timing_lf,
        "config": config,
    }


def save_dataset(data: dict, output_dir: str | Path) -> Path:
    """Save mesh, parameters, LF, and one HF file per species."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = data["config"]

    np.savez_compressed(
        output_dir / "mesh.npz",
        nodes=data["nodes"],
        N=np.array([cfg.N]),
    )

    np.savez_compressed(
        output_dir / "params.npz",
        params=data["params"],
        linear_scale=data["params"][:, 0],
        source_t_peak=data["params"][:, 1],
        source_pulse_width=data["params"][:, 2],
        param_names=data["param_names"],
        times=data["times"],
        num_species=np.array([cfg.num_species]),
        dt=np.array([cfg.dt]),
        T_final=np.array([cfg.T_final]),
        save_every=np.array([cfg.save_every]),
        velocity_scale=np.array([cfg.velocity_scale]),
        timing_hf=data["timing_hf"],
        timing_lf=data["timing_lf"],
    )

    np.savez_compressed(output_dir / "lf.npz", data=data["lf"])

    for species_idx in range(cfg.num_species):
        np.savez_compressed(
            output_dir / f"hf_c{species_idx}.npz",
            data=data["hf"][:, species_idx, :, :],
        )

    return output_dir
