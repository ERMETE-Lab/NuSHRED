"""Load saved HF/LF datasets without requiring FEniCS/dolfinx."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from models.config import DatasetConfig


def load_dataset(output_dir: str | Path) -> dict:
    """Load a dataset written by :func:`models.dataset.save_dataset`."""
    output_dir = Path(output_dir)
    meta = np.load(output_dir / "params.npz")
    mesh = np.load(output_dir / "mesh.npz")
    lf = np.load(output_dir / "lf.npz")["data"]
    num_species = int(meta["num_species"][0])

    hf = np.stack(
        [np.load(output_dir / f"hf_c{i}.npz")["data"] for i in range(num_species)],
        axis=1,
    )

    cfg = DatasetConfig(
        N=int(mesh["N"][0]),
        num_species=num_species,
        dt=float(meta["dt"][0]),
        T_final=float(meta["T_final"][0]),
        save_every=int(meta["save_every"][0]),
        velocity_scale=float(meta["velocity_scale"][0]),
    )

    result = {
        "params": meta["params"],
        "param_names": meta["param_names"],
        "times": meta["times"],
        "nodes": mesh["nodes"],
        "lf": lf,
        "hf": hf,
        "config": cfg,
        "output_dir": output_dir,
    }
    if "timing_hf" in meta.files:
        result["timing_hf"] = meta["timing_hf"]
        result["timing_lf"] = meta["timing_lf"]
    return result
