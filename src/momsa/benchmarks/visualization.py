from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/momsa-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def save_pareto_plot(problem_name: str, true_front: np.ndarray, fronts: list[tuple[str, np.ndarray]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_obj = true_front.shape[1] if len(true_front) else fronts[0][1].shape[1]

    if n_obj == 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        if len(true_front):
            ax.plot(true_front[:, 0], true_front[:, 1], color="black", linewidth=1.5, label="True PF")
        for label, front in fronts:
            if len(front):
                ax.scatter(front[:, 0], front[:, 1], s=16, alpha=0.75, label=label)
        ax.set_title(f"{problem_name} Pareto Front")
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.legend()
        ax.grid(True, alpha=0.25)
    elif n_obj == 3:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
        if len(true_front):
            ax.scatter(true_front[:, 0], true_front[:, 1], true_front[:, 2], s=10, alpha=0.18, label="True PF")
        for label, front in fronts:
            if len(front):
                ax.scatter(front[:, 0], front[:, 1], front[:, 2], s=18, alpha=0.75, label=label)
        ax.set_title(f"{problem_name} Pareto Front")
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.set_zlabel("f3")
        ax.legend()
    else:
        raise ValueError(f"Plotting supports only 2D or 3D fronts, got {n_obj}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
