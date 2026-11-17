"""Plot validation mAP per epoch for the three M3 runs (report figure).

Usage (run from the repository root):
    python -m m3_dl.plot_training
Reads m3_dl/training_logs/<run>/results.csv and writes m3_training_curves.png/.pdf there.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils.data import REPO_ROOT

LOG_DIR = REPO_ROOT / "m3_dl" / "training_logs"
RUNS = [  # (folder, legend name, colour)
    ("rtdetr_l", "RT-DETR-L", "#2a78d6"),
    ("yolo11s_default", "YOLO11s", "#eb6834"),
    ("yolo11s_nomosaic", "YOLO11s no-mosaic", "#1baf7a"),
]
PANELS = [  # (results.csv column, panel title, y-axis upper limit)
    ("metrics/mAP50(B)", "Validation mAP@0.5", 1.0),
    ("metrics/mAP50-95(B)", "Validation mAP@0.5:0.95", 0.6),
]


def main():
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
    for ax, (column, title, y_max) in zip(axes, PANELS):
        for folder, name, colour in RUNS:
            df = pd.read_csv(LOG_DIR / folder / "results.csv")
            ax.plot(df["epoch"], df[column], color=colour, lw=1.5, label=name)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("Epoch")
        ax.set_xlim(1, 60)
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(LOG_DIR / f"m3_training_curves.{ext}", dpi=300)
    print(f"Saved figure to {LOG_DIR}")


if __name__ == "__main__":
    main()
