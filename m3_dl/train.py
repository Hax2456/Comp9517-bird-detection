"""Train the three M3 detectors with the settings used for the reported results.

Usage (run from the repository root, on a GPU machine):
    python -m m3_dl.train --run yolo11s_default
    python -m m3_dl.train --run yolo11s_nomosaic
    python -m m3_dl.train --run rtdetr_l

Each run writes weights, args.yaml and results.csv to runs/m3/<run>/.
The logs of the runs reported in the paper are kept in m3_dl/training_logs/<run>/.
"""
import argparse
import time

import yaml

from utils.data import REPO_ROOT, get_data_root

# The only settings that differ between the three runs
RUNS = {
    "yolo11s_default":  {"model": "yolo11s.pt",  "batch": 16, "mosaic": 1.0},
    "yolo11s_nomosaic": {"model": "yolo11s.pt",  "batch": 16, "mosaic": 0.0},  # mosaic ablation
    "rtdetr_l":         {"model": "rtdetr-l.pt", "batch": 8,  "mosaic": 1.0},  # batch 8 to fit GPU memory
}

# Settings shared by every run (all other hyperparameters are Ultralytics defaults)
COMMON = {"epochs": 60, "imgsz": 512, "seed": 0, "deterministic": True, "workers": 8}


def write_data_yaml():
    """Write the Ultralytics dataset file from configs/paths.yaml, so no paths are hard-coded."""
    cfg = {
        "path": str(get_data_root().resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "bird"},
    }
    out_path = REPO_ROOT / "data" / "processed" / "m3_data.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Train one M3 detector.")
    parser.add_argument("--run", required=True, choices=list(RUNS))
    parser.add_argument("--epochs", type=int, default=COMMON["epochs"], help="override only for a quick test")
    parser.add_argument("--device", default=None, help="e.g. 0 for the first GPU, or cpu")
    args = parser.parse_args()

    from ultralytics import RTDETR, YOLO  # imported here so --help works without ultralytics

    run_cfg = RUNS[args.run]
    model_cls = RTDETR if args.run.startswith("rtdetr") else YOLO
    model = model_cls(run_cfg["model"])  # COCO-pretrained weights, downloaded on first use

    train_kwargs = {**COMMON, "epochs": args.epochs, "batch": run_cfg["batch"], "mosaic": run_cfg["mosaic"]}
    if args.device is not None:
        train_kwargs["device"] = args.device

    data_yaml = write_data_yaml()
    start = time.perf_counter()
    model.train(
        data=str(data_yaml),
        project=str(REPO_ROOT / "runs" / "m3"),
        name=args.run,
        exist_ok=True,
        **train_kwargs,
    )
    minutes = (time.perf_counter() - start) / 60
    print(f"Training time for {args.run}: {minutes:.1f} min")
    print(f"Weights: {REPO_ROOT / 'runs' / 'm3' / args.run / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
