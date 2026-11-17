"""Run an Ultralytics model (YOLO11 / RT-DETR) on one split and save predictions in the shared JSON format.

Usage (run from the repository root):
    python -m m3_dl.predict --weights path/to/best.pt --split val --name yolo11_default
"""
import argparse
import json
import time
import hashlib
import platform
from pathlib import Path

from utils.data import REPO_ROOT, list_images


def load_detector(weights, family="auto"):
    """Use the architecture-specific predictor, including generic best.pt names."""
    from ultralytics import YOLO, RTDETR
    if family == "rtdetr" or (family == "auto" and "rtdetr" in Path(weights).name.lower()):
        return RTDETR(weights)
    model = YOLO(weights)
    if family == "auto":
        underlying = getattr(model, "model", None)
        if "RTDETR" in type(underlying).__name__.upper().replace("_", ""):
            return RTDETR(weights)
    return model


def result_to_preds(result, image_id):
    """Convert one Ultralytics result into prediction dicts in the shared format."""
    preds = []
    xyxy = result.boxes.xyxy.cpu().tolist()   # [[x1, y1, x2, y2], ...] in pixels
    scores = result.boxes.conf.cpu().tolist()  # confidence of each box
    for (x1, y1, x2, y2), score in zip(xyxy, scores):
        preds.append({
            "image_id": image_id,
            "bbox": [x1, y1, x2 - x1, y2 - y1],  # corners -> top-left corner + width/height
            "score": score,
        })
    return preds


def main():
    parser = argparse.ArgumentParser(description="Predict with an Ultralytics model and save shared-format JSON.")
    parser.add_argument("--weights", required=True, help="path to the trained weights, e.g. best.pt")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--name", required=True, help="method name for the output file, e.g. yolo11_default")
    parser.add_argument("--imgsz", type=int, default=512, help="input size; must match training")
    parser.add_argument("--family", choices=["auto", "yolo", "rtdetr"], default="auto")
    parser.add_argument("--device", default=None, help="e.g. cpu, mps or 0")
    parser.add_argument("--conf", type=float, default=0.001, help="low floor for AP ranking")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--bird-class", type=int, help="required for multi-class checkpoints")
    args = parser.parse_args()

    import numpy as np
    import torch
    import ultralytics
    model = load_detector(args.weights, args.family)
    images = list_images(args.split)
    if not images:
        raise ValueError(f"No images found for split {args.split!r}; check configs/paths.yaml")
    if len(model.names) != 1 and args.bird_class is None:
        raise ValueError("Multi-class checkpoint: specify --bird-class so other classes are not exported as birds.")
    if args.bird_class is not None and not 0 <= args.bird_class < len(model.names):
        raise ValueError("--bird-class is outside the checkpoint's class range")
    kwargs = {"imgsz": args.imgsz, "conf": args.conf, "max_det": args.max_det, "verbose": False}
    if args.device is not None:
        kwargs["device"] = args.device
    if args.bird_class is not None:
        kwargs["classes"] = [args.bird_class]

    # Warm-up: the first call on the GPU is slow, so run it once before timing
    model.predict(str(images[0]), **kwargs)
    if torch.cuda.is_available() and args.device != "cpu":
        torch.cuda.synchronize()

    preds = []
    timings = []
    model_only = []
    start = time.perf_counter()
    for img_path in images:
        image_start = time.perf_counter()
        result = model.predict(str(img_path), **kwargs)[0]
        preds.extend(result_to_preds(result, img_path.stem))
        timings.append({"image_id": img_path.stem, "seconds": time.perf_counter() - image_start})
        if "inference" in getattr(result, "speed", {}):
            model_only.append(float(result.speed["inference"]))
    if torch.cuda.is_available() and args.device != "cpu":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    out_path = REPO_ROOT / "preds" / f"{args.name}_{args.split}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(preds, f)
    metadata = {"weights": str(Path(args.weights).resolve()), "family": type(model).__name__,
        "split": args.split, "images": len(images), "image_ids": [p.stem for p in images],
        "imgsz": args.imgsz, "confidence_floor": args.conf, "max_det": args.max_det,
        "bird_class": args.bird_class, "class_names": model.names,
        "device_requested": args.device, "platform": platform.platform(),
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "mean_end_to_end_ms": 1000 * elapsed / len(images),
        "median_end_to_end_ms": 1000 * float(np.median([r["seconds"] for r in timings])),
        "mean_model_only_ms": float(np.mean(model_only)) if model_only else None,
        "per_image_timing": timings,
        "timing_scope": "end to end includes load/preprocess/model/postprocess/CPU conversion; model_only is reported separately"}
    weight_path = Path(args.weights)
    backend = getattr(getattr(model, "predictor", None), "model", None)
    actual_device = getattr(backend, "device", None)
    metadata["actual_device"] = str(actual_device) if actual_device is not None else None
    metadata["gpu_name"] = torch.cuda.get_device_name(actual_device) if actual_device is not None and str(actual_device).startswith("cuda") else None
    metadata["torch_cpu_threads"] = torch.get_num_threads()
    metadata["half_precision"] = getattr(backend, "fp16", None)
    train_args = (getattr(model, "ckpt", None) or {}).get("train_args", {})
    keys = ["model", "pretrained", "epochs", "batch", "imgsz", "seed", "deterministic",
            "mosaic", "close_mosaic", "optimizer", "lr0", "lrf", "patience", "scale", "fliplr", "flipud"]
    metadata["checkpoint_training_args"] = {key: train_args[key] for key in keys if key in train_args}
    if weight_path.is_file():
        with weight_path.open("rb") as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            metadata["weights_sha256"] = digest.hexdigest()
    out_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"Wrote {len(preds)} boxes for {len(images)} images -> {out_path}")
    print(f"Inference time: {1000 * elapsed / len(images):.1f} ms per image (end to end)")


if __name__ == "__main__":
    main()
