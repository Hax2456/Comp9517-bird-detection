"""Unified evaluation script: compare a prediction file with the ground truth and report mAP.

Prediction file format (shared by all methods): a JSON list with one entry per box
    [{"image_id": "beach_bird_0001", "bbox": [x, y, w, h], "score": 0.93}, ...]
    - image_id: image file name without ".jpg"
    - bbox: pixel coordinates, (x, y) is the top-left corner, w and h are width and height
    - score: confidence, higher means more likely to be a bird

Usage (run from the repository root):
    python -m evaluation.evaluate --pred preds/yolo11_default_val.json --split val
"""
import argparse
import contextlib
import io
import json

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.data import REPO_ROOT

GT_DIR = REPO_ROOT / "data" / "processed"
MAX_DETS = 300  # max boxes per image; COCO's default of 100 is too low (up to 130 birds per image)


def load_predictions(pred_path, coco_gt):
    """Load a prediction file and map each image_id from file name to the integer COCO id."""
    name_to_id = {img["file_name"].rsplit(".", 1)[0]: img["id"] for img in coco_gt.dataset["images"]}
    with open(pred_path) as f:
        preds = json.load(f)

    results = []
    for i, p in enumerate(preds):
        name = str(p["image_id"])
        if name not in name_to_id:
            raise ValueError(f"Prediction {i}: image_id={name!r} is not in this split. Did you use the wrong --split?")
        x, y, w, h = map(float, p["bbox"])
        if w <= 0 or h <= 0:
            continue  # drop degenerate boxes
        results.append({
            "image_id": name_to_id[name],
            "category_id": 1,
            "bbox": [x, y, w, h],
            "score": float(p["score"]),
        })
    return results


def evaluate(pred_path, split="val", verbose=True):
    """Return a dict with mAP@0.5, mAP@0.5:0.95 and AP@0.5 for small / medium / large birds."""
    gt_path = GT_DIR / f"{split}_gt.json"
    with contextlib.redirect_stdout(io.StringIO()):  # silence pycocotools' own logging
        coco_gt = COCO(str(gt_path))

    results = load_predictions(pred_path, coco_gt)
    if not results:
        print("Prediction file is empty; all metrics set to 0.")
        return {"AP50": 0.0, "AP50_95": 0.0, "AP50_small": 0.0, "AP50_medium": 0.0, "AP50_large": 0.0}

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(results)
        E = COCOeval(coco_gt, coco_dt, iouType="bbox")
        E.params.maxDets = [1, 10, MAX_DETS]
        E.evaluate()
        E.accumulate()

    # precision has shape [IoU thresholds, recall points, classes, area ranges, maxDets]
    precision = E.eval["precision"]
    iou_thrs = E.params.iouThrs

    def ap(iou=None, area_idx=0):
        p = precision[:, :, 0, area_idx, -1]  # class 0 (bird), last maxDets setting
        if iou is not None:
            p = p[np.isclose(iou_thrs, iou)]
        p = p[p > -1]  # -1 means no ground truth for this setting; exclude it
        return float(p.mean()) if p.size else float("nan")

    metrics = {
        "AP50": ap(iou=0.5),
        "AP50_95": ap(),
        "AP50_small": ap(iou=0.5, area_idx=1),   # area < 32^2
        "AP50_medium": ap(iou=0.5, area_idx=2),  # 32^2 to 96^2
        "AP50_large": ap(iou=0.5, area_idx=3),   # area > 96^2
    }
    if verbose:
        print(f"Predictions: {pred_path}  ({len(results)} boxes, split={split})")
        print(f"  mAP@0.5      = {metrics['AP50']:.4f}   <- main metric")
        print(f"  mAP@0.5:0.95 = {metrics['AP50_95']:.4f}")
        print(f"  AP@0.5 small / medium / large = "
              f"{metrics['AP50_small']:.4f} / {metrics['AP50_medium']:.4f} / {metrics['AP50_large']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a prediction file against the ground truth.")
    parser.add_argument("--pred", required=True, help="path to the prediction JSON file")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()
    evaluate(args.pred, args.split)


if __name__ == "__main__":
    main()