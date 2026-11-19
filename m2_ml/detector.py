"""HOG / HOG+HSV, linear SVM, multiscale windows, NMS and hard negatives.

No pretrained detector is used. Boxes throughout this module are pixel xywh.
Dataset splits are never pooled. See the notebook for a complete experiment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
from time import perf_counter

import joblib
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog
from tqdm.auto import tqdm

from utils.data import read_yolo_boxes


@dataclass
class Config:
    seed: int = 42
    patch_size: int = 32
    orientations: int = 9
    cell_size: int = 4
    block_cells: int = 2
    hog_backend: str = "vectorized"
    hsv_bins: int = 8
    n_templates: int = 9
    stride_fraction: float = 0.25
    min_stride: int = 2
    negatives_per_image: int = 24
    negative_iou: float = 0.10
    negative_coverage: float = 0.10
    max_positives: int = 12000
    C: float = 0.1
    batch_size: int = 256
    score_floor: float = -2.0
    nms_iou: float = 0.3
    pre_nms_topk: int = 5000
    max_detections: int = 300
    hard_negatives_per_image: int = 12
    max_hard_negatives: int = 6000


@dataclass
class Record:
    path: Path
    width: int
    height: int
    boxes: np.ndarray


def load_records(data_root, split):
    """Load a supplied split, requiring a label file even for negative images."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    root = Path(data_root)
    paths = sorted((root / "images" / split).glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No JPG images in {root / 'images' / split}")
    records = []
    for path in paths:
        label = root / "labels" / split / f"{path.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Missing label: {label}; empty labels are allowed.")
        for line in label.read_text().splitlines():
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Malformed YOLO label: {label}")
            values = np.asarray([float(v) for v in fields])
            if not np.isfinite(values).all() or values[0] != 0:
                raise ValueError(f"Expected finite single-class bird label: {label}")
            if np.any(values[1:] < 0) or np.any(values[1:] > 1) or np.any(values[3:] <= 0):
                raise ValueError(f"Invalid normalised YOLO box: {label}")
        with Image.open(path) as image:
            width, height = image.size
        boxes = np.asarray(read_yolo_boxes(label, width, height), dtype=float).reshape(-1, 4)
        if len(boxes) and np.any(boxes[:, 2:] <= 0):
            raise ValueError(f"Degenerate clipped box: {label}")
        records.append(Record(path, width, height, boxes))
    return records


def subset(records, limit=None, seed=42):
    """Fixed image-level subsampling; never split crops across dataset splits."""
    if limit is None or limit >= len(records):
        return list(records)
    if limit <= 0:
        raise ValueError("limit must be positive or None")
    indices = np.sort(np.random.default_rng(seed).choice(len(records), limit, replace=False))
    return [records[i] for i in indices]


def read_rgb(record):
    with Image.open(record.path) as image:
        return np.asarray(image.convert("RGB"))


def intersections(box, boxes):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    box = np.asarray(box, dtype=float)
    lo = np.maximum(box[:2], boxes[:, :2])
    hi = np.minimum(box[:2] + box[2:], boxes[:, :2] + boxes[:, 2:])
    return np.prod(np.maximum(hi - lo, 0), axis=1)


def iou_one_to_many(box, boxes):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    inter = intersections(box, boxes)
    union = box[2] * box[3] + boxes[:, 2] * boxes[:, 3] - inter
    return inter / np.maximum(union, 1e-12)


def is_background(box, gt, cfg):
    """Reject both overlapping windows and large windows containing tiny birds."""
    if len(gt) == 0:
        return True
    inter = intersections(box, gt)
    coverage = inter / np.maximum(gt[:, 2] * gt[:, 3], 1e-12)
    return bool(np.all(iou_one_to_many(box, gt) < cfg.negative_iou)
                and np.all(coverage < cfg.negative_coverage))


def crop_rgb(image, box):
    """Clip and rasterise an xywh box consistently at training and inference."""
    height, width = image.shape[:2]
    x, y, w, h = box
    x0, y0 = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
    x1, y1 = min(width, int(np.ceil(x + w))), min(height, int(np.ceil(y + h)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Empty crop for box {box}")
    return image[y0:y1, x0:x1]


def hog_descriptor(gray, cfg):
    """HOG with vectorised L2-Hys blocks; verified against public skimage HOG.

    Uses scikit-image's compiled cell-histogram kernel, then normalises all
    overlapping blocks together. Gradient convention, histogram kernel, epsilon,
    clipping and ordering match skimage.feature.hog. Set hog_backend='reference'
    to use the public implementation. This does not share image-boundary HOG
    between different crops, so the original crop-wise descriptor is preserved.
    """
    if cfg.hog_backend == "reference":
        return hog(gray, orientations=cfg.orientations,
                   pixels_per_cell=(cfg.cell_size, cfg.cell_size),
                   cells_per_block=(cfg.block_cells, cfg.block_cells),
                   block_norm="L2-Hys", feature_vector=True)
    if cfg.hog_backend != "vectorized":
        raise ValueError("hog_backend must be vectorized or reference")
    from skimage.feature import _hoghistogram
    gray = np.asarray(gray, dtype=float)
    gy, gx = np.zeros_like(gray), np.zeros_like(gray)
    gy[1:-1] = gray[2:] - gray[:-2]
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    height, width = gray.shape
    cell, block = cfg.cell_size, cfg.block_cells
    rows, cols = height // cell, width // cell
    if rows < block or cols < block:
        raise ValueError("Crop is too small for the HOG block geometry")
    hist = np.zeros((rows, cols, cfg.orientations), dtype=float)
    _hoghistogram.hog_histograms(gx, gy, cell, cell, width, height, cols, rows,
                                 cfg.orientations, hist)
    windows = np.lib.stride_tricks.sliding_window_view(hist, (block, block), axis=(0, 1))
    values = windows.transpose(0, 1, 3, 4, 2).copy()
    axes = (2, 3, 4)
    values /= np.sqrt(np.sum(values * values, axis=axes, keepdims=True) + 1e-10)
    np.minimum(values, 0.2, out=values)
    values /= np.sqrt(np.sum(values * values, axis=axes, keepdims=True) + 1e-10)
    return values.ravel()


def extract_features(patch, cfg, use_color=False):
    """Same crop-resize-HOG computation for positives, negatives and detection."""
    resized = np.asarray(Image.fromarray(patch).resize(
        (cfg.patch_size, cfg.patch_size), Image.Resampling.BILINEAR))
    shape = hog_descriptor(rgb2gray(resized), cfg)
    if not use_color:
        return shape.astype(np.float32)
    hsv = rgb2hsv(resized)
    color = []
    for channel in range(3):
        counts, _ = np.histogram(hsv[..., channel], bins=cfg.hsv_bins, range=(0, 1))
        color.extend(counts / max(counts.sum(), 1))
    return np.concatenate([shape, color]).astype(np.float32)


def learn_templates(train_records, cfg):
    """K-means on log(width,height), using training annotations only."""
    sizes = [r.boxes[:, 2:] for r in train_records if len(r.boxes)]
    if not sizes:
        raise ValueError("No positive training boxes found")
    sizes = np.concatenate(sizes)
    unique = np.unique(np.maximum(sizes, 2), axis=0)
    n = min(cfg.n_templates, len(unique))
    if n < 1:
        raise ValueError("n_templates must be positive")
    centers = KMeans(n_clusters=n, random_state=cfg.seed, n_init=10).fit(
        np.log(np.maximum(sizes, 2))).cluster_centers_
    templates = np.unique(np.maximum(2, np.rint(np.exp(centers))).astype(int), axis=0)
    return sorted([tuple(map(int, wh)) for wh in templates], key=lambda wh: wh[0] * wh[1])


def _axis_positions(length, window, step):
    if window > length:
        return []
    values = list(range(0, length - window + 1, step))
    if values[-1] != length - window:
        values.append(length - window)
    return values


def iter_windows(width, height, templates, cfg):
    """Multiscale windows in original pixels; also include right/bottom edges."""
    for w, h in templates:
        sx = max(cfg.min_stride, int(round(w * cfg.stride_fraction)))
        sy = max(cfg.min_stride, int(round(h * cfg.stride_fraction)))
        for y in _axis_positions(height, h, sy):
            for x in _axis_positions(width, w, sx):
                yield (x, y, w, h)


def make_training_features(train_records, templates, cfg, progress=True):
    """Return paired HOG / HSV features from exactly the same crop samples."""
    rng = np.random.default_rng(cfg.seed)
    positives = [(i, j) for i, r in enumerate(train_records) for j in range(len(r.boxes))]
    if len(positives) > cfg.max_positives:
        keep = rng.choice(len(positives), cfg.max_positives, replace=False)
        positives = [positives[i] for i in keep]
    chosen = {}
    for i, j in positives:
        chosen.setdefault(i, []).append(j)
    features, labels, audit = [], [], []
    for index, r in enumerate(tqdm(train_records, desc="Training crops", disable=not progress)):
        image = read_rgb(r)
        for j in chosen.get(index, []):
            features.append(extract_features(crop_rgb(image, r.boxes[j]), cfg, True))
            labels.append(1)
        accepted = 0
        # Bound attempts in dense scenes where clean background is scarce.
        for _ in range(cfg.negatives_per_image * 80):
            if accepted >= cfg.negatives_per_image:
                break
            w, h = templates[int(rng.integers(len(templates)))]
            if w > r.width or h > r.height:
                continue
            box = (int(rng.integers(r.width - w + 1)),
                   int(rng.integers(r.height - h + 1)), w, h)
            if is_background(box, r.boxes, cfg):
                features.append(extract_features(crop_rgb(image, box), cfg, True))
                labels.append(0)
                accepted += 1
        audit.append({"image_id": r.path.stem, "positives": len(chosen.get(index, [])),
                      "negatives": accepted})
    if len(set(labels)) != 2:
        raise ValueError("Training requires both positive and background crops")
    all_features = np.stack(features)
    return {"hog": all_features[:, :-3 * cfg.hsv_bins].copy(),
            "hog_color": all_features}, np.asarray(labels, dtype=np.int8), audit


def fit_model(X, y, templates, cfg, use_color=False):
    """Fit scaling and squared-hinge linear SVM exclusively on training crops."""
    start = perf_counter()
    model = make_pipeline(StandardScaler(), LinearSVC(
        C=cfg.C, class_weight="balanced", dual="auto", max_iter=20000,
        random_state=cfg.seed))
    model.fit(X, y)
    elapsed = perf_counter() - start
    return {"pipeline": model, "templates": templates, "config": asdict(cfg),
            "use_color": bool(use_color), "fit_seconds": elapsed,
            "training_samples": int(len(y))}


def scan_image(image, bundle, score_floor=None):
    """Batch feature extraction/scoring; returns xywh boxes and raw SVM scores.

    score_floor=-inf retains all windows for validation postprocessing sweeps.
    This deliberately uses exact per-crop HOG rather than a different dense-map
    approximation. Small windows / strides can make scanning expensive.
    """
    cfg = Config(**bundle["config"])
    floor = cfg.score_floor if score_floor is None else score_floor
    features, boxes, kept_boxes, kept_scores = [], [], [], []

    def flush():
        if not features:
            return
        scores = bundle["pipeline"].decision_function(np.stack(features))
        keep = np.asarray(scores) >= floor
        if np.any(keep):
            kept_boxes.extend(np.asarray(boxes)[keep])
            kept_scores.extend(scores[keep])
        features.clear()
        boxes.clear()

    height, width = image.shape[:2]
    for box in iter_windows(width, height, bundle["templates"], cfg):
        features.append(extract_features(crop_rgb(image, box), cfg, bundle["use_color"]))
        boxes.append(box)
        if len(features) >= cfg.batch_size:
            flush()
    flush()
    return (np.asarray(kept_boxes, dtype=float).reshape(-1, 4),
            np.asarray(kept_scores, dtype=float))


def nms(boxes, scores, iou_threshold=0.3, max_detections=300):
    """Greedy score-ordered suppression, returning indices in descending score."""
    boxes, scores = np.asarray(boxes).reshape(-1, 4), np.asarray(scores)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while len(order) and len(keep) < max_detections:
        index = int(order[0])
        keep.append(index)
        rest = order[1:]
        order = rest[iou_one_to_many(boxes[index], boxes[rest]) <= iou_threshold]
    return np.asarray(keep, dtype=int)


def scan_image_many(image, bundles, score_floor=-2.0):
    """Score several classifiers on identical window features computed once.

    This changes computation sharing, not feature values or model decisions.
    All models must have identical feature and search geometry. Returned raw
    candidates obey each model's pre-NMS cap, as in scan_records.
    """
    if not bundles:
        return {}
    first = next(iter(bundles.values()))
    cfg = Config(**first["config"])
    geometry = ("patch_size", "orientations", "cell_size", "block_cells",
                "hsv_bins", "stride_fraction", "min_stride")
    for bundle in bundles.values():
        if bundle["templates"] != first["templates"] or any(
                bundle["config"][key] != first["config"][key] for key in geometry):
            raise ValueError("Shared scanning requires identical feature/search geometry")
    color = any(b["use_color"] for b in bundles.values())
    features, boxes = [], []
    outputs = {name: ([], []) for name in bundles}

    def flush():
        if not features:
            return
        X = np.stack(features)
        all_boxes = np.asarray(boxes)
        for name, bundle in bundles.items():
            values = X if bundle["use_color"] or not color else X[:, :-3 * cfg.hsv_bins]
            scores = bundle["pipeline"].decision_function(values)
            mask = scores >= score_floor
            outputs[name][0].extend(all_boxes[mask])
            outputs[name][1].extend(scores[mask])
        features.clear()
        boxes.clear()

    height, width = image.shape[:2]
    for box in iter_windows(width, height, first["templates"], cfg):
        features.append(extract_features(crop_rgb(image, box), cfg, color))
        boxes.append(box)
        if len(features) >= cfg.batch_size:
            flush()
    flush()
    result = {}
    for name, (found_boxes, found_scores) in outputs.items():
        found_boxes = np.asarray(found_boxes, dtype=float).reshape(-1, 4)
        found_scores = np.asarray(found_scores, dtype=float)
        order = np.argsort(-found_scores, kind="stable")[:bundles[name]["config"]["pre_nms_topk"]]
        result[name] = (found_boxes[order], found_scores[order])
    return result


def postprocess(boxes, scores, cfg):
    boxes, scores = np.asarray(boxes).reshape(-1, 4), np.asarray(scores)
    order = np.argsort(-scores, kind="stable")
    order = order[scores[order] >= cfg.score_floor][:cfg.pre_nms_topk]
    boxes, scores = boxes[order], scores[order]
    keep = nms(boxes, scores, cfg.nms_iou, cfg.max_detections)
    return boxes[keep], scores[keep]


def scan_records(records, bundle, score_floor=None, progress=True):
    """Cache at most pre_nms_topk candidates/image for postprocessing sweeps."""
    raw, times = {}, []
    cfg = Config(**bundle["config"])
    for r in tqdm(records, desc="Scanning images", disable=not progress):
        start = perf_counter()
        image = read_rgb(r)
        boxes, scores = scan_image(image, bundle, score_floor)
        order = np.argsort(-scores, kind="stable")[:cfg.pre_nms_topk]
        boxes, scores = boxes[order], scores[order]
        times.append(perf_counter() - start)
        raw[r.path.stem] = (boxes, scores)
    return raw, times


def predictions_from_raw(raw, cfg):
    predictions = []
    for image_id, (boxes, scores) in raw.items():
        boxes, scores = postprocess(boxes, scores, cfg)
        predictions.extend({"image_id": image_id, "bbox": box.tolist(), "score": float(score)}
                           for box, score in zip(boxes, scores))
    return predictions


def predict_records(records, bundle, progress=True):
    """End-to-end per-image timings include load, features, scoring and NMS."""
    cfg = Config(**bundle["config"])
    predictions, times = [], []
    for r in tqdm(records, desc="Detecting birds", disable=not progress):
        start = perf_counter()
        boxes, scores = scan_image(read_rgb(r), bundle)
        boxes, scores = postprocess(boxes, scores, cfg)
        rows = [{"image_id": r.path.stem, "bbox": b.tolist(), "score": float(s)}
                for b, s in zip(boxes, scores)]
        times.append(perf_counter() - start)
        predictions.extend(rows)
    return predictions, times


def mine_hard_negatives(train_records, bundle, progress=True):
    """Mine only unambiguous background from TRAIN images, then extract features."""
    cfg = Config(**bundle["config"])
    features, audit = [], []
    for r in tqdm(train_records, desc="Hard-negative mining", disable=not progress):
        if len(features) >= cfg.max_hard_negatives:
            break
        image = read_rgb(r)
        boxes, scores = scan_image(image, bundle, score_floor=0.0)
        indices = [i for i in range(len(boxes)) if is_background(boxes[i], r.boxes, cfg)]
        selected = nms(boxes[indices], scores[indices], 0.3, cfg.hard_negatives_per_image)
        for j in selected:
            if len(features) >= cfg.max_hard_negatives:
                break
            box = boxes[indices[j]]
            features.append(extract_features(crop_rgb(image, box), cfg, bundle["use_color"]))
            audit.append({"image_id": r.path.stem, "bbox": box.tolist(),
                          "score": float(scores[indices[j]])})
    dim = bundle["pipeline"].n_features_in_
    return (np.stack(features) if features else np.empty((0, dim), dtype=np.float32)), audit


def coco_ground_truth(records):
    """COCO GT for exactly the evaluated image subset, including negative images."""
    images, annotations = [], []
    for index, r in enumerate(records, start=1):
        images.append({"id": index, "file_name": r.path.name,
                       "width": r.width, "height": r.height})
        for box in r.boxes:
            annotations.append({"id": len(annotations) + 1, "image_id": index,
                                "category_id": 1, "bbox": box.tolist(),
                                "area": float(box[2] * box[3]), "iscrowd": 0})
    return {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "bird"}]}


def evaluate_predictions(predictions, records, workspace, tag="evaluation"):
    """Use the team's evaluator against GT for exactly these records.

    Temporarily redirects GT_DIR and restores it even if evaluation raises.
    This helper is intended for sequential notebook use, not concurrent calls.
    """
    from evaluation import evaluate as shared
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    gt = coco_ground_truth(records)
    (workspace / "val_gt.json").write_text(json.dumps(gt))
    pred_path = workspace / f"{tag}.json"
    pred_path.write_text(json.dumps(predictions, allow_nan=False))
    original = shared.GT_DIR
    try:
        shared.GT_DIR = workspace
        return shared.evaluate(pred_path, split="val", verbose=False)
    finally:
        shared.GT_DIR = original


def proposal_recall(records, templates, cfg, threshold=0.5):
    """Fraction of GT boxes covered by ANY searched window before classification."""
    hits, total = 0, 0
    for r in records:
        windows = np.asarray(list(iter_windows(r.width, r.height, templates, cfg)),
                             dtype=float).reshape(-1, 4)
        best = np.asarray([np.max(iou_one_to_many(box, windows), initial=0.0)
                           for box in r.boxes])
        hits += int(np.sum(best >= threshold))
        total += len(best)
    return {"recall": hits / total if total else float("nan"),
            "covered_boxes": hits, "total_boxes": total}


def environment_info():
    from importlib.metadata import version
    names = ["numpy", "Pillow", "scikit-image", "scikit-learn", "pycocotools", "joblib"]
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "packages": {n: version(n) for n in names}}


def save_bundle(bundle, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def cli():
    import argparse
    from utils.data import get_data_root
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Trusted local .joblib model")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Process only N images, e.g. for a quick demo")
    args = parser.parse_args()
    records = load_records(args.data_root or get_data_root(), args.split)
    records = subset(records, args.limit)
    bundle = joblib.load(args.model)
    predictions, times = predict_records(records, bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(predictions, allow_nan=False))
    timing = {"images": len(times), "mean_seconds": float(np.mean(times)),
              "median_seconds": float(np.median(times)), "environment": environment_info()}
    args.output.with_suffix(".timing.json").write_text(json.dumps(timing, indent=2))
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    cli()
