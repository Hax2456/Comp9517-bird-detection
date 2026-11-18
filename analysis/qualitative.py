"""Draw the success / failure examples used in the report and the video.

Usage (run from the repository root):
    python -m analysis.qualitative

Rows are four representative test images, columns are ground truth, M1 (threshold),
M3 YOLO11s and M3 RT-DETR-L. Each detector is shown at the confidence threshold that
gives the best F1 on the validation split. Blue = correct detection (IoU >= 0.5),
red = false positive, yellow = missed bird.

M1 boxes are read from preds/m1_threshold_test.json (run m1_classical.detect first).
M3 boxes for these four images are stored in analysis/m3_selected_test.json, exported
from the full M3 prediction files, which are too large to keep in the repository.
"""
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from utils.data import REPO_ROOT, get_data_root

IMAGES = [  # (file stem, row label)
    ("beach_bird_1734", "Success: sparse,\nwell-separated birds"),
    ("beach_bird_1818", "Failure: very small\nbirds (~8 px)"),
    ("beach_bird_1661", "Failure: cluttered\nbackground"),
    ("beach_bird_1790", "Failure: dense\nflying flock"),
]
COLOURS = {"tp": "#2a78d6", "fp": "#e34948", "fn": "#eda100"}
IOU_THR = 0.5


def load_gt(split):
    """Ground-truth boxes per image stem, from the shared COCO file."""
    gt = json.load(open(REPO_ROOT / "data" / "processed" / f"{split}_gt.json"))
    stem = {im["id"]: im["file_name"].rsplit(".", 1)[0] for im in gt["images"]}
    boxes = defaultdict(list)
    for ann in gt["annotations"]:
        boxes[stem[ann["image_id"]]].append(ann["bbox"])
    return list(stem.values()), boxes


def load_preds(path):
    """Predicted boxes per image stem as [x, y, w, h, score]."""
    boxes = defaultdict(list)
    for p in json.load(open(path)):
        boxes[p["image_id"]].append(p["bbox"] + [p["score"]])
    return boxes


def iou(a, b):
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def match(gts, preds, thr):
    """Greedy matching by score. Returns (tp boxes, fp boxes, missed gt boxes)."""
    preds = sorted([p for p in preds if p[4] >= thr], key=lambda p: -p[4])
    used, tp, fp = set(), [], []
    for p in preds:
        best, best_i = IOU_THR, -1
        for i, g in enumerate(gts):
            if i not in used and iou(p, g) >= best:
                best, best_i = iou(p, g), i
        if best_i >= 0:
            used.add(best_i)
            tp.append(p)
        else:
            fp.append(p)
    fn = [g for i, g in enumerate(gts) if i not in used]
    return tp, fp, fn


def best_f1_threshold(stems, gts, preds, candidates):
    """Confidence threshold with the highest F1 over a split."""
    best = (-1.0, None)
    for thr in candidates:
        n_tp = n_fp = n_fn = 0
        for s in stems:
            tp, fp, fn = match(gts[s], preds[s], thr)
            n_tp, n_fp, n_fn = n_tp + len(tp), n_fp + len(fp), n_fn + len(fn)
        f1 = 2 * n_tp / max(1, 2 * n_tp + n_fp + n_fn)
        best = max(best, (f1, thr))
    return best[1], best[0]


def draw(ax, img, boxes_by_kind, title=None):
    ax.imshow(img)
    for kind, boxes in boxes_by_kind.items():
        for b in boxes:
            ax.add_patch(Rectangle((b[0], b[1]), b[2], b[3], fill=False,
                                   edgecolor=COLOURS[kind], linewidth=0.8))
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def main():
    # M1: pick its display threshold on the validation split
    val_stems, val_gt = load_gt("val")
    m1_val = load_preds(REPO_ROOT / "preds" / "m1_threshold_val.json")
    scores = np.array([p[4] for ps in m1_val.values() for p in ps])
    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 0.99, 40)))
    m1_thr, m1_f1 = best_f1_threshold(val_stems, val_gt, m1_val, candidates)
    print(f"M1 threshold: display threshold {m1_thr:.3f} (val F1 {m1_f1:.3f})")

    _, test_gt = load_gt("test")
    m1_test = load_preds(REPO_ROOT / "preds" / "m1_threshold_test.json")
    m3 = json.load(open(REPO_ROOT / "analysis" / "m3_selected_test.json"))
    methods = [  # (column title, boxes per image, threshold)
        (f"M1 threshold (s≥{m1_thr:.2f})", m1_test, m1_thr),
        (f"M3 YOLO11s (s≥{m3['m3_yolo11s_default']['thr']})",
         m3["m3_yolo11s_default"]["boxes"], m3["m3_yolo11s_default"]["thr"]),
        (f"M3 RT-DETR-L (s≥{m3['m3_rtdetr_l']['thr']})",
         m3["m3_rtdetr_l"]["boxes"], m3["m3_rtdetr_l"]["thr"]),
    ]

    img_dir = get_data_root() / "images" / "test"
    fig, axes = plt.subplots(len(IMAGES), 4, figsize=(7.2, 7.6))
    for r, (stem, label) in enumerate(IMAGES):
        img = Image.open(img_dir / f"{stem}.jpg").convert("RGB")
        gts = test_gt[stem]
        draw(axes[r, 0], img, {"tp": gts}, "Ground truth" if r == 0 else None)
        axes[r, 0].set_ylabel(f"{label}\n({len(gts)} birds)", fontsize=7)
        for c, (title, boxes, thr) in enumerate(methods, start=1):
            tp, fp, fn = match(gts, boxes.get(stem, []), thr)
            draw(axes[r, c], img, {"fn": fn, "fp": fp, "tp": tp}, title if r == 0 else None)
            axes[r, c].set_xlabel(f"TP {len(tp)}  FP {len(fp)}  FN {len(fn)}", fontsize=7)
            print(f"{stem} {title}: TP {len(tp)} FP {len(fp)} FN {len(fn)}")

    handles = [Rectangle((0, 0), 1, 1, fill=False, edgecolor=COLOURS[k], linewidth=1.5)
               for k in ("tp", "fp", "fn")]
    fig.legend(handles, ["Correct (IoU ≥ 0.5)", "False positive", "Missed bird"],
               loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out_dir = REPO_ROOT / "analysis" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    # JPEG keeps the photo panels small enough for the 10 MB report limit
    out_path = out_dir / "qualitative_examples.jpg"
    fig.savefig(out_path, dpi=250, pil_kwargs={"quality": 88})
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
