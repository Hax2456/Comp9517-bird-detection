"""Statistics behind the Discussion section (dataset, M1 diagnostics, overlap, M3 examples).

Usage (run from the repository root, after m1_classical.detect has written its test predictions):
    python -m analysis.discussion_stats
"""
import contextlib
import io
import json
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.data import REPO_ROOT

SMALL, MEDIUM = 32 ** 2, 96 ** 2
DENSITY_BINS = [(0, 0), (1, 5), (6, 20), (21, 50), (51, 10 ** 6)]


def iou(a, b):
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def size_bin(box):
    area = box[2] * box[3]
    return "small" if area < SMALL else ("medium" if area < MEDIUM else "large")


def load_gt(split):
    path = REPO_ROOT / "data" / "processed" / f"{split}_gt.json"
    gt = json.load(open(path))
    stem = {im["id"]: im["file_name"].rsplit(".", 1)[0] for im in gt["images"]}
    boxes = defaultdict(list)
    for a in gt["annotations"]:
        boxes[stem[a["image_id"]]].append(a["bbox"])
    return path, stem, [stem[i] for i in sorted(stem)], boxes


def load_preds(path):
    preds = defaultdict(list)
    for p in json.load(open(path)):
        preds[p["image_id"]].append(p["bbox"] + [p["score"]])
    return preds


def match(gts, preds, iou_thr):
    """Greedy matching by score. Returns matched-gt flags and per-pred TP flags."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i][4])
    gt_used = [False] * len(gts)
    pred_tp = [False] * len(preds)
    for i in order:
        best, best_j = iou_thr, -1
        for j, g in enumerate(gts):
            if not gt_used[j]:
                v = iou(preds[i], g)
                if v >= best:
                    best, best_j = v, j
        if best_j >= 0:
            gt_used[best_j] = True
            pred_tp[i] = True
    return gt_used, pred_tp


def ap_at_iou(gt_path, stem, pred_path, iou_thrs):
    """COCO AP at custom IoU thresholds, for all/small/medium/large (300 dets)."""
    id_of = {s: i for i, s in stem.items()}
    dets = [{"image_id": id_of[p["image_id"]], "category_id": 1, "bbox": p["bbox"], "score": p["score"]}
            for p in json.load(open(pred_path))]
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(dets)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.iouThrs = np.array(iou_thrs)
        ev.params.maxDets = [1, 10, 300]
        ev.evaluate()
        ev.accumulate()
    prec = ev.eval["precision"]  # [T, R, K, A, M]
    out = {}
    for t, thr in enumerate(iou_thrs):
        row = []
        for a in range(4):  # all, small, medium, large
            p = prec[t, :, 0, a, -1]
            p = p[p > -1]
            row.append(float(p.mean()) if p.size else float("nan"))
        out[thr] = row
    return out


def auc(pos, neg):
    """Probability that a random TP box scores higher than a random FP box."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    # average ranks for ties
    for v in np.unique(allv):
        m = allv == v
        ranks[m] = ranks[m].mean()
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def dataset_stats():
    print("=== Dataset ===")
    for split in ("train", "val", "test"):
        _, _, stems, gts = load_gt(split)
        all_boxes = [b for s in stems for b in gts[s]]
        sizes = [size_bin(b) for b in all_boxes]
        sides = np.sqrt([b[2] * b[3] for b in all_boxes])
        print(f"{split}: {len(stems)} imgs, {len(all_boxes)} birds, "
              f"{sum(len(gts[s]) == 0 for s in stems)} empty imgs, "
              f"small {sizes.count('small') / len(sizes):.1%} medium {sizes.count('medium') / len(sizes):.1%} "
              f"large {sizes.count('large') / len(sizes):.1%}, median side {np.median(sides):.1f}px, "
              f"birds<10px side {np.mean(sides < 10):.1%}")
    _, _, stems, gts = load_gt("test")
    print("test density bins: imgs / birds / median side / small share")
    for lo, hi in DENSITY_BINS:
        sel = [s for s in stems if lo <= len(gts[s]) <= hi]
        boxes = [b for s in sel for b in gts[s]]
        if boxes:
            sides = np.sqrt([b[2] * b[3] for b in boxes])
            small = np.mean([size_bin(b) == "small" for b in boxes])
            print(f"  {lo}-{hi}: {len(sel)} / {len(boxes)} / {np.median(sides):.1f}px / {small:.1%}")
        else:
            print(f"  {lo}-{hi}: {len(sel)} / 0")
    # how much do neighbouring birds overlap? (upper bound on what NMS could remove)
    max_iou = []
    for s in stems:
        g = gts[s]
        for i in range(len(g)):
            m = max([iou(g[i], g[j]) for j in range(len(g)) if j != i], default=0.0)
            max_iou.append(m)
    max_iou = np.array(max_iou)
    print("test GT neighbour overlap (max IoU with another bird): "
          + ", ".join(f">{t}: {int((max_iou > t).sum())} birds ({np.mean(max_iou > t):.2%})"
                      for t in (0.0, 0.1, 0.3, 0.5, 0.7)))


def m1_stats():
    gt_path, stem, stems, gts = load_gt("test")
    for variant in ("threshold", "blob"):
        pred_path = REPO_ROOT / "preds" / f"m1_{variant}_test.json"
        preds = load_preds(pred_path)
        print(f"=== M1 {variant} (test, all boxes) ===")
        n_boxes = [len(preds[s]) for s in stems]
        print(f"boxes/img mean {np.mean(n_boxes):.1f}, imgs at 120-box cap {sum(n == 120 for n in n_boxes)}/{len(stems)}")
        for thr in (0.5, 0.3, 0.1):
            found = defaultdict(int)
            total = defaultdict(int)
            tp_scores, fp_scores = [], []
            for s in stems:
                used, tp = match(gts[s], preds[s], thr)
                for g, u in zip(gts[s], used):
                    total[size_bin(g)] += 1
                    found[size_bin(g)] += u
                for p, t in zip(preds[s], tp):
                    (tp_scores if t else fp_scores).append(p[4])
            n_gt = sum(total.values())
            print(f"IoU>={thr}: recall {sum(found.values()) / n_gt:.3f} "
                  f"(small {found['small'] / total['small']:.3f}, medium {found['medium'] / total['medium']:.3f}, "
                  f"large {found['large'] / max(1, total['large']):.3f}); "
                  f"TP share of boxes {len(tp_scores) / (len(tp_scores) + len(fp_scores)):.4f}; "
                  f"score AUC TP-vs-FP {auc(tp_scores, fp_scores):.3f}")
        aps = ap_at_iou(gt_path, stem, pred_path, [0.1, 0.3, 0.5])
        for thr, row in aps.items():
            print(f"AP@{thr}: all {row[0]:.4f} small {row[1]:.4f} medium {row[2]:.4f} large {row[3]:.4f}")


def m3_example_stats():
    print("=== M3 on the four figure images (val-best thresholds) ===")
    _, _, _, gts = load_gt("test")
    m3 = json.load(open(REPO_ROOT / "analysis" / "m3_selected_test.json"))
    for name in ("m3_yolo11s_default", "m3_rtdetr_l"):
        thr = m3[name]["thr"]
        for s, boxes in m3[name]["boxes"].items():
            g = gts[s]
            used, tp = match(g, boxes, 0.5)
            fp = [b for b, t in zip(boxes, tp) if not t]
            fp_type = defaultdict(int)
            for b in fp:
                m = max([iou(b, x) for x in g], default=0.0)
                fp_type["dup/loc>=0.5" if m >= 0.5 else ("loc 0.1-0.5" if m >= 0.1 else "background<0.1")] += 1
            fn = [x for x, u in zip(g, used) if not u]
            fn_type = defaultdict(int)
            for x in fn:
                m = max([iou(x, b) for b in boxes], default=0.0)
                fn_type["near miss 0.1-0.5" if m >= 0.1 else "unseen<0.1"] += 1
            print(f"{name} {s} (thr {thr}): GT {len(g)} TP {sum(tp)} FP {len(fp)} {dict(fp_type)} "
                  f"FN {len(fn)} {dict(fn_type)}")


if __name__ == "__main__":
    dataset_stats()
    m1_stats()
    m3_example_stats()
