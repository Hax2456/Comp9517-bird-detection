"""M1: non-learning bird detection (no training).

Two variants, both treating birds as small regions that stand out from the
smooth beach/water background:

  threshold  - background subtraction (large median blur) + Otsu threshold +
               morphology + connected components.
  blob       - multi-scale Laplacian-of-Gaussian blob detection on the same
               background-subtracted response.

Each detection carries a confidence score (how strongly the region deviates
from the local background), which is required to compute mAP. Overlapping
detections are merged with non-maximum suppression.

Usage (run from the repository root):
    python -m m1_classical.detect --variant threshold --split val
    python -m m1_classical.detect --variant blob --split val
"""
import argparse
import json

import cv2
import numpy as np

from utils.data import REPO_ROOT, list_images


# ----------------------------- helpers -----------------------------

def background_response(gray, blur=71):
    """Return an 8-bit image that is bright where a pixel differs from its
    local background. Birds can be darker (on sand) or brighter (white gulls
    on rocks/water), so both directions are kept."""
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    bg = cv2.medianBlur(g, blur)                       # large-scale background
    diff = cv2.absdiff(g, bg)                          # |pixel - background|
    return diff


def nms(boxes, iou_thr=0.3):
    """Non-maximum suppression. boxes: list of [x, y, w, h, score]."""
    if not boxes:
        return []
    b = np.array([[x, y, x + w, y + h, s] for x, y, w, h, s in boxes], dtype=float)
    idx = b[:, 4].argsort()[::-1]
    keep = []
    while len(idx):
        i = idx[0]
        keep.append(i)
        xx1 = np.maximum(b[i, 0], b[idx[1:], 0]); yy1 = np.maximum(b[i, 1], b[idx[1:], 1])
        xx2 = np.minimum(b[i, 2], b[idx[1:], 2]); yy2 = np.minimum(b[i, 3], b[idx[1:], 3])
        w = np.clip(xx2 - xx1, 0, None); h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        a1 = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])
        a2 = (b[idx[1:], 2] - b[idx[1:], 0]) * (b[idx[1:], 3] - b[idx[1:], 1])
        iou = inter / (a1 + a2 - inter + 1e-9)
        idx = idx[1:][iou < iou_thr]
    return [boxes[i] for i in keep]


def _keep(x, y, w, h, H, W):
    """Area / aspect-ratio sanity filter shared by both variants."""
    area = w * h
    if area < 15 or area > H * W * 0.25:   # allow large birds (up to ~25% of image)
        return False
    ar = w / max(h, 1)
    return 0.3 <= ar <= 3.5


# ----------------------------- variants -----------------------------

def detect_threshold(gray, topk=120):
    resp = background_response(gray)
    _, m = cv2.threshold(resp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    H, W = gray.shape
    out = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if not _keep(x, y, w, h, H, W):
            continue
        score = float(resp[y:y + h, x:x + w].mean()) / 255.0
        out.append([int(x), int(y), int(w), int(h), score])
    out = nms(out, 0.3)
    out.sort(key=lambda b: b[4], reverse=True)
    return out[:topk]


def detect_blob(gray, topk=120):
    from skimage.feature import blob_log
    resp = background_response(gray).astype(np.float32)
    if resp.max() > 0:
        resp /= resp.max()
    blobs = blob_log(resp, min_sigma=2, max_sigma=16, num_sigma=5, threshold=0.10)
    H, W = gray.shape
    out = []
    for yc, xc, sig in blobs:
        r = sig * (2 ** 0.5)
        w = h = max(4, int(2 * r))
        x = int(xc - w / 2); y = int(yc - h / 2)
        x = max(0, x); y = max(0, y); w = min(W - x, w); h = min(H - y, h)
        if not _keep(x, y, w, h, H, W):
            continue
        score = float(resp[int(yc), int(xc)])
        out.append([x, y, w, h, score])
    out = nms(out, 0.3)
    out.sort(key=lambda b: b[4], reverse=True)
    return out[:topk]


VARIANTS = {"threshold": detect_threshold, "blob": detect_blob}


# ----------------------------- main -----------------------------

def main():
    parser = argparse.ArgumentParser(description="M1 non-learning bird detector.")
    parser.add_argument("--variant", required=True, choices=list(VARIANTS))
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()

    detfn = VARIANTS[args.variant]
    preds = []
    images = list_images(args.split)
    for img_path in images:
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        for x, y, w, h, score in detfn(gray):
            preds.append({"image_id": img_path.stem, "bbox": [x, y, w, h], "score": score})

    out_dir = REPO_ROOT / "preds"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"m1_{args.variant}_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(preds, f)
    print(f"Wrote {len(preds)} boxes for {len(images)} images -> {out_path}")


if __name__ == "__main__":
    main()
