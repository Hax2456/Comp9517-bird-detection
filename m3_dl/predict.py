"""Run an Ultralytics model (YOLO11 / RT-DETR) on one split and save predictions in the shared JSON format.

Usage (run from the repository root):
    python -m m3_dl.predict --weights path/to/best.pt --split val --name yolo11_default
"""
import argparse
import json
import time

from ultralytics import YOLO

from utils.data import REPO_ROOT, list_images


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
    args = parser.parse_args()

    model = YOLO(args.weights)
    images = list_images(args.split)

    # Warm-up: the first call on the GPU is slow, so run it once before timing
    model.predict(str(images[0]), imgsz=args.imgsz, verbose=False)

    preds = []
    start = time.perf_counter()
    for img_path in images:
        result = model.predict(str(img_path), imgsz=args.imgsz, conf=0.001, max_det=300, verbose=False)[0]
        preds.extend(result_to_preds(result, img_path.stem))
    elapsed = time.perf_counter() - start

    out_path = REPO_ROOT / "preds" / f"{args.name}_{args.split}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(preds, f)
    print(f"Wrote {len(preds)} boxes for {len(images)} images -> {out_path}")
    print(f"Inference time: {1000 * elapsed / len(images):.1f} ms per image (end to end)")


if __name__ == "__main__":
    main()