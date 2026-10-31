"""Write the ground truth as a prediction file to sanity-check the evaluation script.

    python -m evaluation.make_gt_preds --split val              # unchanged: mAP should be 1.0
    python -m evaluation.make_gt_preds --split val --shift 5    # shift every box 5 px to the right
"""
import argparse
import json

from utils.data import REPO_ROOT, load_split


def main():
    parser = argparse.ArgumentParser(description="Turn ground-truth boxes into a prediction file.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--shift", type=float, default=0.0, help="shift every box right by this many pixels")
    args = parser.parse_args()

    preds = []
    for img_path, _, boxes in load_split(args.split):
        for x, y, w, h in boxes:
            preds.append({
                "image_id": img_path.stem,
                "bbox": [x + args.shift, y, w, h],
                "score": 1.0,
            })

    out_dir = REPO_ROOT / "preds"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"gt_shift{args.shift:g}_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(preds, f)
    print(f"Wrote {len(preds)} boxes -> {out_path}")


if __name__ == "__main__":
    main()