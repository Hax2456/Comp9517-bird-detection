"""Convert YOLO labels into COCO-format ground truth for the evaluation script.

Usage (run from the repository root):
    python -m evaluation.make_coco_gt
"""
import json

from utils.data import REPO_ROOT, load_split

OUT_DIR = REPO_ROOT / "data" / "processed"


def build_coco_gt(split: str) -> dict:
    """Build a COCO ground-truth dict for one split (train / val / test)."""
    images, annotations = [], []
    ann_id = 1
    for img_id, (img_path, (w, h), boxes) in enumerate(load_split(split), start=1):
        images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
        for x, y, bw, bh in boxes:
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2),
                "iscrowd": 0,
            })
            ann_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "bird"}],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        coco = build_coco_gt(split)
        out_path = OUT_DIR / f"{split}_gt.json"
        with open(out_path, "w") as f:
            json.dump(coco, f)
        print(f"{split}: {len(coco['images'])} images, {len(coco['annotations'])} boxes -> {out_path}")


if __name__ == "__main__":
    main()