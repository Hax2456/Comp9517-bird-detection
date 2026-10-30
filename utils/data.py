"""Shared data-loading utilities used by every method."""

from pathlib import Path 

import yaml 
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
def get_data_root() -> Path: 
    """Read the dataset root directory from configs/paths.yaml."""
    cfg_path = REPO_ROOT / "configs" / "paths.yaml"
    with open(cfg_path, encoding="utf-8") as f: 
        cfg = yaml.safe_load(f)
    root = Path(cfg["data_root"])
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root


def list_images(split: str) -> list[Path]:

    img_dir= get_data_root() / "images" / split
    return sorted(img_dir.glob("*.jpg"))

def label_path_for(img_path:Path) -> Path: 

    split = img_path.parent.name
    return get_data_root() / "labels" / split / (img_path.stem + ".txt")


def read_yolo_boxes(label_path: Path, img_w: int, img_h: int) -> list[list[float]]:
    """Read a YOLO label file and return pixel boxes as [x, y, w, h] (top-left corner + size)."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:  # skip empty or malformed lines
            continue
        _, xc, yc, w, h = map(float, parts)
        bw, bh = w * img_w, h * img_h                      # normalised size -> pixels
        x, y = xc * img_w - bw / 2, yc * img_h - bh / 2    # centre -> top-left corner
        # Clip to the image so boxes never extend past the border
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(img_w), x + bw), min(float(img_h), y + bh)
        boxes.append([x0, y0, x1 - x0, y1 - y0])
    return boxes


def load_split(split: str) -> list[tuple[Path, tuple[int, int], list[list[float]]]]:
    """Load a whole split as [(image_path, (width, height), boxes), ...]."""
    items = []
    for img_path in list_images(split):
        with Image.open(img_path) as im:
            w, h = im.size
        items.append((img_path, (w, h), read_yolo_boxes(label_path_for(img_path), w, h)))
    return items