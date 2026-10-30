# COMP9517 26T3 Group Project: Beach Bird Detection

Bird detection in beach and coastal images using the **Beach Bird Detection Dataset v1.0** (single class: `bird`).
We implement and compare three families of methods:

| Method | Category | Approach | Folder |
| --- | --- | --- | --- |
| M1 | Non-learning | Adaptive thresholding + morphology; multi-scale LoG blob detection | `m1_classical/` |
| M2 | Traditional machine learning | Sliding window + HOG / colour features + linear SVM | `m2_ml/` |
| M3 | Deep learning | YOLO11 and RT-DETR (Ultralytics) | `m3_dl/` |

Every method writes its predictions in the same JSON format, and all methods are scored by one shared
evaluation script. The main metric is **mAP@0.5**; we also report training and inference time.

> **Status:** work in progress. Items marked *(planned)* are not implemented yet.

## Repository structure

```
Comp9517-bird-detection/
├── configs/
│   ├── paths.example.yaml   # template for the dataset location (committed)
│   └── paths.yaml           # your own copy (not committed)
├── utils/
│   └── data.py              # shared data loading: images, YOLO labels, box conversion
├── evaluation/
│   ├── make_coco_gt.py      # converts YOLO labels to COCO ground truth
│   └── evaluate.py          # computes mAP for a prediction file
├── m1_classical/            # M1 non-learning methods (planned)
├── m2_ml/                   # M2 traditional ML (planned)
├── m3_dl/                   # M3 deep learning (planned)
├── analysis/                # density / size analysis, robustness, figures (planned)
├── data/                    # dataset and generated files (not committed)
│   ├── raw/
│   └── processed/
└── preds/                   # prediction files (not committed)
```

## Setup

Requires Python 3.10 or later.

```bash
git clone https://github.com/Hax2456/Comp9517-bird-detection.git
cd Comp9517-bird-detection
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The deep learning models are trained on Google Colab; their extra dependencies will be listed in
`requirements-dl.txt` *(planned)*.

## Dataset

1. Download **Beach Bird Detection Dataset v1.0** from Kaggle (link in the project specification).
2. Unzip it into `data/raw/`, so that you have:
   ```
   data/raw/Beach Bird Detection Dataset v1.0/
   ├── images/{train,val,test}/
   ├── labels/{train,val,test}/
   └── dataset.yaml
   ```
3. Create your local path config:
   ```bash
   cp configs/paths.example.yaml configs/paths.yaml
   ```
   Edit `data_root` in `configs/paths.yaml` if you stored the dataset somewhere else.
   Relative paths are resolved from the repository root.

Dataset summary: 1,369 train / 291 val / 240 test images, all 512 × 512 pixels, with labels in YOLO format
(`class x_center y_center width height`, normalised to 0–1). Images without birds have empty label files.

## Evaluation

Generate the COCO-format ground truth once:

```bash
python -m evaluation.make_coco_gt
```

This writes `data/processed/{train,val,test}_gt.json`.

Evaluate a prediction file *(in progress)*:

```bash
python -m evaluation.evaluate --pred preds/<method>_<variant>_val.json --split val
```

### Prediction format

All methods must output a JSON list with one entry per predicted box:

```json
[
  {"image_id": "beach_bird_0001", "bbox": [x, y, w, h], "score": 0.93}
]
```

- `image_id`: image file name without `.jpg`
- `bbox`: pixel coordinates; `(x, y)` is the top-left corner, `w` and `h` are width and height
- `score`: confidence; higher means more likely to be a bird (required to compute mAP)

Name prediction files `preds/{method}_{variant}_{split}.json`, for example `preds/m1_blob_val.json`.

### Rules for fair comparison

- Tune parameters and select models on `train` and `val` only.
- The `test` split is evaluated once per method, after all methods are frozen.
- Fix random seeds and report the hardware used for timing.

## Running the methods *(planned)*

```bash
python -m m1_classical.detect --variant blob --split val
python -m m2_ml.detect --features hog_color --split val
python m3_dl/predict.py --weights <path/to/best.pt> --split val
```

## Team

| Role | Member | Responsibilities |
| --- | --- | --- |
| A | _TBD_ | Deep learning (M3), data and evaluation pipeline |
| B | _TBD_ | Traditional machine learning (M2) |
| C | _TBD_ | Non-learning methods (M1) |
| D | _TBD_ | Analysis, robustness evaluation, report formatting |

## External libraries and code

| Library | Used for |
| --- | --- |
| NumPy, pandas, Matplotlib, Pillow, PyYAML, tqdm | General data handling and plotting |
| OpenCV, scikit-image | Image processing (M1) |
| scikit-learn | SVM classifier (M2) |
| pycocotools | COCO-style mAP evaluation |
| PyTorch, Ultralytics *(planned)* | YOLO11 and RT-DETR training and inference (M3) |

The dataset is used under the terms in its own `LICENSE.md`.