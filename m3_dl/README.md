# M3: deep learning (YOLO11s and RT-DETR-L)

M3 fine-tunes two COCO-pretrained Ultralytics detectors on the beach-bird training split:
a one-stage CNN detector (YOLO11s) and a transformer detector (RT-DETR-L).
A third run repeats YOLO11s with mosaic augmentation switched off, as an ablation.

## Files

| File | Purpose |
|---|---|
| `train.py` | Trains one run with the exact settings used for the reported results |
| `predict.py` | Runs a trained model on a split and writes predictions in the shared JSON format |
| `plot_training.py` | Draws the validation-mAP training curves from the logs |
| `training_logs/<run>/args.yaml` | Full Ultralytics configuration of each reported run |
| `training_logs/<run>/results.csv` | Per-epoch losses and validation metrics of each reported run |
| `training_logs/m3_training_curves.png` | Report figure produced by `plot_training.py` |

Model weights (`best.pt`) are not committed because of their size.

## Training setup

All runs: 60 epochs, input size 512, seed 0, deterministic mode, COCO-pretrained
initialisation, Ultralytics defaults for every other hyperparameter
(optimizer `auto`, `lr0=0.01`, `close_mosaic=10`, AMP on).
Hardware: one NVIDIA RTX 4090 (RunPod), AMD Ryzen 9 7950X CPU.

| Run | Weights | Batch | Mosaic |
|---|---|---|---|
| `yolo11s_default` | `yolo11s.pt` | 16 | 1.0 |
| `yolo11s_nomosaic` | `yolo11s.pt` | 16 | 0.0 |
| `rtdetr_l` | `rtdetr-l.pt` | 8 | 1.0 |

## Reproducing

Run from the repository root on a GPU machine, with `configs/paths.yaml` set up
and `requirements-dl.txt` installed:

```bash
python -m m3_dl.train --run rtdetr_l          # also: yolo11s_default, yolo11s_nomosaic
python -m m3_dl.predict --family rtdetr --weights runs/m3/rtdetr_l/weights/best.pt \
  --split val --name m3_rtdetr_l --imgsz 512
python -m evaluation.evaluate --pred preds/m3_rtdetr_l_val.json --split val
python -m m3_dl.plot_training                 # redraw the training-curve figure
```

`train.py` generates the Ultralytics dataset file from `configs/paths.yaml`,
so no paths need to be edited.

## Results

Scored with the shared evaluator (`evaluation/evaluate.py`, pycocotools, 300 detections per image),
on the original validation and test splits. Model selection used validation only; the test split
was evaluated once, after all settings were frozen.

| Run | Split | mAP@0.5 | mAP@0.5:0.95 | AP50 small | AP50 medium | AP50 large |
|---|---|---|---|---|---|---|
| RT-DETR-L | val | 0.891 | 0.501 | 0.823 | 0.910 | 0.921 |
| RT-DETR-L | test | 0.901 | 0.518 | 0.823 | 0.942 | 0.951 |
| YOLO11s | val | 0.853 | 0.463 | 0.734 | 0.899 | 0.910 |
| YOLO11s | test | 0.860 | 0.474 | 0.762 | 0.910 | 0.927 |
| YOLO11s no-mosaic | val | 0.840 | 0.452 | 0.693 | 0.890 | 0.925 |
| YOLO11s no-mosaic | test | 0.865 | 0.465 | 0.788 | 0.912 | 0.920 |

Training time is the wall-clock time in `results.csv`. Inference time is measured by `predict.py`
on the test split (one RTX 4090, FP32, batch 1, one warm-up image): end to end includes image
loading, pre-processing, the network and post-processing; model only is Ultralytics' network time.

| Run | Training | Inference, end to end | Inference, model only |
|---|---|---|---|
| RT-DETR-L | 17.7 min | 29.6 ms/img | 25.0 ms/img |
| YOLO11s | 8.1 min | 9.9 ms/img | 6.4 ms/img |
| YOLO11s no-mosaic | 3.2 min | 10.7 ms/img | 6.9 ms/img |

Software: Ultralytics 8.4.215, PyTorch 2.8.0 (CUDA 12.8).

Ultralytics' own per-epoch validation numbers in `results.csv` differ slightly from these,
because Ultralytics uses its own AP implementation; the report uses the shared evaluator
for every method.

Note: in `rtdetr_l/results.csv` the validation losses are `nan` in many epochs while
precision, recall and mAP stay normal. The `nan` appears only in the logged loss
values, not in the detections.
