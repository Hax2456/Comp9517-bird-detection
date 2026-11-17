# Why M3 can score much higher, and what the current folder proves

Reviewed 17 November 2026. The two supplied screenshots are reference material;
their method description informed the M2 retry. Their scores are not measurements
reproduced by this audit.

## 1. What is actually available

The local `m3_dl/` originally contained only `predict.py` and an empty
`__init__.py`. A search including ignored project files found no M3 `.pt`
checkpoints, training notebooks, `args.yaml`, learning curves or prediction files.
`requirements-dl.txt` declares PyTorch/Ultralytics. The README says deep training
is performed on Colab. Thus the inference implementation can be audited, but
the exact training procedure and screenshot scores cannot be reconstructed from
this folder alone.

The supplied table reports:

| Reference model | Reported AP50 | Reported AP50–95 | Reported small-object AP | Training | Inference |
|---|---:|---:|---:|---:|---:|
| RT-DETR-L | 0.891 | 0.501 | 0.823 | 17.7 min | 11 ms |
| YOLO11 default | 0.857 | 0.472 | 0.738 | 8.1 min | 5 ms |
| YOLO11 without mosaic | 0.842 | 0.461 | 0.697 | 3.2 min | 4 ms |

These are **user-reported, unverified** values. The evaluated split, YOLO model
size, checkpoint initialisation, actual epochs, seed, hardware and timing scope
are absent. The small-object column must also specify its IoU range: AP50-small
and AP50–95-small are different metrics.

## 2. Why the approach has an advantage

**Representation.** M2 fixes the visual representation to HOG edges and optional
colour statistics. YOLO11 learns its visual representation and detection heads.
This gives it a way to distinguish object appearance from background texture
beyond a linear combination of fixed edge histograms. Official YOLO11 `.pt`
models are COCO-pretrained; COCO includes `bird` (class index 14). If those weights
initialised M3, prior bird/object knowledge is transferred before fine-tuning on
the 1,369 beach training images. That initialisation is typical, not confirmed
for this screenshot. [YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/),
[COCO class list](https://docs.ultralytics.com/datasets/detect/coco/).

**Scale and context.** RT-DETR's backbone/hybrid encoder combines multiscale
features and its transformer decoder predicts objects. These learned contextual
features can help with small or overlapping birds. Dense HOG also shares work
over a pyramid, but its representation remains handcrafted. The RT-DETR
architecture is described in the [original paper](https://arxiv.org/abs/2304.08069)
and [official model documentation](https://docs.ultralytics.com/models/rtdetr/).

**Localisation.** M3 learns bounding-box coordinates. The original M2 classifier
only scored preset boxes and did not refine their coordinates. NMS removes
duplicates; it cannot learn where the bird begins or ends. The revision therefore
includes a separate, explicitly labelled linear box-regression ablation using
HOG features, following the standard centre-offset/log-size idea; it is not a
neural detector. See [R-CNN Appendix C](https://arxiv.org/html/1311.2524v5).

**Training examples.** An image detector is exposed to positions, scales and hard
backgrounds at search time. Training M2 only on exact GT crops and a few random
backgrounds underrepresents those conditions. In the original M2 run, validation
crop ROC AUC was about 0.94, but full-image AP50 was only 0.017 before mining.
Those different tasks must not be confused. The revision uses aligned search
windows, position/scale variation, context and a much larger training-only mining
pass. These changes address a real implementation weakness; they do not make
HOG equivalent to learned neural features.

The low-false-positive tail matters especially for sliding windows. As an
illustration, a background false-positive rate of only 0.1% over the original
47,527 windows would give roughly 48 raw false alarms before NMS, while this
dataset averages about 6.3 birds/image. This is illustrative arithmetic, not a
measured M2 false-positive rate. A good crop ROC AUC does not guarantee that tail
is good enough for detection. Raising a display threshold may hide boxes but does
not improve the ordering of correct and incorrect detections that AP measures.

**Mosaic.** The supplied default/no-mosaic comparison differs by 0.015 AP50 and
0.041 in the reported small-object column. Mosaic changes scale/context during
training, but the table alone does not establish a controlled causal effect:
seeds, initial weights, epochs and other settings need to match. Different
training durations can reflect augmentation cost or different stopping points.
[Ultralytics training settings](https://docs.ultralytics.com/modes/train/).

## 3. Why the speed figures are not directly comparable

The previous M2 numbers were CPU end-to-end timings, including image loading,
feature extraction and NMS. The screenshot's 4–11 ms could be model-only GPU
timings; this is not established by the screenshot. A neural full-image forward
pass also shares computation rather than independently describing every crop.
The revised M2 likewise computes dense feature maps once per image-pyramid level.
Report hardware, input size, batch size, warm-up, synchronisation and timing scope
before comparing milliseconds to CPU seconds.

## 4. Inference-script issues corrected

The old wrapper loaded every checkpoint with `YOLO(...)`. RT-DETR has its own
predictor/validator and an NMS-free architecture. The revised wrapper provides
`--family`, uses `RTDETR(...)` for RT-DETR, and detects its underlying model when
the checkpoint is generically named `best.pt`. This routing is checked with
mocked loaders; a real neural checkpoint is still needed for an execution test.
[RT-DETR task map](https://docs.ultralytics.com/reference/models/rtdetr/model/),
[YOLO task map](https://docs.ultralytics.com/reference/models/yolo/model/).

The wrapper also rejects an unfiltered multi-class checkpoint: exporting every
COCO class as `bird` would be incorrect. An explicit `--bird-class` can be used
for a multi-class model. Fine-tuned single-class checkpoints continue to work.
Warm-up uses the same prediction options as timed inference. Metadata records
split/image IDs, input size, confidence floor, 300-detection cap, model family,
software/platform, weights hash, and separate end-to-end/model-only timings.
These fixes do not prove that the supplied table was generated by the old wrapper.

Example reproduction when trained weights are supplied:

```bash
python -m m3_dl.predict --family rtdetr --weights PATH/weights/best.pt \
  --split val --name rtdetr_l --imgsz 512 --conf 0.001 --max-det 300
python -m evaluation.evaluate --pred preds/rtdetr_l_val.json --split val
```

## 5. Evaluation boundary for the retry

The original official test set has already been evaluated and its failure images
were inspected in the earlier M2 work. The retry chooses every model and NMS
setting using training/validation only. Any later official-test result is labelled
as a follow-up on a previously seen test set, not as a fresh untouched holdout.
Original results and reports are preserved. The M3 screenshot is kept separately
from locally measured results and is not used as a target to fabricate scores.
