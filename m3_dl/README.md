# M3 inference and reproduction evidence

At the 17 November review, this folder supplied prediction code but no training
notebook, checkpoint or run logs. The reported table is transcribed and discussed
in `../m2_ml/M3_comparison_audit.md`; those values are not independently reproduced.

With the actual trained single-class checkpoint and deep-learning environment:

```bash
python -m m3_dl.predict --family rtdetr --weights PATH/weights/best.pt \
  --split val --name rtdetr_l --imgsz 512 --conf 0.001 --max-det 300
python -m evaluation.evaluate --pred preds/rtdetr_l_val.json --split val
```

Use `--family yolo` for YOLO, or `auto` to detect RT-DETR's underlying model.
RT-DETR uses its own predictor. Multi-class weights require `--bird-class`;
otherwise other classes could be incorrectly exported as birds. The metadata
JSON beside predictions records architecture, input/evaluation settings, weights
hash, available checkpoint training arguments, actual device and separate timing
scopes. Per-image end-to-end timings and Ultralytics model-only times differ.

To reproduce the screenshot, supply its weights and training configuration,
including initial model/pretraining provenance, YOLO variant, actual epochs,
seed, augmentation, split, input size, hardware and metric definitions. Confirm
that the small-object column uses the same IoU range and area thresholds as M2.
Standard official YOLO11 `.pt` weights are COCO-pretrained and COCO includes birds;
the screenshot's actual initialisation cannot be inferred from its score alone.

`python -m unittest m3_dl.test_predict -v` checks loader routing with mocks.
It does not download weights or execute a neural network.
