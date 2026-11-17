# M2: HOG / HSV + linear SVM

## 17 November retry and M3 audit

The separate retry is explained in `M2_revision.ipynb` and `M2_revision_report.md`.
`M2_revision_requirements_review.md` maps this retry to the brief and rubric.
`M3_comparison_audit.md` distinguishes the supplied M3 screenshot from reproducible
evidence and explains the architecture/pretraining/localisation differences.
The original reports and results below are preserved.

The highest revised validation AP50 is 0.0410 (R4-BoxRefine), compared with 0.0448
for the original M2-C. The revised model is faster in small serial CPU samples
(about 0.40 s/image versus 10.25 s/image), but does not establish an accuracy win.
The original official test set was previously inspected, so revised test results
are explicitly labelled follow-up measurements. Model selection uses validation.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m m2_ml.run_dense \
  --out results/m2/revision_20261117/NEW_RUN --workers 6 --kernel

.venv/bin/python -m m2_ml.predict_dense \
  --model results/m2/revision_20261117/full/R4-BoxRefine.joblib \
  --split val --limit 3 --out results/m2/revision_20261117/demo.json
```

The second command is a short inference demo; its subset output is not a
full-split score. Omit `--limit` for full-split predictions. Read AP from the
shared evaluator rather than counting low-confidence candidate boxes. For the
completed retry, `results/m2/revision_20261117/full/` stores the protocol, source
snapshot, models, training audit, parameter search, predictions, timings and figures.

## Original full experiment

Read **`M2_full_results.ipynb`** for the measured full experiment and
**`M2_report_for_group.md`** for the concise report contribution. The detailed
report is **`M2_report.md`**, and **`M2_requirements_review.md`** maps the work to
the brief and rubric. **`M2_walkthrough.ipynb`** explains the method and trains the three
controlled variants, selects settings on validation data, generates report figures
and tables, and provides a separate final-test step. **`M2_report.md`** contains
the report contribution organised by section. Measured numbers come from the
completed-run files, not from the explanatory schematic or synthetic checks.

## Setup and execution

From the repository root, with Python 3.10+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-m2.txt
python -m jupyterlab m2_ml/M2_walkthrough.ipynb
```

Choose this environment as the kernel. Set `DATA_ROOT` to the supplied dataset,
enable `RUN_EXPERIMENTS`, and start with `PROFILE="quick"`. The full profile is
intended for the final report. Check the one-image runtime estimate before running
all scans: exact per-crop HOG is deliberately transparent but can be expensive.
The notebook can be read and its diagrams executed without the dataset.

This local workspace has a prepared Python 3.12 `.venv`. Use `.venv/bin/python`
or activate it; the macOS system `python3` may be older than the required version.
`configs/paths.yaml` points at the supplied top-level dataset directory.

## Full experiment used for the report

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m m2_ml.run_full \
  --run-dir results/m2/full/NEW_RUN_NAME --workers 8 \
  --mining-images 160 --benchmark-images 5 --robustness-images 20
```

The original full run is `results/m2/full/20261114_full/`. It uses every original
training/validation/test image, a fixed 160-image mining subset, five validation
images for serial latency and twenty validation images for frozen robustness checks.
The full runner shares identical features across classifiers and distributes
images across local workers. `M2_walkthrough.ipynb` shows the same learning steps
sequentially for teaching; its timing is not the full runner's measured timing.
Add `--test` only when the validation-based protocol is final. All variant settings
are frozen before the final test batch. Use a new run directory for a new experiment;
do not tune models from the saved test scores.

To regenerate writing, figures and the results notebook from saved predictions:

```bash
MPLCONFIGDIR=/tmp/m2-matplotlib .venv/bin/python -m m2_ml.summarise_run \
  --run-dir results/m2/full/20261114_full
```

This last command scores only the supplementary crop diagnostic and reformats
saved detection results; it does not retrain or repeat test inference.

Generated files go to `results/m2/<profile>/<run_id>/` (already ignored by Git):

- `validation_summary.csv`: selected variant metrics and complete detection times.
- `validation_search.csv`: all initial and mined postprocessing trials.
- `training_crop_counts.csv`, `hard_negatives.csv`: training audit information.
- `manifest.json`: image lists, seed, templates, settings and software versions.
- `M2-*.joblib`, `final_model.joblib`: fitted local models.
- `M2-*_val.json`: predictions in the shared format.
- PNG plots and qualitative examples for the report.
- `test_summary.csv`, `m2_final_test.json`: created only by the final-test cell.

Training pipeline times explicitly include shared generation of both feature sets.
See the notebook/report for timing boundaries. Undefined size-based AP is N/A.
Quick-run predictions are evaluated against exactly the sampled validation images;
do not score them against full-split GT and report that as a complete run.

## Predict using a fitted model

After training, replace `<run_id>` and dataset paths below:

```bash
python -m m2_ml.detector \
  --model 'results/m2/full/<run_id>/final_model.joblib' \
  --data-root 'data/raw/Beach Bird Detection Dataset v1.0' \
  --split val --output preds/m2_final_val.json

python -m evaluation.make_coco_gt
python -m evaluation.evaluate --pred preds/m2_final_val.json --split val
```

`--data-root` is optional if `configs/paths.yaml` is configured. The CLI also writes
a timing/environment JSON beside the prediction file. The final-test cell is the
recommended way to run the held-out test once after selection. Only load local
joblib models you trust, since the format uses Python object serialisation.

## Checks and provenance

```bash
python -m unittest m2_ml.test_detector -v
```

These tests use small synthetic geometric objects. They check feature pairing,
box geometry, NMS, strict background mining, model persistence, full prediction
flow and the shared evaluator. They are not Beach Bird performance measurements.

The implementation uses scikit-image HOG, scikit-learn K-means/StandardScaler/
LinearSVC, Pillow image operations, NumPy, joblib and tqdm. Evaluation reuses the
team's pycocotools wrapper. Notebook figures/tables use Matplotlib and pandas.
See `M2_report.md` and the notebook for primary references and attribution.
No pretrained detector or externally copied detection pipeline is included.

The vectorised HOG backend uses scikit-image's compiled cell-histogram kernel and
vectorises matching L2-Hys block normalisation. Its private kernel interface is
covered by reference-equivalence tests; use the recorded package versions for
reproduction. `Config(hog_backend="reference")` selects the public HOG implementation.

The dataset is available locally. Real AP, timing and analysis outputs are stored
separately under `results/`; exclude datasets, weights and result images from the
source submission. The group must still assemble the final IEEE-format PDF and
add the other method families and video contribution.
