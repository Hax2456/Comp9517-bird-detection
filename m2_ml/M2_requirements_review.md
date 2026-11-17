# M2 requirements and rubric review

Reviewed against the local **requirement.pdf** (seven pages) and **rubric.pdf**
(one page). The reference experiment is `results/m2/full/20261114_full/`.
This review covers the M2 contribution; the group still owns the combined report,
cross-family comparison, video and final submission.

## Required method and evaluation work

| Requirement | M2 implementation and evidence |
|---|---|
| Use the Beach Bird dataset and respect its original splits | Every original training image supplies eligible crops; every validation and test image is evaluated in its supplied split. `data_audit.json`, image lists in `manifest.json`, strict missing-label checks. |
| Prevent training/validation/test leakage | Crop sampling, K-means templates and feature scaling use training only. Mining uses a fixed 160-image training subset. Hyperparameters and model selection use validation only. Source-group audit finds no groups crossing splits. |
| Final test only after tuning | `FROZEN_SELECTION.json` records all variant settings and validation-selected operating thresholds before a single final test batch. `FINAL_TEST_STARTED.json` links that frozen file by SHA-256; `TEST_COMPLETED.json` records completion. |
| Compute at least mAP@0.5 | All methods reuse the team's evaluator. AP50 is single-class mAP@0.5. `validation_summary.csv`, `test_summary.csv`. The 300-detection evaluator limit is shared. |
| Compare training and testing/inference times | Shared preparation, SVM fitting and mining costs are explicit. Independent serial latency uses the same five validation images per model, with no concurrent workers and one numerical thread. Per-image samples and exact boundaries are reported; full-test throughput is separate. |
| Representative successes and failures | Real final-test overlays and per-image TP/FP/FN counts. Explicit selection criteria include a successful positive example, most missed birds, a negative scene and the densest flock. Frozen validation-F1 thresholds control display; they do not change AP ranking. |
| Explain strengths, weaknesses and future work | Report discussion compares actual feature/mining results, candidate coverage, size/density metrics and inspected examples. It distinguishes observations from possible causes and future improvements. |
| Relevant literature and source attribution | HOG reference, classifier/scaler documentation, dataset citation and dependency attribution in report/README. Vectorised HOG uses scikit-image's histogram kernel and is checked against its public descriptor. |
| Structured, documented code | `detector.py`, `run_full.py`, `summarise_run.py`, explanatory notebook, real-results notebook, CLI instructions and synthetic integration tests. |

### Meaningfully different methods are a group requirement

M2-A, M2-B and M2-C are **controlled variants of the same detector family**.
They should not be presented as satisfying the group's minimum of two meaningfully
different approaches by themselves. Compare this M2 family with the group's M1
non-learning method and/or M3 deep-learning method. Those methods must use the
same split and evaluator configuration. They are not implemented or trained by
the M2 work in this chat.

## Depth beyond the minimum

The specification gives examples of greater analytical depth rather than requiring
every suggested extension. M2 includes:

- Controlled HOG-versus-HOG+HSV comparison using identical initial crop samples and
  equal hyperparameter budgets.
- Hard-negative retraining with both matched and reselected postprocessing.
- Candidate-window recall on all validation annotations to separate search geometry
  limitations from classifier performance.
- Small/medium/large-object AP50 and flock-density breakdowns, with sample counts.
- Fixed clean/blur/brightness comparisons on the same 20 validation images, with
  frozen models and settings; this limited subset is stated explicitly.
- Precision/recall/F1 and counting MAE at validation-selected score thresholds,
  kept separate from the score range used for AP.
- Reproducibility evidence: seeds, exact image names, templates, package versions,
  source hashes, per-image predictions and timing samples.

These support analysis and discussion; they do not guarantee a particular mark.
The rubric explicitly says grading is based on soundness/originality, rigorous
evaluation and comprehensive results/discussion rather than metric values alone.

## Report rubric mapping (13 marks for the complete group report)

| Assessed aspect | Marks | M2 contribution |
|---|---:|---|
| Introduction | 1 | Motivation for a traditional learned detector; dataset context. Integrate with the group introduction. |
| Literature review | 2 | HOG/sliding-window/SVM background and citations; relationship to the implemented adaptation. |
| Methods | 3 | Pipeline, crop rules, train-derived geometry, feature dimensions, classifier, NMS and mining details. |
| Experimental results | 3 | Fixed split protocol, full validation/test metrics, timing boundaries, size/density/robustness analyses and figures. |
| Discussion | 2 | Evidence-based feature/mining comparison, failures, limitations, and remaining M1/M3 group comparison. |
| Conclusion | 1 | Measured selected configuration and limitations, with targeted future work. |
| Overall quality | 1 | Consistent terminology, clear table/figure captions, references and readable section structure. Final IEEE assembly remains a group step. |

Code carries **4 marks** and should remain structured, documented and reproducible.
The video carries **13 marks**, including methods, results, discussion and a
software demonstration. The notebooks and CLI support the M2 demonstration; a
video has not been recorded by this work.

## Group assembly and submission constraints

- The final report must be a **two-column IEEE-format PDF**, at most **10 pages of
  main text including figures/tables**, excluding references; upload limit **10 MB**.
  `M2_report.md` is contribution text to merge, not the final combined IEEE PDF.
  Select the most informative M2 tables/figures to fit the shared page budget.
- The source submission is a **ZIP of at most 25 MB**. Exclude the dataset, trained
  model weights and result images. Strip notebook outputs for the source archive;
  keep measured results separately for reporting. Cite external libraries/code in
  the README. Do not zip the whole working folder with its dataset/results.
- The video is at most **10 minutes**, roughly eight minutes of explanation plus
  two minutes of demonstration, in **720p/1080p MP4**, at most **100 MB**. All group
  members must appear presenting with live webcam video and introduce themselves.
- All members must contribute to development/code as well as presentation/report.
  Only one group member submits the combined deliverables.
- The stated deadline is **20 November 2026, 18:00 AET**. Verify the course's
  submission instructions and use the stated course timezone.

## Scope and limits to retain in the final writing

Full-split AP uses all 291 validation and 240 test images. Standalone serial timing
uses five validation images, and robustness uses twenty validation images. Mining
uses 160 training images; initial training uses all 1,369 images and all 8,663 bird
boxes. Report these differing sample sizes explicitly. A no-bird group's AP is
undefined, not zero. The 51+ test group contains only two images. One seed and a
small timing/stress-test sample do not establish broad statistical certainty.

No model changes should be made in response to final-test scores from this run.
Any follow-up research needs its own predeclared validation protocol and must not
present repeated test-driven optimisation as an untouched final evaluation.
