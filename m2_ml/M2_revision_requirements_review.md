# M2 retry: requirements and rubric check

Rechecked 17 November 2026 against the seven-page project specification and
one-page rubric in the parent folder. The original `M2_requirements_review.md`
documents the original experiment; this file covers the separate retry.

| Requirement or assessed aspect | Retry evidence and limits |
|---|---|
| Original dataset splits | All 1,369 training and 291 validation images; all 240 test images only in the explicitly labelled follow-up. Split lists are in the run protocol. |
| Training and validation separation | Templates, positive/negative windows, mining, SVM, PCA, kernel map and coordinate regression are fitted from training only. C, NMS and revised model selection use validation. |
| Test only for final evaluation | Settings were frozen before follow-up test inference. However, this test set had already been evaluated and inspected in the original work. The revision cannot claim a fresh untouched test result. No subsequent model changes were selected from these follow-up scores. |
| At least mAP@0.5 | Full validation and follow-up test predictions, AP50, AP50–95 and size AP50. The shared evaluator uses a 300-detection cap. |
| Training and inference times | Preparation, fitting and mining records are saved separately; serial inference includes loading/features/postprocessing. Three validation images per revised model: a small timing sample, different from the original five-image sample. |
| Methods and literature | HOG versus HOG+HSV, one mining round and NMS implement the supplied reference. Box regression and approximate kernel SVM are additional disclosed experiments. Primary references and dependency attribution are in the report. |
| Results and discussion | All five variants are reported, including unsuccessful changes. The best revised validation AP50 is 0.0410 versus the original 0.0448; the original remains the stronger accuracy choice. Validation overlays show successes, missed whole birds, background false alarms and a dense flock. |
| Comparison across method families | The M3 audit explains architectural differences and missing provenance. Its screenshot is reported reference material; there are no local trained M3 weights or training logs to verify those scores. A valid numerical comparison still needs M3 predictions from matching splits and the shared evaluator. |
| Structured, documented code (4 marks) | Separate training, detector, prediction, summarisation and tests; runnable README commands; six-cell executed results/explanation notebook; source snapshot and reproducibility records. |

The report rubric assigns introduction 1, literature review 2, methods 3,
experimental results 3, discussion 2, conclusion 1 and overall quality 1 marks.
The revised M2 report contributes methods, experiments, discussion, conclusion
and references; the group must integrate its introduction and literature context.
Metric values themselves are not the marking criterion: sound methods, rigorous
evaluation and complete discussion matter. This does not guarantee a mark.

The five variants remain one traditional-ML detector family. They alone do not
satisfy the requirement for at least two meaningfully different group approaches.
The original robustness/counting/density analyses remain original-run evidence;
they were not rerun for the dense revision and must not be attributed to it.

The new source ZIP excludes input/result images and model weights and strips
notebook outputs. The 25 MB limit applies to the combined group source submission.
The group still needs its two-column IEEE PDF (10 main-text pages, references
excluded; 10 MB), and presentation/demo video (10 minutes, 720p/1080p MP4,
100 MB, every member presenting with live webcam). These are group deliverables,
not outputs completed by the M2 retry.
