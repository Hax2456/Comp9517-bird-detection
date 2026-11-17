# M2 retry: dense HOG/SVM detection and comparison with M3

This is a separate revision. Original experiment files and the original report
are preserved. All revised model choices use the original validation split.
The official test set had already been evaluated and its examples inspected;
the new test table is an explicitly labelled follow-up, **not a fresh untouched
holdout**. The M3 screenshot is user-reported reference material, not a reproduced
result. See `M3_comparison_audit.md` for the code audit and sources.

## 1. Why retry the original M2

The supplied method description calls for multiscale windows, HOG versus HOG plus
colour, SVM classification, hard-negative mining and NMS. The original M2 used
these components, but trained on tight GT crops and random backgrounds before
searching many imperfectly aligned windows. Its validation AP50 was
0.0448, despite much higher classification performance on known bird crops.
It also described each overlapping crop independently, making search expensive.

The retry keeps the handcrafted-feature/traditional-ML category. It changes the
representation support and training sampling to match actual search-time windows.
Its first baseline changes several implementation choices together; an improvement
over the original would not isolate one single causal fix.

## 2. Revised methods

**Dense HOG and context.** Twelve width/height templates are learned from training
boxes. Each template defines a resized image-pyramid level. A 48 × 48 feature
window contains a central 36 × 36 predicted region and a six-pixel context margin.
HOG uses six-pixel cells, 2 × 2-cell blocks and nine unsigned bins: 7 × 7 × 4 × 9 =
1,764 dimensions. Reflection padding supplies context at image boundaries.
Block descriptors are computed once over each level, then sliced for each window.
Training and inference use the exact same maps and coordinate mapping.

**Positive variation.** Candidate windows with GT IoU ≥ 0.5 become positives,
retaining the best alignment plus up to three other matching positions/scales
per bird. This supplies position/scale variation rather than treating every
positive as a perfectly cropped object. The training audit records positive
coverage; birds without a matching candidate are not silently called positives.

**Backgrounds and SVM.** Forty background windows per training image are sampled
subject to IoU < 0.2 and less than 30% coverage of every GT bird. The initial
training set contains 33,648 positive windows and
54,760 negatives; candidate
positives cover 8465/8663
annotated training birds. HOG is already L2-Hys normalised, and colour histograms
are normalised to unit sum, so the revised linear SVM does not apply per-dimension
standardisation. C ∈ {0.01,0.1,1.0} and NMS IoU ∈ {0.3,0.5} are selected by
validation AP50. Initial class weighting is uniform. Scores are margins, not
probabilities. The candidate score floor is −1.5, with a 6,000-candidate cap and
300 final detections per image. These limits can restrict recall.

**Colour.** R2 adds 24 HSV histogram values. Cell histograms/integral sums provide
the exact histogram for the same 48 × 48 window without computing it separately
for every crop. Direct-histogram equivalence is tested.

**One hard-negative round.** The better validated initial model
(R1-HOG) scans training images only. Up to 32 false-alarm
windows per image are retained after background checks and NMS; a seeded global
cap of 40,000 prevents unbounded training growth. This produced
40,000 mined negatives. All original examples are retained
when refitting the SVM at the source C.

**Box refinement.** R4 adds a separate ridge regressor (α=10) trained only on
positive training windows. Targets are centre offsets normalised by proposal
width/height and log width/height ratios. Predicted offsets and scales are bounded
and boxes clipped to the image. This tests one localisation idea also used in
learned detectors while retaining HOG features; it is not a neural model.

**Kernel-SVM ablation.** R5 uses the selected initial linear detector to propose
candidates, then re-ranks its top 6,000 with an approximate nonlinear RBF SVM.
Training-only PCA (96 dimensions) and a 512-component Nyström map precede a
class-balanced linear SVM in kernel space (C=1). The bandwidth is computed from
training distances. This is an extra traditional-SVM experiment; it is not hidden
inside the four-step baseline. It cannot recover birds missing from its proposals.

| Variant | Meaning |
|---|---|
| R1-HOG | Aligned dense HOG + linear SVM + NMS |
| R2-HOG-HSV | Same sampling/search, with HSV histograms |
| R3-HardNeg | Best initial feature model after one mining round |
| R4-BoxRefine | R3 plus learned linear coordinate refinement |
| R5-KernelSVM | Initial proposals re-ranked with a nonlinear SVM approximation |

## 3. Experimental setup and measured results

The original 1,369 training and 291 validation images are retained. No validation
or test descriptor enters classifier, PCA, kernel-map or coordinate-regressor
fitting. The seed is 42. The full numerical source snapshot and protocol are in
`results/m2/revision_20261117/full`. The screenshot's results are not substituted for our measurements.

### Validation: model selection

| Variant | AP50 | AP50–95 | AP50 small | AP50 medium | AP50 large |
| --- | --- | --- | --- | --- | --- |
| R1-HOG | 0.0314 | 0.0080 | 0.0107 | 0.0897 | 0.0896 |
| R2-HOG-HSV | 0.0296 | 0.0076 | 0.0090 | 0.0899 | 0.0956 |
| R3-HardNeg | 0.0258 | 0.0059 | 0.0042 | 0.0579 | 0.0541 |
| R4-BoxRefine | 0.0410 | 0.0120 | 0.0069 | 0.0919 | 0.0785 |
| R5-KernelSVM | 0.0280 | 0.0066 | 0.0057 | 0.0705 | 0.0888 |

Validation selected **R4-BoxRefine**, AP50 **0.0410**. This is a signed
change of **-0.38 percentage points** from the original M2-C validation
AP50 (0.0448). There is no assertion that every tested modification helps.
The original M2-C remains the stronger accuracy choice by validation AP50. The revised detector should not replace it on accuracy grounds; its speed and methodological comparisons remain useful results.

The size breakdown is mixed: selected-revision validation AP50 is
0.0919 for medium birds versus 0.0750
originally, but 0.0069 for small birds versus
0.0160. This is not a uniform accuracy improvement.

### Follow-up evaluation on the previously seen official test set

| Variant | AP50 | AP50–95 | AP50 small | AP50 medium | AP50 large |
| --- | --- | --- | --- | --- | --- |
| R1-HOG | 0.0250 | 0.0069 | 0.0225 | 0.0412 | 0.1312 |
| R2-HOG-HSV | 0.0256 | 0.0066 | 0.0206 | 0.0451 | 0.1323 |
| R3-HardNeg | 0.0214 | 0.0056 | 0.0074 | 0.0243 | 0.0784 |
| R4-BoxRefine | 0.0334 | 0.0102 | 0.0110 | 0.0393 | 0.1142 |
| R5-KernelSVM | 0.0276 | 0.0052 | 0.0072 | 0.0509 | 0.0849 |

The validation-selected model scored AP50 0.0334; the original M2-C
test AP50 was 0.0296. Models and NMS settings were frozen in
`FROZEN_REVISION.json` before this follow-up. The test set's previous exposure
remains a limitation; these are not new independent generalisation estimates.

### CPU inference timing

| Variant | Latency (s/img) |
| --- | --- |
| R1-HOG | 0.357 |
| R2-HOG-HSV | 0.952 |
| R3-HardNeg | 0.346 |
| R4-BoxRefine | 0.402 |
| R5-KernelSVM | 1.072 |

Timing includes loading, dense feature maps, scores, optional refinement/reranking
and NMS, using the same three fixed validation images per revised variant in
serial, with one numerical thread. Original M2 timing used five images, so a
precise cross-run speedup should not be inferred from two small timing samples.
Training feature preparation took 275.9s
and the six-worker mining pass took 483.4s.
These wall times are not per-image serial latency.

![Revised results and timing](../results/m2/revision_20261117/full/revision_comparison.png)

## 4. What explains M3's reported advantage

The existing M3 script supplies inference only; its training notebook, checkpoints
and prediction files are absent. The screenshot reports RT-DETR-L AP50 0.891 and
YOLO11 AP50 0.857, but its split, initialisation and hardware are not established.
Those values cannot yet be certified as matching this exact evaluator/protocol.

YOLO11 learns features and box coordinates. Standard official `.pt` weights are
COCO-pretrained, including a bird category; if those weights initialised this M3,
that transfers prior object knowledge unavailable to handcrafted HOG. RT-DETR
combines learned multiscale/contextual features with transformer-based prediction.
These architectural advantages are supported by the official documentation;
pretraining for the screenshot's particular run is still unverified. Mosaic can
change training scale/context, but its reported effect is not isolated unless
seeds, epochs, initial weights and other settings match.

The old M3 wrapper incorrectly used the YOLO loader for every architecture. It
now selects RT-DETR's dedicated predictor, rejects unfiltered multi-class
checkpoints, and records settings and timing metadata. Mocked routing checks pass;
a real checkpoint is needed for an actual neural execution test. No M3 score has
been invented or claimed as reproduced. GPU/model-only milliseconds also cannot
be compared directly with CPU end-to-end seconds.

## 5. Successes, failures and limitations

The display score 0.5 is chosen from cached
validation predictions by F1; it changes the visualisation, not the AP ranking
or trained model. Cases are selected by explicit criteria: a positive image with
at least one TP and few errors, the most missed birds among remaining images with
at most twenty birds, the negative image with most FP, and the densest flock.

![Validation examples](../results/m2/revision_20261117/full/revision_examples.jpg)

The validation examples show an important distinction between detecting bird-like
parts and localising whole birds. In `beach_bird_1582`, four dark birds against
relatively simple water are all correctly localised. In `beach_bird_1660`, many
small detections lie on parts of much larger birds or on the surrounding ground;
none matches the twenty full-bird annotations at IoU 0.5. The negative image
`beach_bird_1378` produces 27 false detections on high-contrast rock/water texture.
In `beach_bird_1658`, only one of 100 annotations is matched at the display
threshold, alongside 33 false detections. These are observed cases, not a separate
estimate of population performance.

A small candidate's feature support extends only one third beyond its predicted
width/height. A region covering a head may therefore lack the full-body context
needed to decide whether it represents a complete bird. Ridge refinement is
trained on already-overlapping positive proposals, so it is not designed to turn
every tiny part proposal into a correct full-bird box. This is consistent with
the observed localisation and background-confusion failures, without proving that
one feature or stage explains every error. Full-image neural detectors can use
learned contextual features beyond the final output box.


The fixed representation still lacks learned part/texture semantics. Finite
templates, context warping, ambiguous partial windows, candidate caps, NMS and
linear/nonlinear model capacity can all affect detection. Box refinement does not
repair a missing proposal; kernel reranking does not make its initial candidate
set complete. An aggregate change does not by itself prove a particular causal
mechanism. One seed, small timing samples and a previously inspected official
test set limit certainty. M3 reproduction still requires its trained weights and
training configuration.

## 6. Conclusion

The retry preserves a genuine traditional-ML detector and reports all attempted
variants. The selected revised method is R4-BoxRefine, validation AP50
0.0410, with a declared follow-up test AP50 of 0.0334.
Dense feature sharing addresses redundant computation; aligned examples and
mining address search-time training mismatch; box/kernel stages test additional
classical improvements. These changes do not supply the pretrained learned
representation used by standard deep detectors. Further choices should use
validation or a newly reserved group-aware holdout, not repeated test-score tuning.

## References and attribution

1. Dalal & Triggs, [Histograms of Oriented Gradients for Human Detection](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf), CVPR 2005.
2. Girshick et al., [R-CNN, Appendix C](https://arxiv.org/html/1311.2524v5), CVPR 2014; box-regression parameterisation, adapted here to HOG.
3. scikit-learn, [Nyström kernel approximation](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html).
4. Ultralytics, [YOLO11](https://docs.ultralytics.com/models/yolo11/), [COCO class list](https://docs.ultralytics.com/datasets/detect/coco/) and [RT-DETR](https://docs.ultralytics.com/models/rtdetr/).
5. Zhao et al., [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069), CVPR 2024.
6. Kai Liu, [Beach Bird Detection Dataset v1.0](https://www.kaggle.com/datasets/liugalhigh/beach-bird), Kaggle, September 2026; dataset attribution from the project brief.

Libraries: scikit-image's compiled HOG histogram kernel, scikit-learn's K-means,
SVM/PCA/Nyström/ridge implementations, NumPy, Pillow, pandas, Matplotlib, joblib,
threadpoolctl and the shared pycocotools evaluator. Sampling, dense-map slicing,
integral colour features, mining, NMS, coordinate mapping and experiment control
are local code. This report is a contribution to the group's IEEE PDF, not a
complete group submission or proof that all M3 results have been reproduced.
