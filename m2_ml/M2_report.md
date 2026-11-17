# M2: traditional machine-learning bird detection

**Measured full-run contribution.** This text is organised for insertion into the
group report. Results come from `results/m2/full/20261114_full`, using all 1,369 training images,
291 validation images and 240 final-test images. The method is one family with
three controlled variants. The complete group report must also include the other
method families and be assembled in two-column IEEE PDF format, within ten pages
of main text excluding references. This M2 contribution alone is not the full group submission.

## Introduction contribution

Bird localisation in coastal scenes is difficult because object size, flock
density and background texture vary. M2 investigates how far handcrafted
appearance descriptors and a learned binary classifier can address this task.
It provides an interpretable CPU-based comparison with the group's non-learning
image-processing methods and deep detectors, while separating crop classification
from full-image object detection.

## 1. Literature review

Traditional object detection separates handcrafted feature extraction from learned
classification. Dalal and Triggs [1] demonstrated a sliding-window framework using
histograms of oriented gradients (HOG) with a linear support vector machine (SVM).
HOG summarises local edge directions and uses overlapping normalised blocks to
reduce sensitivity to local contrast. Their original task was pedestrian detection;
we evaluate the same principles for bird localisation in beach imagery. The SVM
framework of Cortes and Vapnik [6] supports learned separation when training
classes are not perfectly separable; our implementation uses a regularised linear
classifier with the solver and loss specified below.

Colour is a potentially complementary cue because birds and beach backgrounds can
have different appearances, but it may also encode scene or illumination bias.
We therefore compare HOG alone with HOG concatenated with HSV histograms rather
than assuming a colour benefit. We also investigate hard-negative mining: retraining
on false alarms from training images. Our window search, sampling, NMS and mining
orchestration are local code. HOG uses scikit-image's compiled histogram kernel with
vectorised block normalisation, checked against its public HOG implementation [2].
Standardisation, K-means and the SVM solver use scikit-learn [3,4].

## 2. Methods

### 2.1. Pipeline and controlled variants

The detector searches windows at multiple sizes and aspect ratios, resizes each
crop to 32 × 32 pixels, extracts features, standardises them, computes a linear
SVM score, and removes duplicate boxes with greedy non-maximum suppression (NMS).
M2-A uses HOG; M2-B adds HSV histograms; M2-C retrains the better validated initial
feature model with hard negatives. These are ablations within M2, not three
fundamentally distinct detector families.

### 2.2. Sampling and window geometry

All 8,663 annotated training birds provided positive crops.
Up to 24 background windows per training image produced 32,856
initial negatives. A negative window had to satisfy both IoU < 0.10 and intersection
over each bird's area < 0.10 against every annotation. The coverage restriction
prevents a large window containing a small bird from being incorrectly labelled
background. Empty-labelled training images were included.

Nine templates were learned by K-means on log training-box widths and heights,
with seed 42. The resulting pixel dimensions were
`[[12, 11], [21, 20], [38, 27], [29, 43], [58, 45], [65, 86], [114, 68], [141, 130], [270, 222]]`. Templates slide across the original 512 × 512 image at
one-quarter-width/height strides, rounded and bounded below by two pixels. Right
and bottom edges are explicitly searched. This yields 47,527 windows per image.
Template fitting uses training annotations only.

### 2.3. Features and classifier

HOG uses nine unsigned orientation bins, 4 × 4-pixel cells, 2 × 2-cell overlapping
blocks and L2-Hys normalisation. There are 7 × 7 block positions, giving
7 × 7 × 2 × 2 × 9 = 1,764 features. M2-B adds three independently normalised
eight-bin H, S and V histograms, giving 1,788 features. Square resizing standardises
dimensions but can distort shape; enlarging tiny birds cannot restore missing detail.

A scaler is fitted exclusively on training crops, followed by an L2-regularised,
class-balanced linear SVM with squared-hinge loss and a 20,000-iteration limit.
The decision score is $s(\mathbf x)=\mathbf w^\top\mathbf z(\mathbf x)+b$,
where $\mathbf z$ is the standardised descriptor. Scores rank detections and
are not calibrated probabilities. Both initial feature sets use identical crops.

Each initial variant receives the same validation grid: C ∈ {0.1, 1.0}, NMS IoU
∈ {0.3, 0.5}, and candidate score floor ∈ {−2.0, −1.0}. AP50 selects settings;
the first declared candidate wins a tie. At most 5,000 candidates enter NMS and
300 detections survive per image, matching the shared evaluator's detection cap.
The final selected configuration was **M2-C**, C=1.0, NMS IoU=0.5,
and AP candidate floor=-2.0.

### 2.4. Hard-negative retraining and computation

M2-C used M2-B as its source and scanned a fixed random subset of
160 training images (seed 42). Windows with score ≥ 0
that met both background restrictions were mined, with NMS at 0.3, up to 12 per
image and 6,000 globally. This added 1,920 negatives.
The original samples were retained, and the scaler and SVM were refitted with the
source C fixed. NMS and score-floor settings were reselected on validation.
A matched-postprocessing comparison also evaluates retraining with the source
thresholds unchanged.

For efficiency, overlapping block normalisation is vectorised and each candidate's
features are reused across classifiers during grid evaluation. Eight worker
processes distribute images, with numerical-library threading limited to one per
process. Tests verify descriptor agreement with public scikit-image HOG and agreement
between individual and shared model scoring. Per-method latency is measured
separately in serial; shared-search wall time is not reported as standalone latency.

## 3. Experimental setup

The original source-group-aware splits were retained: 1,369/291/240 images and
8,663/1,843/1,518 boxes in train/validation/test. A manifest audit found zero source
groups spanning splits, and image IDs were distinct across splits. No validation
or test crops entered scaler or classifier fitting. All configurations and display/
counting score thresholds were frozen in `FROZEN_SELECTION.json` before one final
batch evaluated all frozen variants on test. Final selection used validation AP50,
not the test ranking.

The primary metric is mAP@0.5 (AP50 for this single class). The shared pycocotools
evaluator also reports AP50–95 and size-based AP50. Size thresholds follow COCO
box areas (<32², 32²–96² and >96² pixels). Candidate-window recall at IoU 0.5 is
0.9566 (1763/1843)
over **all validation annotations**, measuring geometry coverage before classification.

Runs used local macOS ARM64, ten logical CPUs available, Python
3.12.14, NumPy 2.5.4,
scikit-image 0.26.0, scikit-learn
1.9.1 and pycocotools
2.0.11. Exact versions, image lists,
source hashes and configurations are saved with the run. Hardware-specific timing
cannot be compared directly with results measured on another machine.

## 4. Experimental results

### 4.1. Validation and final-test detection accuracy

Validation (291 images; model selection):

| Variant | AP50 | AP50–95 | AP50 small | AP50 medium | AP50 large |
| --- | --- | --- | --- | --- | --- |
| M2-A | 0.0171 | 0.0040 | 0.0050 | 0.0386 | 0.1010 |
| M2-B | 0.0172 | 0.0039 | 0.0043 | 0.0397 | 0.1099 |
| M2-C | 0.0448 | 0.0114 | 0.0160 | 0.0750 | 0.1044 |

Final test (240 images; frozen settings):

| Variant | AP50 | AP50–95 | AP50 small | AP50 medium | AP50 large |
| --- | --- | --- | --- | --- | --- |
| M2-A | 0.0122 | 0.0030 | 0.0088 | 0.0215 | 0.1028 |
| M2-B | 0.0153 | 0.0036 | 0.0116 | 0.0260 | 0.0978 |
| M2-C | 0.0296 | 0.0071 | 0.0243 | 0.0322 | 0.1198 |

Adding colour improved validation AP50 by 0.01 percentage
points. Hard-negative retraining improved validation AP50 by
2.76 points relative to its M2-B source after postprocessing
selection. With the source postprocessing unchanged, the mined model's AP50 was
0.0448, versus 0.0172 for its source.
The validation-selected **M2-C** achieved final-test AP50
**0.0296** and AP50–95 **0.0071**.
The colour difference is very small; a single seed without uncertainty estimates does not establish a reliable benefit.

![M2 accuracy and latency comparison](../results/m2/full/20261114_full/m2_comparison.png)

### 4.2. Timing

| Variant | Train pipeline (s) | SVM fit (s) | Latency (s/img) | Timed images |
| --- | --- | --- | --- | --- |
| M2-A | 26.010 | 13.895 | 3.335 | 5 |
| M2-B | 25.861 | 13.746 | 10.214 | 5 |
| M2-C | 469.319 | 16.524 | 10.247 | 5 |

Training pipeline time includes the shared 12.11s
preparation of both feature sets and the selected fit. M2-C additionally includes
its source fit, parallel mining wall time and refit. It excludes validation grid
search and is not the minimum possible HOG-only training cost. Serial inference
timings cover the same five fixed validation images per variant, after grid
evaluation, with no other worker activity. They include loading, resizing/features,
standardisation, scoring, NMS and prediction construction; model loading, JSON
writing and evaluation are excluded. These five-image latency estimates are
separate from the full-split accuracy measurements; they are not a test-set
latency benchmark.

In this implementation, adding HSV increased mean serial latency by approximately
3.06×
while its validation AP50 gain was only 0.01 percentage points.
Per-window colour conversion/histogram extraction is a plausible explanation for
the added cost; this timing result is specific to this implementation and hardware.

### 4.3. Small objects, density and thresholded counts

![Final-test AP50 by size](../results/m2/full/20261114_full/m2_size_analysis.png)

For M2-C, small-bird test AP50 was 0.0243, lower than medium-bird
AP50 (0.0322). The selected model's density breakdown was:

| Birds/image | Images | Bird boxes | AP50 | AP50–95 |
| --- | --- | --- | --- | --- |
| 0 | 27 | 0 | N/A | N/A |
| 1-5 | 135 | 319 | 0.0337 | 0.0084 |
| 6-20 | 63 | 691 | 0.0378 | 0.0093 |
| 21-50 | 13 | 386 | 0.0725 | 0.0167 |
| 51+ | 2 | 122 | 0.1270 | 0.0343 |

The no-bird group's AP is undefined and reported as N/A. Dense-group results must
be interpreted with their sample counts, especially the very small 51+ group.
Each group AP is recomputed using only that group's images and false positives;
it is not an additive component of overall AP and should not be averaged across
groups. Higher AP in one small group does not establish that density helps detection.
For supplementary precision, recall, F1 and counting MAE, a margin threshold was
selected from a fixed grid by validation F1 and frozen before test. AP uses the
lower candidate floor and does not use this operating threshold.

| Variant | Operating score | Precision | Recall | F1 | Count MAE |
| --- | --- | --- | --- | --- | --- |
| M2-A | 2.5000 | 0.0360 | 0.0942 | 0.0521 | 16.5458 |
| M2-B | 2.5000 | 0.0443 | 0.1080 | 0.0628 | 14.8250 |
| M2-C | 1.5000 | 0.1042 | 0.0685 | 0.0827 | 6.2000 |

For context, always predicting zero birds would have a test counting MAE of
6.325 birds/image. Counting MAE should therefore be interpreted with
precision and recall rather than presented alone as evidence of useful counting.

### 4.4. Fixed-distortion robustness

On the same fixed random subset of 20 validation images, frozen models were
evaluated on clean images, Gaussian blur (Pillow radius 1 pixel), and brightness
multiplied by 0.6. Labels and model settings were unchanged. This is a limited
validation stress test rather than a whole-test robustness claim.

| Variant | Condition | Images | AP50 | AP50–95 |
| --- | --- | --- | --- | --- |
| M2-A | clean | 20 | 0.0113 | 0.0021 |
| M2-B | clean | 20 | 0.0131 | 0.0024 |
| M2-C | clean | 20 | 0.0259 | 0.0057 |
| M2-A | blur_sigma1 | 20 | 0.0129 | 0.0023 |
| M2-B | blur_sigma1 | 20 | 0.0157 | 0.0028 |
| M2-C | blur_sigma1 | 20 | 0.0331 | 0.0074 |
| M2-A | brightness_0.6 | 20 | 0.0113 | 0.0028 |
| M2-B | brightness_0.6 | 20 | 0.0131 | 0.0035 |
| M2-C | brightness_0.6 | 20 | 0.0340 | 0.0089 |

![M2 fixed-distortion robustness](../results/m2/full/20261114_full/m2_robustness.png)

## 5. Discussion

The controlled colour comparison shows that appearance information improved
aggregate validation detection under this setup; it does not establish that colour
is reliable in every scene. Hard-negative retraining improved AP50 relative
to its source, and the matched-threshold result separates the retraining comparison
from postprocessing reselection. Adding negatives also changes class-balanced
weights, so the experiment evaluates the complete retraining procedure rather
than isolating one causal mechanism.

A supplementary classification diagnostic uses the 1,843 annotated validation
bird crops and 6,984 randomly sampled background crops. No model is fitted or
selected using this diagnostic. Its task is easier because bird locations are
already supplied:

| Model | Crop ROC AUC | Crop AP | Balanced accuracy at 0 |
| --- | --- | --- | --- |
| M2-A | 0.9405 | 0.8429 | 0.8707 |
| M2-B | 0.9398 | 0.8389 | 0.8679 |
| M2-C | 0.9335 | 0.8415 | 0.8586 |

These crop metrics must not be substituted for detection AP. They explain why a
plausible patch classifier need not make a strong image detector: full-image
search evaluates tens of thousands of background and imperfectly aligned windows
per image, rather than a curated mixture of exact bird crops and random negatives.
Hard negatives address part of this distribution difference. Training with
IoU-controlled jittered positive windows is a targeted future experiment; it was
not performed here.

Candidate recall of 0.9566 shows that the finite template search
can cover most validation annotations at IoU 0.5, while some remain unreachable
even before classification. Differences between candidate recall and achieved
AP cannot be attributed to geometry alone: descriptor ambiguity, background
confusion, localisation scores and NMS also matter. Size- and density-stratified
results and the following thresholded examples provide evidence of remaining
misses and false alarms.

Validation geometric coverage after model ranking and postprocessing was:

| Variant | Images hitting cap | Top-candidate coverage | Post-NMS coverage |
| --- | --- | --- | --- |
| M2-A | 291 | 0.8969 | 0.5534 |
| M2-B | 291 | 0.9012 | 0.5513 |
| M2-C | 291 | 0.9067 | 0.5741 |

Retained-candidate coverage uses the selected candidate floor and pre-NMS cap;
post-NMS coverage uses all final AP candidates. Both are one-to-many geometric
diagnostics, not confidence-thresholded one-to-one recall or AP. Frequent cap hits
and losses relative to the initial search identify limits introduced after
geometry generation. They do not by themselves distinguish score ranking from
every subsequent source of error.

![Measured success and failure examples](../results/m2/full/20261114_full/m2_success_failure_examples.jpg)

Examples were selected by stated criteria: lowest FP+FN among positive test
images with at least one true positive (ties favour more true positives), most missed annotated birds among the remaining images with at most 20 birds, most false
alarms among negative test images, and the largest flock. Successful example: `beach_bird_1816` (TP=1, FP=0, FN=0); Misses (at most 20 GT): `beach_bird_1806` (TP=0, FP=0, FN=20); Negative-scene example: `beach_bird_1829` (TP=0, FP=38, FN=0); Dense flock: `beach_bird_1790` (TP=0, FP=3, FN=66).
Green dashed boxes are annotations; orange boxes are predictions at the frozen operating
threshold. These examples are descriptive final analysis, not a basis for retuning.

Visual inspection shows that `beach_bird_1816` contains a large, isolated bird that
is detected correctly at IoU 0.5, although the fixed template omits part of the
bill and lower extremities. In `beach_bird_1806`, the twenty birds occupy far fewer
pixels near the waterline, and none survives the frozen score threshold. The
negative scene `beach_bird_1829` produces 38 false detections, many on branches and
stump edges, illustrating confusion between background structure and the learned
appearance cues. In the dense flying flock `beach_bird_1790`, all 66 annotated
birds are missed at the same operating threshold, with three unmatched detections.
These observations are consistent with limited small-object detail, background
confusion and coarse localisation. They do not isolate which individual feature
or postprocessing stage causes every error. Counts refer to the frozen score
threshold of 1.5; AP also includes lower-scoring detections.


The main design limitations are independently scored overlapping crops, finite
window geometry, aspect-ratio distortion from square resizing, limited detail in
tiny bird crops and possible suppression of adjacent birds. Robustness numbers
describe only two chosen distortions at one severity on twenty images. Five-image
timing estimates and a single random seed also limit precision. Comparisons with
M1 and M3 must be added by the group using identical test splits, evaluator limits
and clearly stated timing hardware; those methods are not implemented by this M2
contribution.

## 6. Conclusion

M2 provides a complete handcrafted-feature detector with learned linear
classification. Validation selected M2-C; it obtained test AP50
0.0296 with a measured mean serial latency of
10.247s per image on the five-image validation
timing sample. The experiments quantify the effect of colour features and
hard-negative retraining, together with small-object, density and fixed-distortion
behaviour. Future work should target the observed misses and false alarms using
aspect-preserving features, better candidate geometry or crowded-scene suppression,
and evaluate more seeds and timing images. These are future directions, not
improvements demonstrated in this run.

## 7. References and implementation attribution

[1] N. Dalal and B. Triggs, “Histograms of Oriented Gradients for Human Detection,”
IEEE CVPR, vol. 1, pp. 886–893, 2005. DOI: 10.1109/CVPR.2005.177.
[Paper](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf).

[2] scikit-image contributors, [`hog` documentation and linked source](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog).
The implementation reuses its compiled histogram kernel and vectorises matching
L2-Hys block normalisation; reference-equivalence checks are included.

[3] scikit-learn contributors, [`LinearSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html).

[4] scikit-learn contributors, [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

[5] K. Liu, *Beach Bird Detection Dataset v1.0*, Kaggle, September 2026.
[Dataset](https://www.kaggle.com/datasets/liugalhigh/beach-bird).

[6] C. Cortes and V. Vapnik, “Support-vector networks,” *Machine Learning*,
vol. 20, pp. 273–297, 1995. DOI: 10.1007/BF00994018.
[Publisher](https://link.springer.com/article/10.1007/BF00994018).

Additional software: NumPy, Pillow, PyYAML, joblib, tqdm, pandas, Matplotlib,
threadpoolctl, Jupyter and the team's pycocotools evaluation wrapper. The run's
`environment-lock.txt` records installed versions. Window search, sampling,
vectorised block normalisation, NMS, mining, shared scanning, analysis and
orchestration are local project code; the classifier and clustering solvers are
library implementations. Merge reference numbers with the group's bibliography.
