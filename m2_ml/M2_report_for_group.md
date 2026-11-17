# M2 contribution for the group report

This concise contribution is based on the completed run `results/m2/full/20261114_full`. Merge its
subsections into the matching sections of the group's two-column IEEE report.
The full method, evidence and references are in `M2_report.md`; the overall group
report still needs M1/M3 results and must fit ten pages of main text.

## Introduction contribution

M2 investigates handcrafted features and learned classification for bird
localisation amid variable coastal backgrounds, object sizes and flock densities.
This interpretable CPU baseline complements the group's non-learning and deep
approaches and separates the challenges of crop classification and image detection.

## Literature review contribution

Handcrafted-feature detectors separate image representation from classification.
Dalal and Triggs [1] combine histograms of oriented gradients (HOG) with a linear
support vector machine (SVM) and sliding-window search. HOG captures local edge
structure with overlapping contrast-normalised blocks. We adapt this framework
to single-class bird detection, investigate complementary HSV colour histograms,
and retrain on difficult background false alarms. The underlying SVM framework
is described by Cortes and Vapnik [6]; the exact library solver and descriptor
implementations are attributed in [2–4].

## Methods contribution

M2 searches nine window templates learned by K-means on log widths/heights of
training bird boxes. Quarter-window strides, with a two-pixel minimum and explicit
image-edge positions, produce 47,527 windows per 512 × 512 image. Every crop is
resized to 32 × 32. HOG uses nine unsigned bins, 4 × 4-pixel cells, 2 × 2-cell
blocks and L2-Hys normalisation, giving 1,764 features. M2-B adds 24 independently
normalised HSV histogram features to M2-A's HOG descriptor.

All 8,663 training bird boxes provide positive crops; 24 sampled backgrounds per
training image provide 32,856 negatives. A negative must have IoU < 0.10 with every
bird and cover less than 10% of every bird's area. A training-fitted standardiser
precedes a class-balanced, L2-regularised linear SVM with squared-hinge loss.
Decision values rank detections and are not probabilities. The validation grid
is C ∈ {0.1,1.0}, NMS IoU ∈ {0.3,0.5}, and score floor ∈ {−2,−1}.
Greedy NMS processes at most 5,000 candidates and retains at most 300 boxes.

M2-C starts from the better validated initial representation (M2-B), mines
training-only false alarms with nonnegative scores, and refits the scaler/SVM
while retaining the original samples. A fixed 160-image training subset (seed 42)
and up to 12 eligible negatives per image produced 1,920
additional negatives. Source C is held fixed; postprocessing is reselected on
validation and also evaluated unchanged for a controlled comparison.

## Experimental setup and results contribution

The original 1,369/291/240 train/validation/test images and 8,663/1,843/1,518 boxes
were kept separate. Source-group and image-ID audits found no cross-split overlap.
Training data fit templates, features/scaler and classifier; validation selected
settings and the final variant. Every variant and supplementary counting threshold
was frozen before one final batch evaluated the original test split. AP50 is the
project's single-class mAP@0.5, measured with the shared pycocotools evaluator and
300-detection limit.

| Variant | AP50 | AP50–95 | Latency (s/img) |
| --- | --- | --- | --- |
| M2-A | 0.0122 | 0.0030 | 3.3347 |
| M2-B | 0.0153 | 0.0036 | 10.2144 |
| M2-C | 0.0296 | 0.0071 | 10.2466 |

Accuracy uses all 240 test images. The latency column is measured independently
on the same five fixed validation images per variant in serial on macOS ARM64,
with one numerical thread and no worker concurrency. It includes image loading
through NMS/prediction construction and excludes model loading, disk writing and
evaluation; it is not a full-test latency benchmark. Initial shared feature/template
preparation took 12.11s. Pipeline training
times were M2-A: 26.01s, M2-B: 25.86s, M2-C: 469.32s;
M2-C includes source fitting, parallel mining wall time and refitting. Validation
search is excluded, and paired feature preparation is charged equally to all models.

Colour improved validation AP50 by 0.01 percentage points.
Mining improved it by 2.76 points relative to M2-B.
With unchanged source postprocessing, mined AP50 was 0.0448 versus
0.0172. Validation selected **M2-C**, with C=1.0,
NMS IoU=0.5 and candidate floor=-2.0; its final-test
AP50 was **0.0296**. The small colour difference is not evidence
of a reliable gain from a single-seed run.

## Discussion contribution

Candidate-window recall was 0.9566 over all validation boxes,
whereas detection AP was much lower. Supplementary classification of known
validation bird crops and sampled backgrounds performs much better than full
detection, but these are different tasks and metrics. Exact annotated crops omit
the localisation errors and abundance of difficult windows encountered at search
time. The results therefore motivate hard-negative learning and a future
IoU-controlled positive-jitter experiment rather than treating crop classification
as evidence of a strong detector.

For M2-C, test AP50 was 0.0243 for small birds and 0.0322 for medium
birds. Density results and real success/failure overlays are provided with sample
counts in the detailed report. The 51+ test group has only two images, limiting
generalisability. On a fixed 20-image validation subset, frozen M2-C scored
AP50 0.0259 clean,
0.0331 under radius-1 Gaussian blur and
0.0340 at brightness ×0.6. These limited
stress tests do not establish broad robustness.

The selected examples show one correctly detected large isolated bird, twenty
missed birds near a waterline, 38 false detections on a negative tree/branch scene,
and a 66-bird flying flock with no true detection at the frozen operating threshold.
These support the observed small-object and background-confusion limitations;
they are illustrative cases, not a separate statistical estimate.


Include the measured accuracy/time figure and representative success/failure
overlay from `results/m2/full/20261114_full`. The main limitations are background confusion,
finite template geometry, square-resize distortion, little detail in small crops,
NMS interactions and repeated overlapping-window computation. Compare M2 with
the group's M1/M3 results using identical split/evaluator settings and explicit
timing hardware. These three M2 ablations constitute one method family.

## Conclusion contribution

The validation-selected M2-C achieved test AP50 0.0296 and
mean serial latency 10.247s on the five-image
validation timing sample. The controlled experiments quantify the limited benefit
of colour and the effect of hard-negative retraining. Future work should address
search-time background and alignment differences with better positive sampling,
hard-negative strategies and candidate geometry, followed by evaluation under a
fresh predeclared validation protocol. Final-test scores from this run must not
be used for further tuning.

References [1–6] are listed in `M2_report.md`; merge them into the group bibliography.
