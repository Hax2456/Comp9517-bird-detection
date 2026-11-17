"""Generate the measured M2 report, figures and results notebook from a saved run.

Run after the experiment: python -m m2_ml.summarise_run --run-dir <run-directory>
This reads predictions and metrics without fitting models or repeating inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

from m2_ml.detector import Config, load_records, read_rgb, make_training_features, iou_one_to_many
from utils.data import REPO_ROOT


def table(frame, columns, digits=4):
    selected = frame[columns]
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.{digits}f}" if np.isfinite(v) else "N/A"
        return str(v)
    labels={"variant":"Variant","AP50_95":"AP50–95","AP50_small":"AP50 small",
        "AP50_medium":"AP50 medium","AP50_large":"AP50 large","images":"Images",
        "bird_boxes":"Bird boxes","density":"Birds/image","model":"Model",
        "training_pipeline_seconds":"Train pipeline (s)","svm_fit_seconds":"SVM fit (s)",
        "mean_inference_seconds":"Latency (s/img)","timed_images":"Timed images",
        "operating_score_floor":"Operating score","precision":"Precision","recall":"Recall",
        "count_MAE":"Count MAE","distortion":"Condition","ROC_AUC":"Crop ROC AUC",
        "crop_average_precision":"Crop AP","balanced_accuracy_at_zero":"Balanced accuracy at 0",
        "candidate_cap_images":"Images hitting cap","retained_candidate_recall":"Top-candidate coverage",
        "postNMS_geometric_recall":"Post-NMS coverage"}
    header = "| " + " | ".join(labels.get(name,name) for name in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    return "\n".join([header, rule] + ["| " + " | ".join(fmt(v) for v in row) + " |"
                    for row in selected.itertuples(index=False, name=None)])


def crop_diagnostic(out, manifest):
    """Classification of known GT crops is a diagnostic, never detector AP."""
    path=out/"crop_classification_diagnostic.csv"
    frame=pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(frame) and "M2-C" in set(frame.model):
        return frame
    val=load_records(Path(manifest["data_root"]),"val")
    X,y,_=make_training_features(val,manifest["templates"],Config(),progress=False)
    rows=[]
    for name in ["M2-A","M2-B","M2-C"]:
        p=out/f"{name}.joblib"
        if not p.exists():continue
        bundle=joblib.load(p)
        scores=bundle["pipeline"].decision_function(X[bundle["feature_name"]])
        rows.append({"model":name,"crop_split":"val","positive_crops":int((y==1).sum()),
            "negative_crops":int((y==0).sum()),"ROC_AUC":roc_auc_score(y,scores),
            "crop_average_precision":average_precision_score(y,scores),
            "balanced_accuracy_at_zero":balanced_accuracy_score(y,scores>=0)})
    frame=pd.concat([frame,pd.DataFrame(rows)],ignore_index=True)
    frame.to_csv(path,index=False)
    return frame


def candidate_diagnostic(out, manifest):
    """Geometric coverage through ranking/candidate caps and NMS, without refitting."""
    records=load_records(Path(manifest["data_root"]),"val")
    rows=[]
    for variant in [row["variant"] for row in manifest["validation_summary"]]:
        bundle=joblib.load(out/f"{variant}.joblib")
        predictions=json.loads((out/f"{variant}_val.json").read_text())
        by_image={r.path.stem:[] for r in records}
        for p in predictions:by_image[p["image_id"]].append(p["bbox"])
        key=f"{bundle['feature_name']}_C{bundle['config']['C']}" if variant!="M2-C" else "M2-C"
        stage="initial_validation" if variant!="M2-C" else "mined_validation"
        cap_hits,retained_hits,nms_hits,total=0,0,0,0
        for record in records:
            _,raw,_=joblib.load(out/"cache"/stage/f"{record.path.stem}.joblib")
            boxes,scores=raw[key]
            cap_hits+=int(len(boxes)==bundle["config"]["pre_nms_topk"])
            boxes=boxes[scores>=bundle["config"]["score_floor"]]
            final_boxes=np.asarray(by_image[record.path.stem],dtype=float).reshape(-1,4)
            for gt in record.boxes:
                retained_hits+=int(np.max(iou_one_to_many(gt,boxes),initial=0.0)>=0.5)
                nms_hits+=int(np.max(iou_one_to_many(gt,final_boxes),initial=0.0)>=0.5)
                total+=1
        rows.append({"variant":variant,"images":len(records),"candidate_cap_images":cap_hits,
            "bird_boxes":total,"retained_candidate_recall":retained_hits/total,
            "postNMS_geometric_recall":nms_hits/total})
    frame=pd.DataFrame(rows);frame.to_csv(out/"candidate_retention_diagnostic.csv",index=False)
    return frame


def comparison_figures(out, val, test, robust):
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    names=val.variant.tolist(); x=np.arange(len(names));width=.34
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    axes[0].bar(x-width/2,val.AP50,width,label="Validation")
    axes[0].bar(x+width/2,test.set_index("variant").loc[names].AP50,width,label="Final test")
    ceiling=min(1.0,max(.05,1.25*max(val.AP50.max(),test.AP50.max())))
    axes[0].set(xticks=x,xticklabels=names,ylim=(0,ceiling),ylabel="AP50 (0–1)",title="Full original splits")
    axes[0].legend()
    axes[1].bar(x,val.mean_inference_seconds,color="tab:orange")
    axes[1].set(xticks=x,xticklabels=names,ylabel="Seconds / image",
                title="Serial timing on 5 validation images")
    fig.tight_layout();fig.savefig(out/"m2_comparison.png",bbox_inches="tight");plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,3.6))
    for i,key in enumerate(["AP50_small","AP50_medium","AP50_large"]):
        ax.bar(x+(i-1)*.24,test.set_index("variant").loc[names,key],.24,label=key.replace("AP50_",""))
    ceiling=min(1.0,max(.05,1.25*np.nanmax(test[["AP50_small","AP50_medium","AP50_large"]].to_numpy())))
    ax.set(xticks=x,xticklabels=names,ylim=(0,ceiling),ylabel="Final-test AP50 (0–1)",title="Object-size breakdown")
    ax.legend();fig.tight_layout();fig.savefig(out/"m2_size_analysis.png",bbox_inches="tight");plt.close(fig)
    if len(robust):
        fig,ax=plt.subplots(figsize=(7,3.7))
        distortions=["clean","blur_sigma1","brightness_0.6"]
        for i,name in enumerate(names):
            values=robust[robust.variant==name].set_index("distortion").loc[distortions,"AP50"]
            ax.bar(np.arange(3)+(i-(len(names)-1)/2)*.24,values,.24,label=name)
        ax.set(xticks=np.arange(3),xticklabels=["Clean","Blur σ=1","Brightness ×0.6"],
               ylim=(0,min(1.0,max(.05,1.25*robust.AP50.max()))),ylabel="AP50 (0–1)",title="Frozen models; same 20 validation images")
        ax.legend();fig.tight_layout();fig.savefig(out/"m2_robustness.png",bbox_inches="tight");plt.close(fig)


def examples(out, data_root, variant, threshold):
    records=load_records(data_root,"test")
    by_id={r.path.stem:r for r in records}
    predictions=json.loads((out/f"{variant}_test.json").read_text())
    stats=pd.read_csv(out/f"{variant}_test_per_image.csv")
    positives=stats[stats.ground_truth>0]
    successful=positives[positives.TP>0]
    success_pool=successful if len(successful) else positives
    good=success_pool.assign(errors=success_pool.FP+success_pool.FN).sort_values(
        ["errors","TP"],ascending=[True,False]).iloc[0]
    ordinary=positives[(positives.ground_truth<=20) & (positives.image_id!=good.image_id)]
    missed=(ordinary if len(ordinary) else positives).sort_values(["FN","FP"],ascending=False).iloc[0]
    negative=stats[stats.ground_truth==0]
    false=(negative if len(negative) else stats).sort_values("FP",ascending=False).iloc[0]
    dense=stats.sort_values("ground_truth",ascending=False).iloc[0]
    chosen=[("Successful example",good),("Misses (at most 20 GT)",missed),
            ("Negative-scene example",false),("Dense flock",dense)]
    fig,axes=plt.subplots(2,2,figsize=(11,11))
    rows=[]
    for ax,(label,row) in zip(axes.ravel(),chosen):
        r=by_id[row.image_id];ax.imshow(read_rgb(r));ax.axis("off")
        for box in r.boxes:
            ax.add_patch(Rectangle(box[:2],box[2],box[3],fill=False,color="lime",lw=1.1,linestyle="--"))
        visible=[p for p in predictions if p["image_id"]==r.path.stem and p["score"]>=threshold]
        for p in visible:
            b=p["bbox"];ax.add_patch(Rectangle(b[:2],b[2],b[3],fill=False,color="darkorange",lw=1))
        ax.set_title(f"{label}: {r.path.stem}\nTP {int(row.TP)}, FP {int(row.FP)}, FN {int(row.FN)}",fontsize=10)
        rows.append({"role":label,**row.to_dict()})
    fig.suptitle(f"{variant}, score ≥ {threshold:g}; green = annotations, orange = detections",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(out/"m2_success_failure_examples.png",bbox_inches="tight")
    fig.savefig(out/"m2_success_failure_examples.jpg",bbox_inches="tight",
                pil_kwargs={"quality":92,"optimize":True})
    plt.close(fig);pd.DataFrame(rows).to_csv(out/"qualitative_examples.csv",index=False)
    return rows


def build_report(out, manifest, val, test, robust, selected, rows):
    final=manifest["selected_variant"]
    v=val.set_index("variant");t=test.set_index("variant")
    cfg=manifest["selected_config"]
    diagnostic=manifest["proposal_diagnostic"]
    source=manifest["mining_source"]
    matched=manifest["matched_mining_metrics"]
    delta_color=100*(v.loc["M2-B","AP50"]-v.loc["M2-A","AP50"])
    delta_mining=100*(v.loc["M2-C","AP50"]-v.loc[source,"AP50"]) if "M2-C" in v.index else float("nan")
    train_count=manifest["training_crop_counts"]
    density=pd.read_csv(out/"test_density.csv")
    density=density[density.variant==final]
    small=t.loc[final,"AP50_small"];medium=t.loc[final,"AP50_medium"]
    relation="lower" if small<medium else "higher"
    relative=out.relative_to(REPO_ROOT).as_posix()
    asset=f"../{relative}"
    color_finding="improved" if delta_color>0 else ("reduced" if delta_color<0 else "did not change")
    mining_finding="improved" if delta_mining>0 else ("reduced" if delta_mining<0 else "did not change")
    summary_columns=["variant","AP50","AP50_95","AP50_small","AP50_medium","AP50_large"]
    operating=["variant","operating_score_floor","precision","recall","F1","count_MAE"]
    timing=["variant","training_pipeline_seconds","svm_fit_seconds","mean_inference_seconds","timed_images"]
    example_text="; ".join(f"{r['role']}: `{r['image_id']}` (TP={int(r['TP'])}, FP={int(r['FP'])}, FN={int(r['FN'])})" for r in rows)
    crop=pd.read_csv(out/"crop_classification_diagnostic.csv")
    crop=crop[(crop.crop_split=="val") & crop.model.isin(["M2-A","M2-B","M2-C"])]
    retention=pd.read_csv(out/"candidate_retention_diagnostic.csv")
    negligible_color=("The colour difference is very small; a single seed without uncertainty estimates does not establish a reliable benefit."
                      if abs(delta_color)<0.5 else "")
    visual_notes=(out/"visual_review.md").read_text() if (out/"visual_review.md").exists() else ""
    visual_short=(out/"visual_review_short.md").read_text() if (out/"visual_review_short.md").exists() else ""
    text=f'''# M2: traditional machine-learning bird detection

**Measured full-run contribution.** This text is organised for insertion into the
group report. Results come from `{relative}`, using all 1,369 training images,
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

All {train_count['positive']:,} annotated training birds provided positive crops.
Up to 24 background windows per training image produced {train_count['negative']:,}
initial negatives. A negative window had to satisfy both IoU < 0.10 and intersection
over each bird's area < 0.10 against every annotation. The coverage restriction
prevents a large window containing a small bird from being incorrectly labelled
background. Empty-labelled training images were included.

Nine templates were learned by K-means on log training-box widths and heights,
with seed 42. The resulting pixel dimensions were
`{manifest['templates']}`. Templates slide across the original 512 × 512 image at
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
The decision score is $s(\\mathbf x)=\\mathbf w^\\top\\mathbf z(\\mathbf x)+b$,
where $\\mathbf z$ is the standardised descriptor. Scores rank detections and
are not calibrated probabilities. Both initial feature sets use identical crops.

Each initial variant receives the same validation grid: C ∈ {{0.1, 1.0}}, NMS IoU
∈ {{0.3, 0.5}}, and candidate score floor ∈ {{−2.0, −1.0}}. AP50 selects settings;
the first declared candidate wins a tie. At most 5,000 candidates enter NMS and
300 detections survive per image, matching the shared evaluator's detection cap.
The final selected configuration was **{final}**, C={cfg['C']}, NMS IoU={cfg['nms_iou']},
and AP candidate floor={cfg['score_floor']}.

### 2.4. Hard-negative retraining and computation

M2-C used {source} as its source and scanned a fixed random subset of
{len(manifest['mining_images'])} training images (seed 42). Windows with score ≥ 0
that met both background restrictions were mined, with NMS at 0.3, up to 12 per
image and 6,000 globally. This added {manifest['hard_negative_count']:,} negatives.
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
{diagnostic['recall']:.4f} ({diagnostic['covered_boxes']}/{diagnostic['total_boxes']})
over **all validation annotations**, measuring geometry coverage before classification.

Runs used local macOS ARM64, ten logical CPUs available, Python
{manifest['environment']['python']}, NumPy {manifest['environment']['packages']['numpy']},
scikit-image {manifest['environment']['packages']['scikit-image']}, scikit-learn
{manifest['environment']['packages']['scikit-learn']} and pycocotools
{manifest['environment']['packages']['pycocotools']}. Exact versions, image lists,
source hashes and configurations are saved with the run. Hardware-specific timing
cannot be compared directly with results measured on another machine.

## 4. Experimental results

### 4.1. Validation and final-test detection accuracy

Validation (291 images; model selection):

{table(val,summary_columns)}

Final test (240 images; frozen settings):

{table(test,summary_columns)}

Adding colour {color_finding} validation AP50 by {abs(delta_color):.2f} percentage
points. Hard-negative retraining {mining_finding} validation AP50 by
{abs(delta_mining):.2f} points relative to its {source} source after postprocessing
selection. With the source postprocessing unchanged, the mined model's AP50 was
{matched['AP50']:.4f}, versus {v.loc[source,'AP50']:.4f} for its source.
The validation-selected **{final}** achieved final-test AP50
**{t.loc[final,'AP50']:.4f}** and AP50–95 **{t.loc[final,'AP50_95']:.4f}**.
{negligible_color}

![M2 accuracy and latency comparison]({asset}/m2_comparison.png)

### 4.2. Timing

{table(val,timing,3)}

Training pipeline time includes the shared {manifest['shared_preparation_seconds']:.2f}s
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
{v.loc['M2-B','mean_inference_seconds']/v.loc['M2-A','mean_inference_seconds']:.2f}×
while its validation AP50 gain was only {delta_color:.2f} percentage points.
Per-window colour conversion/histogram extraction is a plausible explanation for
the added cost; this timing result is specific to this implementation and hardware.

### 4.3. Small objects, density and thresholded counts

![Final-test AP50 by size]({asset}/m2_size_analysis.png)

For {final}, small-bird test AP50 was {small:.4f}, {relation} than medium-bird
AP50 ({medium:.4f}). The selected model's density breakdown was:

{table(density,['density','images','bird_boxes','AP50','AP50_95'])}

The no-bird group's AP is undefined and reported as N/A. Dense-group results must
be interpreted with their sample counts, especially the very small 51+ group.
Each group AP is recomputed using only that group's images and false positives;
it is not an additive component of overall AP and should not be averaged across
groups. Higher AP in one small group does not establish that density helps detection.
For supplementary precision, recall, F1 and counting MAE, a margin threshold was
selected from a fixed grid by validation F1 and frozen before test. AP uses the
lower candidate floor and does not use this operating threshold.

{table(test,operating)}

For context, always predicting zero birds would have a test counting MAE of
{1518/240:.3f} birds/image. Counting MAE should therefore be interpreted with
precision and recall rather than presented alone as evidence of useful counting.

### 4.4. Fixed-distortion robustness

On the same fixed random subset of 20 validation images, frozen models were
evaluated on clean images, Gaussian blur (Pillow radius 1 pixel), and brightness
multiplied by 0.6. Labels and model settings were unchanged. This is a limited
validation stress test rather than a whole-test robustness claim.

{table(robust,['variant','distortion','images','AP50','AP50_95'])}

![M2 fixed-distortion robustness]({asset}/m2_robustness.png)

## 5. Discussion

The controlled colour comparison shows that appearance information {color_finding}
aggregate validation detection under this setup; it does not establish that colour
is reliable in every scene. Hard-negative retraining {mining_finding} AP50 relative
to its source, and the matched-threshold result separates the retraining comparison
from postprocessing reselection. Adding negatives also changes class-balanced
weights, so the experiment evaluates the complete retraining procedure rather
than isolating one causal mechanism.

A supplementary classification diagnostic uses the 1,843 annotated validation
bird crops and 6,984 randomly sampled background crops. No model is fitted or
selected using this diagnostic. Its task is easier because bird locations are
already supplied:

{table(crop,['model','ROC_AUC','crop_average_precision','balanced_accuracy_at_zero'])}

These crop metrics must not be substituted for detection AP. They explain why a
plausible patch classifier need not make a strong image detector: full-image
search evaluates tens of thousands of background and imperfectly aligned windows
per image, rather than a curated mixture of exact bird crops and random negatives.
Hard negatives address part of this distribution difference. Training with
IoU-controlled jittered positive windows is a targeted future experiment; it was
not performed here.

Candidate recall of {diagnostic['recall']:.4f} shows that the finite template search
can cover most validation annotations at IoU 0.5, while some remain unreachable
even before classification. Differences between candidate recall and achieved
AP cannot be attributed to geometry alone: descriptor ambiguity, background
confusion, localisation scores and NMS also matter. Size- and density-stratified
results and the following thresholded examples provide evidence of remaining
misses and false alarms.

Validation geometric coverage after model ranking and postprocessing was:

{table(retention,['variant','candidate_cap_images','retained_candidate_recall','postNMS_geometric_recall'])}

Retained-candidate coverage uses the selected candidate floor and pre-NMS cap;
post-NMS coverage uses all final AP candidates. Both are one-to-many geometric
diagnostics, not confidence-thresholded one-to-one recall or AP. Frequent cap hits
and losses relative to the initial search identify limits introduced after
geometry generation. They do not by themselves distinguish score ranking from
every subsequent source of error.

![Measured success and failure examples]({asset}/m2_success_failure_examples.jpg)

Examples were selected by stated criteria: lowest FP+FN among positive test
images with at least one true positive (ties favour more true positives), most missed annotated birds among the remaining images with at most 20 birds, most false
alarms among negative test images, and the largest flock. {example_text}.
Green dashed boxes are annotations; orange boxes are predictions at the frozen operating
threshold. These examples are descriptive final analysis, not a basis for retuning.

{visual_notes}

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
classification. Validation selected {final}; it obtained test AP50
{t.loc[final,'AP50']:.4f} with a measured mean serial latency of
{v.loc[final,'mean_inference_seconds']:.3f}s per image on the five-image validation
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
'''
    (REPO_ROOT/"m2_ml/M2_report.md").write_text(text)
    selected_robust=robust[robust.variant==final].set_index("distortion")
    compact=f'''# M2 contribution for the group report

This concise contribution is based on the completed run `{relative}`. Merge its
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
is C ∈ {{0.1,1.0}}, NMS IoU ∈ {{0.3,0.5}}, and score floor ∈ {{−2,−1}}.
Greedy NMS processes at most 5,000 candidates and retains at most 300 boxes.

M2-C starts from the better validated initial representation ({source}), mines
training-only false alarms with nonnegative scores, and refits the scaler/SVM
while retaining the original samples. A fixed 160-image training subset (seed 42)
and up to 12 eligible negatives per image produced {manifest['hard_negative_count']:,}
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

{table(test,['variant','AP50','AP50_95','mean_inference_seconds'])}

Accuracy uses all 240 test images. The latency column is measured independently
on the same five fixed validation images per variant in serial on macOS ARM64,
with one numerical thread and no worker concurrency. It includes image loading
through NMS/prediction construction and excludes model loading, disk writing and
evaluation; it is not a full-test latency benchmark. Initial shared feature/template
preparation took {manifest['shared_preparation_seconds']:.2f}s. Pipeline training
times were {', '.join(f"{name}: {v.loc[name,'training_pipeline_seconds']:.2f}s" for name in v.index)};
M2-C includes source fitting, parallel mining wall time and refitting. Validation
search is excluded, and paired feature preparation is charged equally to all models.

Colour {color_finding} validation AP50 by {abs(delta_color):.2f} percentage points.
Mining {mining_finding} it by {abs(delta_mining):.2f} points relative to {source}.
With unchanged source postprocessing, mined AP50 was {matched['AP50']:.4f} versus
{v.loc[source,'AP50']:.4f}. Validation selected **{final}**, with C={cfg['C']},
NMS IoU={cfg['nms_iou']} and candidate floor={cfg['score_floor']}; its final-test
AP50 was **{t.loc[final,'AP50']:.4f}**. The small colour difference is not evidence
of a reliable gain from a single-seed run.

## Discussion contribution

Candidate-window recall was {diagnostic['recall']:.4f} over all validation boxes,
whereas detection AP was much lower. Supplementary classification of known
validation bird crops and sampled backgrounds performs much better than full
detection, but these are different tasks and metrics. Exact annotated crops omit
the localisation errors and abundance of difficult windows encountered at search
time. The results therefore motivate hard-negative learning and a future
IoU-controlled positive-jitter experiment rather than treating crop classification
as evidence of a strong detector.

For {final}, test AP50 was {small:.4f} for small birds and {medium:.4f} for medium
birds. Density results and real success/failure overlays are provided with sample
counts in the detailed report. The 51+ test group has only two images, limiting
generalisability. On a fixed 20-image validation subset, frozen {final} scored
AP50 {selected_robust.loc['clean','AP50']:.4f} clean,
{selected_robust.loc['blur_sigma1','AP50']:.4f} under radius-1 Gaussian blur and
{selected_robust.loc['brightness_0.6','AP50']:.4f} at brightness ×0.6. These limited
stress tests do not establish broad robustness.

{visual_short}

Include the measured accuracy/time figure and representative success/failure
overlay from `{relative}`. The main limitations are background confusion,
finite template geometry, square-resize distortion, little detail in small crops,
NMS interactions and repeated overlapping-window computation. Compare M2 with
the group's M1/M3 results using identical split/evaluator settings and explicit
timing hardware. These three M2 ablations constitute one method family.

## Conclusion contribution

The validation-selected {final} achieved test AP50 {t.loc[final,'AP50']:.4f} and
mean serial latency {v.loc[final,'mean_inference_seconds']:.3f}s on the five-image
validation timing sample. The controlled experiments quantify the limited benefit
of colour and the effect of hard-negative retraining. Future work should address
search-time background and alignment differences with better positive sampling,
hard-negative strategies and candidate geometry, followed by evaluation under a
fresh predeclared validation protocol. Final-test scores from this run must not
be used for further tuning.

References [1–6] are listed in `M2_report.md`; merge them into the group bibliography.
'''
    (REPO_ROOT/"m2_ml/M2_report_for_group.md").write_text(compact)


def build_results_notebook(out, manifest):
    import nbformat as nbf
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()))
    def code(s):cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip()))
    relative=out.relative_to(REPO_ROOT).as_posix()
    md(f'''# M2 — measured full-dataset results

This notebook presents actual outputs from **`{relative}`**. It does not retrain
models or retune test results. The full implementation and 20-section explanation
are in `M2_walkthrough.ipynb`; section-by-section report prose is in `M2_report.md`.
All three variants were frozen using validation data before one final test batch.

**Selected by validation AP50:** {manifest['selected_variant']}.
''')
    code(f'''
    from pathlib import Path
    import json
    import pandas as pd
    from IPython.display import display, Image, Markdown
    roots=[Path.cwd(),*Path.cwd().parents]
    ROOT=next(p for base in roots for p in [base,base/"Comp9517-bird-detection"]
              if (p/"m2_ml/detector.py").exists())
    RUN=ROOT/{relative!r}
    manifest=json.loads((RUN/"manifest.json").read_text())
    display(Markdown("**Run:** `"+str(RUN)+"`"))
    ''')
    md('''## 1. Dataset and leakage checks

Training uses every original training image. Validation selects settings and the
final variant. The held-out test is evaluated after selection is frozen. No crops
move between splits. The manifest's source-group audit checks the supplied split.
''')
    code('''
    display(pd.DataFrame(json.loads((RUN/"data_audit.json").read_text())["splits"]))
    display(pd.DataFrame([manifest["training_crop_counts"]]))
    print("Training images:",len(manifest["train_images"]))
    print("Validation images:",len(manifest["validation_images"]))
    print("Test images:",len(manifest["test_images"]))
    print("Window templates:",manifest["templates"])
    print("Candidate recall:",manifest["proposal_diagnostic"])
    ''')
    md('''## 2. How the detector works

Bird and background crops become fixed-size 32 × 32 patches. HOG describes edge
directions in 1,764 numbers. Optional HSV histograms add 24 colour numbers. A
training-fitted standardiser and linear SVM turn those features into a bird score.
The detector slides training-derived templates across each image, then NMS removes
duplicate boxes. M2-C refits the model with confusing background examples mined
from a fixed 160-image training subset. Both original classes of samples are retained.

Shared features and vectorised HOG accelerate the same descriptor calculation;
tests compare against reference HOG and independent model scoring.
''')
    code('''
    frozen=json.loads((RUN/"FROZEN_SELECTION.json").read_text())
    display(pd.DataFrame(frozen["configs"]).T)
    print("Mining source:",manifest["mining_source"])
    print("Mined negative samples:",manifest["hard_negative_count"])
    print("Matched-postprocessing mined-model metrics:",manifest["matched_mining_metrics"])
    ''')
    md('''## 3. Validation and final-test results

AP50 is the project's mAP@0.5 metric for one class. AP50–95 tests stricter box
alignment. Small/medium/large AP50 use COCO area bins. The final variant is chosen
from validation AP50; the test table is a final comparison of already-frozen models.
''')
    code('''
    val=pd.read_csv(RUN/"validation_summary.csv")
    test=pd.read_csv(RUN/"test_summary.csv")
    columns=["variant","AP50","AP50_95","AP50_small","AP50_medium","AP50_large"]
    display(Markdown("**Validation: 291 images**"));display(val[columns].round(4))
    display(Markdown("**Final test: 240 images**"));display(test[columns].round(4))
    display(Image(filename=str(RUN/"m2_comparison.png")))
    ''')
    md('''## 4. Timing boundaries

Training time includes shared paired feature preparation and selected fitting;
M2-C adds source fitting, eight-process mining wall time and refitting. Validation
search is separate. The original HOG-only variant is charged the same paired
preparation cost, so this is an experimental-pipeline time.

Standalone inference was timed in serial on the same five randomly selected
validation images for each model, with no concurrent workers and one numerical
library thread. It includes image loading through NMS and prediction construction,
and excludes model loading, disk output and evaluation. These are five-image
validation latency estimates, not timing measurements across all test images.
''')
    code('''
    display(val[["variant","training_pipeline_seconds","svm_fit_seconds",
                 "mean_inference_seconds","median_inference_seconds","timed_images"]].round(3))
    display(pd.read_csv(RUN/"serial_inference_times.csv").round(3))
    print("Software environment:",manifest["environment"])
    ''')
    md('''### Why good crop classification does not imply good detection

This supplementary diagnostic uses ground-truth-localised positive crops and
randomly sampled backgrounds. Models are only scored, not refitted. These crop
ROC/AP/accuracy values describe a different, easier task than image detection;
they must not be reported as detector AP. Full search exposes many more difficult
backgrounds and imperfectly aligned candidate windows.
''')
    code('''
    crop=pd.read_csv(RUN/"crop_classification_diagnostic.csv")
    display(crop[(crop.crop_split=="val") & crop.model.isin(["M2-A","M2-B","M2-C"])].round(4))
    ''')
    md('''### Candidate coverage after ranking and NMS

The original search covers 95.66% of validation annotations at IoU 0.5. The table
below measures geometric coverage after each model's score floor/candidate cap
and after NMS. Coverage allows one candidate to overlap more than one annotation;
it is a diagnostic, not confidence-thresholded one-to-one recall or AP. All models
hit the 5,000-candidate cap on every validation image in this run.
''')
    code('''
    display(pd.read_csv(RUN/"candidate_retention_diagnostic.csv").round(4))
    ''')
    md('''## 5. Small objects, dense scenes and counting

Density results report sample sizes; the 51+ test group is tiny. AP is undefined
for no-bird images. Supplementary precision/recall/F1 and counting MAE use a
validation-selected margin threshold frozen for each model before testing. That
threshold is distinct from the lower AP candidate floor.
''')
    code('''
    display(Image(filename=str(RUN/"m2_size_analysis.png")))
    density=pd.read_csv(RUN/"test_density.csv")
    display(density[density.variant==manifest["selected_variant"]].round(4))
    display(test[["variant","operating_score_floor","precision","recall","F1","count_MAE"]].round(4))
    ''')
    md('''## 6. Fixed-distortion robustness

The exact same 20 validation images are used for clean, Gaussian blur (radius 1
pixel) and brightness ×0.6 versions. Models, labels and thresholds stay fixed.
This is a small stress test with one severity, not a broad robustness guarantee.
''')
    code('''
    display(pd.read_csv(RUN/"robustness_summary.csv").round(4))
    display(Image(filename=str(RUN/"m2_robustness.png")))
    ''')
    md('''## 7. Successes and failures

Examples use explicit criteria: a positive image with at least one true detection
and the fewest errors, the most
missed birds among remaining images with at most 20 annotations, the negative
image with the most false alarms, and the densest flock.
They illustrate frozen final outputs and are not used to tune test settings.
Green dashed boxes are annotations; orange boxes are detections at the frozen operating
threshold. Read the TP/FP/FN counts alongside the image.
''')
    code('''
    display(pd.read_csv(RUN/"qualitative_examples.csv"))
    display(Image(filename=str(RUN/"m2_success_failure_examples.jpg")))
    if (RUN/"visual_review.md").exists():
        display(Markdown((RUN/"visual_review.md").read_text()))
    ''')
    md('''## 8. Requirements and rubric review

M2 supplies literature motivation, implemented methods, controlled comparisons,
AP50 and timing, successes and failures, discussion, limitations, references and
documented code. Robustness and size/density analyses add depth. These results do
not replace the group's M1/M3 comparison, IEEE two-column PDF assembly, video or
contribution requirements. See `M2_requirements_review.md` for the exact mapping.

The source-code submission must exclude datasets, trained weights and result
images. The group report limit is ten pages of main text excluding references.
Metrics are not the sole grading basis: the rubric emphasises soundness,
evaluation rigour and discussion. All claims in the report should match the
saved numbers and actual example images.
''')
    md('''## 9. Reproduce or inspect the run

```bash
python -m pip install -r requirements-m2.txt
python -m unittest m2_ml.test_detector -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m m2_ml.run_full \\
  --run-dir results/m2/full/NEW_RUN_NAME --workers 8 \\
  --mining-images 160 --benchmark-images 5 --robustness-images 20
```

Use a new run name for an independent training experiment. Add `--test` only when
the validation-based protocol and settings are final. The completed run here
already contains final test results; inspecting or reformatting saved files does
not require running inference again. `protocol.json`, `manifest.json`,
`environment-lock.txt`, `FROZEN_SELECTION.json`, per-image predictions and timing
CSVs preserve evidence. The source-code tests and earlier synthetic checks do not
provide bird performance scores; all tables in this notebook use the real dataset.

References: [Dalal & Triggs](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf),
[scikit-image HOG](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog),
[LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html),
[StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).
''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"name":"python3","display_name":"Python 3","language":"python"},
                                              "language_info":{"name":"python"}})
    nbf.validate(nb);nbf.write(nb,REPO_ROOT/"m2_ml/M2_full_results.ipynb")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    args=parser.parse_args();out=args.run_dir.resolve()
    manifest=json.loads((out/"manifest.json").read_text())
    if not manifest["test_evaluated"]:
        raise RuntimeError("The measured final report requires completed final test results")
    val=pd.read_csv(out/"validation_summary.csv");test=pd.read_csv(out/"test_summary.csv")
    robust=pd.read_csv(out/"robustness_summary.csv")
    selected=manifest["selected_variant"]
    threshold=float(test[test.variant==selected].operating_score_floor.iloc[0])
    crop_diagnostic(out,manifest)
    candidate_diagnostic(out,manifest)
    comparison_figures(out,val,test,robust)
    rows=examples(out,Path(manifest["data_root"]),selected,threshold)
    build_report(out,manifest,val,test,robust,selected,rows)
    build_results_notebook(out,manifest)
    print("Measured report, figures and results notebook generated from",out)


if __name__=="__main__":
    main()
