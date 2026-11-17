"""Write the revised report/notebook from a completed, separately saved retry."""
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
import nbformat as nbf

from m2_ml.detector import load_records,read_rgb
from m2_ml.run_full import match_predictions,write_json
from m2_ml.summarise_run import table
from utils.data import REPO_ROOT,get_data_root


def visualise(out,selected,cfg):
    records=load_records(get_data_root(),"val")
    predictions=json.loads((out/f"{selected}_val.json").read_text())
    trials=[]
    for value in [-1.,-.5,0.,.5,1.,1.5,2.,2.5,3.]:
        _,stats=match_predictions(predictions,records,value);trials.append(stats)
    best=max(trials,key=lambda row:row["F1"])
    stats,_=match_predictions(predictions,records,best["operating_score_floor"])
    stats.to_csv(out/"selected_val_per_image.csv",index=False)
    write_json(out/"display_threshold_validation_only.json",best)
    positive=stats[stats.TP>0]
    pool=positive if len(positive) else stats[stats.ground_truth>0]
    successful=pool.assign(errors=pool.FP+pool.FN).sort_values(["errors","TP"],ascending=[True,False]).iloc[0]
    ordinary=stats[(stats.ground_truth>0)&(stats.ground_truth<=20)&(stats.image_id!=successful.image_id)]
    missed=ordinary.sort_values("FN",ascending=False).iloc[0]
    negative=stats[stats.ground_truth==0].sort_values("FP",ascending=False).iloc[0]
    dense=stats.sort_values("ground_truth",ascending=False).iloc[0]
    chosen=[("Successful detection",successful),("Misses (≤20 birds)",missed),
            ("Negative scene",negative),("Dense flock",dense)]
    by_id={r.path.stem:r for r in records};fig,axes=plt.subplots(2,2,figsize=(10,10))
    for ax,(role,row) in zip(axes.ravel(),chosen):
        r=by_id[row.image_id];ax.imshow(read_rgb(r));ax.axis("off")
        for b in r.boxes:ax.add_patch(Rectangle(b[:2],b[2],b[3],fill=False,color="lime",linestyle="--",lw=1))
        for p in predictions:
            if p["image_id"]==r.path.stem and p["score"]>=best["operating_score_floor"]:
                b=p["bbox"];ax.add_patch(Rectangle(b[:2],b[2],b[3],fill=False,color="darkorange",lw=1))
        ax.set_title(f"{role}: {row.image_id}\nTP {int(row.TP)}, FP {int(row.FP)}, FN {int(row.FN)}",fontsize=10)
    fig.suptitle(f"Validation: {selected}, score ≥ {best['operating_score_floor']:g}\nGreen dashed = GT; orange = predictions",fontsize=11)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/"revision_examples.jpg",dpi=130,
        bbox_inches="tight",pil_kwargs={"quality":92,"optimize":True});plt.close(fig)
    pd.DataFrame([{"role":role,**row.to_dict()} for role,row in chosen]).to_csv(out/"example_selection.csv",index=False)
    return best


def plots(out,val,test,old):
    names=val.variant.tolist();x=np.arange(len(names))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].bar(x-.18,val.AP50,.36,label="Validation")
    if len(test):axes[0].bar(x+.18,test.set_index("variant").loc[names].AP50,.36,label="Previously seen test")
    axes[0].axhline(old,color="black",linestyle="--",lw=1,label="Original M2-C validation")
    axes[0].set(xticks=x,xticklabels=[n.split('-')[0] for n in names],ylabel="AP50 (0–1)",
                title="Revised traditional detectors",ylim=(0,max(.1,1.25*max(val.AP50.max(),test.AP50.max() if len(test) else 0))))
    axes[0].legend(fontsize=8)
    axes[1].bar(x,val.mean_inference_seconds,color="tab:orange")
    axes[1].set(xticks=x,xticklabels=[n.split('-')[0] for n in names],ylabel="Seconds / image",
                title="Serial CPU timing: 3 validation images")
    fig.tight_layout();fig.savefig(out/"revision_comparison.png",dpi=140,bbox_inches="tight");plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",type=Path,required=True)
    out=p.parse_args().out.resolve()
    manifest=json.loads((out/"manifest.json").read_text());protocol=json.loads((out/"protocol.json").read_text())
    val=pd.read_csv(out/"validation_summary.csv")
    if len(protocol["train_images"])!=1369 or len(protocol["val_images"])!=291:
        raise ValueError("This full-run report requires all original training and validation images")
    test=pd.read_csv(out/"followup_test_summary.csv") if (out/"followup_test_summary.csv").exists() else pd.DataFrame()
    audit=pd.read_csv(out/"training_audit.csv");selected=manifest["selected"]
    old_dir=REPO_ROOT/"results/m2/full/20261114_full"
    old_val_row=pd.read_csv(old_dir/"validation_summary.csv").set_index("variant").loc["M2-C"]
    old_val=float(old_val_row.AP50)
    old_test=pd.read_csv(old_dir/"test_summary.csv").set_index("variant").loc["M2-C","AP50"]
    selected_val=float(val[val.variant==selected].AP50.iloc[0])
    selected_row=val.set_index("variant").loc[selected]
    visual_notes=(out/"visual_review.md").read_text() if (out/"visual_review.md").exists() else ""
    selected_test=float(test[test.variant==selected].AP50.iloc[0]) if len(test) else float("nan")
    change=100*(selected_val-old_val)
    recommendation=(f"The revised {selected} has the highest observed validation AP50 across the original and revised comparisons."
        if selected_val>=old_val else
        "The original M2-C remains the stronger accuracy choice by validation AP50. The revised detector should not replace it on accuracy grounds; its speed and methodological comparisons remain useful results.")
    plots(out,val,test,old_val);operating=visualise(out,selected,protocol["config"])
    cols=["variant","AP50","AP50_95","AP50_small","AP50_medium","AP50_large"]
    relative=out.relative_to(REPO_ROOT).as_posix();asset=f"../{relative}"
    text=f'''# M2 retry: dense HOG/SVM detection and comparison with M3

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
{old_val:.4f}, despite much higher classification performance on known bird crops.
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
training set contains {manifest['positive_windows']:,} positive windows and
{manifest['training_windows']-manifest['positive_windows']:,} negatives; candidate
positives cover {int(audit.birds_with_positive_windows.sum())}/{int(audit.birds.sum())}
annotated training birds. HOG is already L2-Hys normalised, and colour histograms
are normalised to unit sum, so the revised linear SVM does not apply per-dimension
standardisation. C ∈ {{0.01,0.1,1.0}} and NMS IoU ∈ {{0.3,0.5}} are selected by
validation AP50. Initial class weighting is uniform. Scores are margins, not
probabilities. The candidate score floor is −1.5, with a 6,000-candidate cap and
300 final detections per image. These limits can restrict recall.

**Colour.** R2 adds 24 HSV histogram values. Cell histograms/integral sums provide
the exact histogram for the same 48 × 48 window without computing it separately
for every crop. Direct-histogram equivalence is tested.

**One hard-negative round.** The better validated initial model
({manifest['source_for_mining']}) scans training images only. Up to 32 false-alarm
windows per image are retained after background checks and NMS; a seeded global
cap of 40,000 prevents unbounded training growth. This produced
{manifest['hard_negatives']:,} mined negatives. All original examples are retained
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
`{relative}`. The screenshot's results are not substituted for our measurements.

### Validation: model selection

{table(val,cols)}

Validation selected **{selected}**, AP50 **{selected_val:.4f}**. This is a signed
change of **{change:+.2f} percentage points** from the original M2-C validation
AP50 ({old_val:.4f}). There is no assertion that every tested modification helps.
{recommendation}

The size breakdown is mixed: selected-revision validation AP50 is
{selected_row.AP50_medium:.4f} for medium birds versus {old_val_row.AP50_medium:.4f}
originally, but {selected_row.AP50_small:.4f} for small birds versus
{old_val_row.AP50_small:.4f}. This is not a uniform accuracy improvement.

### Follow-up evaluation on the previously seen official test set

{table(test,cols) if len(test) else 'No revised test inference was performed.'}

The validation-selected model scored AP50 {selected_test:.4f}; the original M2-C
test AP50 was {old_test:.4f}. Models and NMS settings were frozen in
`FROZEN_REVISION.json` before this follow-up. The test set's previous exposure
remains a limitation; these are not new independent generalisation estimates.

### CPU inference timing

{table(val,['variant','mean_inference_seconds'],3)}

Timing includes loading, dense feature maps, scores, optional refinement/reranking
and NMS, using the same three fixed validation images per revised variant in
serial, with one numerical thread. Original M2 timing used five images, so a
precise cross-run speedup should not be inferred from two small timing samples.
Training feature preparation took {manifest['preparation_wall_seconds']:.1f}s
and the six-worker mining pass took {manifest['mining_wall_seconds']:.1f}s.
These wall times are not per-image serial latency.

![Revised results and timing]({asset}/revision_comparison.png)

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

The display score {operating['operating_score_floor']:g} is chosen from cached
validation predictions by F1; it changes the visualisation, not the AP ranking
or trained model. Cases are selected by explicit criteria: a positive image with
at least one TP and few errors, the most missed birds among remaining images with
at most twenty birds, the negative image with most FP, and the densest flock.

![Validation examples]({asset}/revision_examples.jpg)

{visual_notes}

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
variants. The selected revised method is {selected}, validation AP50
{selected_val:.4f}, with a declared follow-up test AP50 of {selected_test:.4f}.
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
'''
    (REPO_ROOT/"m2_ml/M2_revision_report.md").write_text(text)
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()))
    def code(s):cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip()))
    md(f'''# M2 retry: methods, real results, and why M3 differs

Selected revised variant: **{selected}**. {recommendation} This notebook reads the completed
separate run `{relative}`. The old report/results are preserved. M3's screenshot
is a reported reference, not a reproduced experiment. The official test set was
previously evaluated; any revised test score is explicitly a follow-up.
''')
    code(f'''
    from pathlib import Path
    import json
    import pandas as pd
    from IPython.display import display,Image,Markdown
    roots=[Path.cwd(),*Path.cwd().parents]
    ROOT=next(p for base in roots for p in [base,base/"Comp9517-bird-detection"] if (p/"m2_ml/dense_detector.py").exists())
    RUN=ROOT/{relative!r}
    protocol=json.loads((RUN/"protocol.json").read_text())
    manifest=json.loads((RUN/"manifest.json").read_text())
    ''')
    md('''## 1. The four reference steps, implemented as a detector

1. Search multiple window sizes/aspect ratios.
2. Describe each window with HOG, optionally HSV histograms.
3. Train an SVM on bird/background windows, then mine difficult training false alarms.
4. Apply score-ordered NMS to remove duplicates.

The crucial correction is **what counts as a training window**. Tight GT crops
alone are unlike the shifted, contextual windows encountered by an image detector.
Here positives and negatives come from the same dense maps used for detection.
''')
    md('''## 2. Dense features and matching geometry

A 48 × 48 feature window includes context around its central 36 × 36 predicted
box. Six-pixel cells, 2 × 2-cell blocks and nine bins give 1,764 HOG values.
Up to four matched positions/scales per bird provide positive variation. Integral
cell histograms add 24 HSV values efficiently. The full level is described once;
overlapping descriptors are sliced from it instead of recomputed per crop.
''')
    code('''
    display(pd.DataFrame([protocol["config"]]).T)
    audit=pd.read_csv(RUN/"training_audit.csv")
    display(audit[["birds","birds_with_positive_windows","positive_windows","negative_windows"]].sum().to_frame("Count"))
    print("Templates:",protocol["templates"])
    ''')
    md('''## 3. Controlled variants and additional classical experiments

R1/R2 compare HOG with HOG+HSV using identical windows. R3 adds one training-only
mining pass. R4 adds ridge coordinate refinement; it remains a handcrafted-feature
method. R5 uses train-only PCA and an approximate RBF SVM to rerank initial
proposals. Those last two stages are explicit additional ablations, not hidden
changes presented as the basic screenshot method. No neural features or
pretrained object-detector weights are used by M2.
''')
    code('''
    val=pd.read_csv(RUN/"validation_summary.csv")
    display(val.round(4))
    display(pd.read_csv(RUN/"validation_search.csv").round(4))
    display(Image(filename=str(RUN/"revision_comparison.png")))
    ''')
    md('''## 4. Follow-up test and timing boundaries

Selection uses validation only; `FROZEN_REVISION.json` predates follow-up test
inference. Because this test set and failure images were already inspected in the
original work, this is not a new untouched holdout. Serial latency uses three
validation images, with one numerical thread. It includes image loading through
postprocessing. It is different from model-only GPU milliseconds.
''')
    code('''
    if (RUN/"followup_test_summary.csv").exists():display(pd.read_csv(RUN/"followup_test_summary.csv").round(4))
    display(pd.read_csv(RUN/"serial_times.csv").round(4))
    print("Frozen choices:",json.loads((RUN/"FROZEN_REVISION.json").read_text()))
    ''')
    md('''## 5. Why M3 can be much stronger

M3 learns image features and coordinates. Standard YOLO11 checkpoints transfer
COCO knowledge, including a bird class; the screenshot's actual initialisation
needs confirmation. RT-DETR learns multiscale/contextual representations and
uses transformer predictions. Fixed HOG cannot acquire those learned semantics
simply by adding NMS. Dense sharing reduces computation but does not change this
representation difference.

The local folder only supplied M3 inference, not its training or weights. Its
loader is now architecture-aware and records settings/timing metadata. Read the
audit below for confirmed facts, unknowns and sources.
''')
    code('''
    display(Markdown((ROOT/"m2_ml/M3_comparison_audit.md").read_text()))
    ''')
    md('''## 6. Actual successes and failures

Green dashed boxes are annotations; orange boxes are predictions at a display
threshold selected using validation F1. This threshold does not replace AP's
ranked score range. Examples follow stated criteria and are not proof of a
general causal mechanism.
''')
    code('''
    display(pd.read_csv(RUN/"example_selection.csv"))
    display(Image(filename=str(RUN/"revision_examples.jpg")))
    if (RUN/"visual_review.md").exists():display(Markdown((RUN/"visual_review.md").read_text()))
    ''')
    md('''## 7. Reproduce and read the report

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m m2_ml.run_dense \\
  --out results/m2/revision_20261117/NEW_RUN --workers 6 --kernel
```

`--evaluate-previous-test` explicitly enables a labelled follow-up test after
validation selection. Use a new run name for a new experiment. Original outputs
are in `results/m2/full/20261114_full/`; revision outputs are separate. The report
is `m2_ml/M2_revision_report.md`; its conclusions use these saved measurements.
The group's final document still needs IEEE formatting and the other methods.
''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"name":"python3","display_name":"Python 3","language":"python"},"language_info":{"name":"python"}})
    nbf.validate(nb);nbf.write(nb,REPO_ROOT/"m2_ml/M2_revision.ipynb")
    print("Revised report/notebook generated from",out)


if __name__=="__main__":main()
