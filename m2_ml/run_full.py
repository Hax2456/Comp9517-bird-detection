"""Reproducible full M2 experiment with shared features and local worker processes.

python -m m2_ml.run_full --run-dir results/m2/full/<run> --workers 6 --test
Checkpoints and per-image raw candidates allow interrupted stages to resume.
Test selection is frozen in FROZEN_SELECTION.json before test predictions exist.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from m2_ml.detector import (
    Config, coco_ground_truth, environment_info, evaluate_predictions, fit_model,
    iou_one_to_many, learn_templates, load_records, make_training_features,
    mine_hard_negatives, predict_records, predictions_from_raw, proposal_recall,
    read_rgb, save_bundle, scan_image_many, subset,
)
from utils.data import REPO_ROOT, get_data_root

_BUNDLES = {}
_MODE = "scan"
_DISTORTION = "clean"
_CACHE = None
_THREAD_LIMITER = None


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(safe_json(value), indent=2, allow_nan=False))
    temp.replace(path)


def log(message):
    print(datetime.now(timezone.utc).strftime("%H:%M:%S UTC"), message, flush=True)


def _init_worker(bundles, mode, distortion, cache):
    global _BUNDLES, _MODE, _DISTORTION, _CACHE, _THREAD_LIMITER
    _BUNDLES, _MODE, _DISTORTION, _CACHE = bundles, mode, distortion, Path(cache)
    _THREAD_LIMITER = threadpool_limits(limits=1)


def _image_task(record):
    path = _CACHE / f"{record.path.stem}.joblib"
    if path.exists():
        return joblib.load(path)
    start = perf_counter()
    if _MODE == "mine":
        X, audit = mine_hard_negatives([record], next(iter(_BUNDLES.values())), progress=False)
        output = (record.path.stem, X, audit, perf_counter() - start)
    else:
        image = read_rgb(record)
        if _DISTORTION == "blur_sigma1":
            image = np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=1.0)))
        elif _DISTORTION == "brightness_0.6":
            image = np.asarray(ImageEnhance.Brightness(Image.fromarray(image)).enhance(0.6))
        elif _DISTORTION != "clean":
            raise ValueError(_DISTORTION)
        raw = scan_image_many(image, _BUNDLES, score_floor=-2.0)
        output = (record.path.stem, raw, perf_counter() - start)
    temp = path.with_suffix(".tmp")
    joblib.dump(output, temp)
    temp.replace(path)
    return output


def parallel_stage(records, bundles, out, name, workers, mode="scan", distortion="clean"):
    cache = out / "cache" / name
    cache.mkdir(parents=True, exist_ok=True)
    start, last = perf_counter(), perf_counter()
    results = []
    log(f"{name}: {len(records)} images, {workers} workers")
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(bundles, mode, distortion, str(cache))) as executor:
        for i, result in enumerate(executor.map(_image_task, records, chunksize=1), start=1):
            results.append(result)
            if i == 1 or i == len(records) or perf_counter() - last >= 20:
                log(f"{name}: {i}/{len(records)} images; elapsed {perf_counter()-start:.1f}s")
                write_json(out / "progress.json", {"stage": name, "images_done": i,
                    "images_total": len(records), "elapsed_seconds": perf_counter()-start})
                last = perf_counter()
    return results, perf_counter() - start


def collect_raw(results, names):
    return {name: {image_id: values[name] for image_id, values, _ in results} for name in names}


def tune(raw, bundle, variant, records, out):
    trials = []
    for threshold in (0.3, 0.5):
        for floor in (-2.0, -1.0):
            cfg = replace(Config(**bundle["config"]), nms_iou=threshold, score_floor=floor)
            predictions = predictions_from_raw(raw, cfg)
            metrics = evaluate_predictions(predictions, records, out / "eval",
                                            f"{variant}_{cfg.C}_{threshold}_{floor}")
            trials.append({"variant": variant, "C": cfg.C, "nms_iou": threshold,
                           "score_floor": floor, **metrics})
    winner = max(trials, key=lambda row: row["AP50"])
    result = dict(bundle)
    result["config"] = asdict(replace(Config(**bundle["config"]),
        nms_iou=winner["nms_iou"], score_floor=winner["score_floor"]))
    return result, winner, trials


def match_predictions(predictions, records, score_floor):
    """Greedy one-to-one IoU>=.5 matching for thresholded P/R/F1 and counts."""
    by_image = {r.path.stem: [] for r in records}
    for p in predictions:
        if p["score"] >= score_floor:
            by_image[p["image_id"]].append(p)
    rows = []
    for r in records:
        found = sorted(by_image[r.path.stem], key=lambda p: -p["score"])
        matched = np.zeros(len(r.boxes), dtype=bool)
        tp = 0
        for p in found:
            overlaps = iou_one_to_many(p["bbox"], r.boxes)
            overlaps[matched] = -1
            if len(overlaps) and overlaps.max() >= 0.5:
                matched[int(overlaps.argmax())] = True
                tp += 1
        fp, fn = len(found) - tp, len(r.boxes) - tp
        rows.append({"image_id": r.path.stem, "ground_truth": len(r.boxes),
                     "predicted": len(found), "TP": tp, "FP": fp, "FN": fn,
                     "absolute_count_error": abs(len(found)-len(r.boxes))})
    frame = pd.DataFrame(rows)
    tp, fp, fn = frame[["TP", "FP", "FN"]].sum()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return frame, {"operating_score_floor": score_floor, "TP": int(tp), "FP": int(fp),
        "FN": int(fn), "precision": precision, "recall": recall,
        "F1": 2*precision*recall/(precision+recall) if precision+recall else 0.0,
        "count_MAE": frame.absolute_count_error.mean()}


def density_metrics(predictions, records, out, tag):
    rows = []
    for name, lower, upper in [("0",0,0),("1-5",1,5),("6-20",6,20),
                                ("21-50",21,50),("51+",51,100000)]:
        group = [r for r in records if lower <= len(r.boxes) <= upper]
        ids = {r.path.stem for r in group}
        group_preds = [p for p in predictions if p["image_id"] in ids]
        if not group:
            continue
        # AP is undefined on a group with no ground-truth birds.
        metrics = evaluate_predictions(group_preds, group, out / "density_eval", f"{tag}_{name}")
        if lower == upper == 0:
            metrics = {key: float("nan") for key in metrics}
        rows.append({"density": name, "images": len(group),
                     "bird_boxes": sum(len(r.boxes) for r in group), **metrics})
    return rows


def benchmark(bundles, records, out):
    """Serial per-method latency; performed without other worker activity."""
    rows = []
    for name, bundle in bundles.items():
        log(f"Serial inference benchmark {name}: {len(records)} images")
        _, times = predict_records(records, bundle, progress=False)
        for record, seconds in zip(records, times):
            rows.append({"variant": name, "image_id": record.path.stem, "seconds": seconds})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "serial_inference_times.csv", index=False)
    return frame.groupby("variant").seconds.agg(["mean", "median", "count"]).reset_index()


def run(args):
    warnings.filterwarnings("error", category=ConvergenceWarning)
    threadpool_limits(limits=1)
    root, out = args.data_root.resolve(), args.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "TEST_COMPLETED.json").exists():
        log(f"Completed final run detected; preserving all saved results: {out}")
        log("Use a new run directory for a new experiment.")
        return
    cfg = Config()
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in (REPO_ROOT / "m2_ml").glob("*.py")}
    write_json(out / "protocol.json", {"dataset": str(root), "config": asdict(cfg),
        "workers": args.workers, "mining_image_budget": args.mining_images,
        "benchmark_images": args.benchmark_images, "robustness_images": args.robustness_images,
        "C_grid": [0.1,1.0], "NMS_grid": [0.3,0.5], "score_floors": [-2.0,-1.0],
        "robustness_distortions": ["clean", "blur_sigma1", "brightness_0.6"],
        "source_sha256": sources, "environment": environment_info(),
        "thread_limits": 1, "CPU": platform.processor(),
        "test_policy": "Freeze all settings and operating thresholds using validation only."})
    log("Auditing original training and validation splits")
    train, val = load_records(root, "train"), load_records(root, "val")
    if {r.path.stem for r in train} & {r.path.stem for r in val}:
        raise ValueError("Train/validation image-ID collision")
    audit = []
    for name, records in [("train",train),("val",val)]:
        audit.append({"split":name,"images":len(records),"bird_boxes":sum(len(r.boxes) for r in records),
                      "negative_images":sum(len(r.boxes)==0 for r in records)})
    manifest_file = root / "manifest.csv"
    grouping = None
    if manifest_file.exists():
        mf = pd.read_csv(manifest_file)
        grouping = int((mf.groupby("group_fingerprint").split.nunique() > 1).sum())
        if grouping:
            raise ValueError("Dataset source groups leak across splits")
    write_json(out / "data_audit.json", {"splits":audit,"cross_split_source_groups":grouping,
                                        "all_images_512_square":all(r.width==r.height==512 for r in train+val)})
    prepared = out / "training_features.joblib"
    if prepared.exists():
        templates, X, y, crop_audit, prep_seconds = joblib.load(prepared)
        log("Loaded training feature checkpoint")
    else:
        start = perf_counter()
        templates = learn_templates(train,cfg)
        X,y,crop_audit = make_training_features(train,templates,cfg,progress=False)
        prep_seconds = perf_counter()-start
        joblib.dump((templates,X,y,crop_audit,prep_seconds),prepared)
    pd.DataFrame(crop_audit).to_csv(out/"training_crop_counts.csv",index=False)
    log(f"Training crops: {(y==1).sum()} birds, {(y==0).sum()} background; prep {prep_seconds:.1f}s")
    models_path=out/"initial_models.joblib"
    if models_path.exists():
        initial=joblib.load(models_path)
    else:
        initial={}
        for feature,color in [("hog",False),("hog_color",True)]:
            for C in (0.1,1.0):
                log(f"Fitting {feature} C={C}")
                bundle=fit_model(X[feature],y,templates,replace(cfg,C=C),color)
                bundle["feature_name"]=feature
                initial[f"{feature}_C{C}"]=bundle
        joblib.dump(initial,models_path)
    diagnostic=proposal_recall(val,templates,cfg)
    write_json(out/"proposal_recall.json",diagnostic)
    log(f"Candidate recall on all validation boxes: {diagnostic['recall']:.4f}")
    # Mathematical-coordinate sanity check against the shared evaluator.
    perfect=[{"image_id":r.path.stem,"bbox":b.tolist(),"score":1.0} for r in val for b in r.boxes]
    assert np.isclose(evaluate_predictions(perfect,val,out/"eval","perfect_sanity")["AP50"],1)
    results,initial_scan_seconds=parallel_stage(val,initial,out,"initial_validation",args.workers)
    initial_raw=collect_raw(results,initial)
    del results
    bundles,chosen,all_trials={}, {}, []
    for variant,feature in [("M2-A","hog"),("M2-B","hog_color")]:
        choices=[]
        for name,bundle in initial.items():
            if bundle["feature_name"]!=feature:
                continue
            selected,row,trials=tune(initial_raw[name],bundle,variant,val,out)
            choices.append((selected,row,name));all_trials.extend(trials)
        bundle,row,key=max(choices,key=lambda item:item[1]["AP50"])
        bundle.update({"variant":variant,"shared_preparation_seconds":prep_seconds,
                       "training_pipeline_seconds":prep_seconds+bundle["fit_seconds"]})
        bundles[variant],chosen[variant]=(bundle,dict(row))
        log(f"Selected {variant}: {row}")
    raw_by_variant={name:initial_raw[f"{b['feature_name']}_C{b['config']['C']}"]
                    for name,b in bundles.items()}
    del initial_raw
    source=max(chosen,key=lambda name:chosen[name]["AP50"])
    mining_records=subset(train,args.mining_images,cfg.seed)
    mined,mining_seconds=parallel_stage(mining_records,{source:bundles[source]},out,
                                       "hard_negative_mining",args.workers,mode="mine")
    pieces=[values for _,values,_,_ in mined if len(values)]
    mining_audit=[row for _,_,audit_rows,_ in mined for row in audit_rows][:cfg.max_hard_negatives]
    hard_X=np.concatenate(pieces)[:cfg.max_hard_negatives] if pieces else None
    pd.DataFrame(mining_audit,columns=["image_id","bbox","score"]).to_csv(out/"hard_negatives.csv",index=False)
    log(f"Hard negatives: {len(mining_audit)} from {len(mining_records)} training images")
    matched_metrics=None
    if hard_X is not None and len(hard_X):
        mined_model_path=out/"mined_model.joblib"
        if mined_model_path.exists():
            fitted=joblib.load(mined_model_path)
        else:
            base=bundles[source]
            fitted=fit_model(np.vstack([X[base["feature_name"]],hard_X]),
                np.concatenate([y,np.zeros(len(hard_X),dtype=np.int8)]),templates,
                Config(**base["config"]),base["use_color"])
            fitted["feature_name"]=base["feature_name"]
            fitted.update({"variant":"M2-C","source_variant":source,
                "hard_negative_count":len(hard_X),"mining_seconds":mining_seconds,
                "shared_preparation_seconds":prep_seconds,
                "training_pipeline_seconds":prep_seconds+base["fit_seconds"]+mining_seconds+fitted["fit_seconds"]})
            save_bundle(fitted,mined_model_path)
        results,mined_scan_seconds=parallel_stage(val,{"M2-C":fitted},out,"mined_validation",args.workers)
        raw_mined=collect_raw(results,["M2-C"])["M2-C"]
        matched_metrics=evaluate_predictions(predictions_from_raw(raw_mined,Config(**bundles[source]["config"])),
                                             val,out/"eval","mining_matched_postprocessing")
        fitted,row,trials=tune(raw_mined,fitted,"M2-C",val,out)
        bundles["M2-C"],chosen["M2-C"],raw_by_variant["M2-C"]=fitted,row,raw_mined
        all_trials.extend(trials)
        log(f"Selected M2-C: {row}")
        del results
    else:
        mined_scan_seconds=0
    pd.DataFrame(all_trials).to_csv(out/"validation_search.csv",index=False)
    # Use independent serial timings on the same predeclared validation images.
    benchmark_records=subset(val,args.benchmark_images,cfg.seed)
    benchmark_path=out/"serial_benchmark_summary.csv"
    if benchmark_path.exists():
        timings=pd.read_csv(benchmark_path)
    else:
        timings=benchmark(bundles,benchmark_records,out)
        timings.to_csv(benchmark_path,index=False)
    validation_rows=[];operating={};val_predictions={};density_rows=[]
    for name,bundle in bundles.items():
        predictions=predictions_from_raw(raw_by_variant[name],Config(**bundle["config"]))
        val_predictions[name]=predictions
        write_json(out/f"{name}_val.json",predictions)
        metrics=evaluate_predictions(predictions,val,out/"eval",f"{name}_selected_val")
        threshold_trials=[]
        for threshold in [-1.0,0.0,0.5,1.0,1.5,2.0,2.5,3.0]:
            _,op=match_predictions(predictions,val,threshold);threshold_trials.append(op)
        op=max(threshold_trials,key=lambda v:v["F1"]);operating[name]=op
        per_image,_=match_predictions(predictions,val,op["operating_score_floor"])
        per_image.to_csv(out/f"{name}_val_per_image.csv",index=False)
        time_row=timings[timings.variant==name].iloc[0]
        validation_rows.append({"variant":name,"profile":"full","split":"val","images":len(val),
            **metrics,"C":bundle["config"]["C"],"nms_iou":bundle["config"]["nms_iou"],
            "score_floor":bundle["config"]["score_floor"],
            "training_pipeline_seconds":bundle["training_pipeline_seconds"],
            "svm_fit_seconds":bundle["fit_seconds"],"mean_inference_seconds":time_row["mean"],
            "median_inference_seconds":time_row["median"],"timed_images":int(time_row["count"]),**op})
        density_rows.extend({"variant":name,**row} for row in density_metrics(predictions,val,out,name))
        bundle["validation_metrics"]=metrics
        save_bundle(bundle,out/f"{name}.joblib")
    summary=pd.DataFrame(validation_rows);summary.to_csv(out/"validation_summary.csv",index=False)
    pd.DataFrame(density_rows).to_csv(out/"validation_density.csv",index=False)
    final=max(chosen,key=lambda name:chosen[name]["AP50"])
    # Freeze before robustness and final test; none of these results affect selection.
    frozen={"selected_variant":final,"configs":{n:b["config"] for n,b in bundles.items()},
        "operating_thresholds":operating,"selection_metric":"validation AP50",
        "frozen_utc":datetime.now(timezone.utc).isoformat()}
    frozen_path=out/"FROZEN_SELECTION.json"
    if frozen_path.exists():
        old=json.loads(frozen_path.read_text())
        if old["selected_variant"]!=final or old["configs"]!=safe_json(frozen["configs"]):
            raise RuntimeError("A previous selection was already frozen with different settings")
    else:
        write_json(frozen_path,frozen)
    save_bundle(bundles[final],out/"final_model.joblib")
    log(f"FROZEN final model: {final}; test has not influenced selection")
    robust_rows=[]
    robust_records=subset(val,args.robustness_images,cfg.seed) if args.robustness_images else []
    robust_ids={r.path.stem for r in robust_records}
    if args.robustness_images:
        for distortion in ["clean","blur_sigma1","brightness_0.6"]:
            if distortion=="clean":
                predictions_by_name={name:[p for p in vals if p["image_id"] in robust_ids]
                                     for name,vals in val_predictions.items()}
            else:
                results,_=parallel_stage(robust_records,bundles,out,f"robustness_{distortion}",
                                          args.workers,distortion=distortion)
                raw=collect_raw(results,bundles)
                predictions_by_name={name:predictions_from_raw(raw[name],Config(**b["config"]))
                                     for name,b in bundles.items()}
            for name,predictions in predictions_by_name.items():
                metrics=evaluate_predictions(predictions,robust_records,out/"robustness_eval",f"{name}_{distortion}")
                robust_rows.append({"variant":name,"distortion":distortion,"images":len(robust_records),**metrics})
        pd.DataFrame(robust_rows).to_csv(out/"robustness_summary.csv",index=False)
    manifest={"profile":"full","data_root":str(root),"protocol_file":"protocol.json",
        "train_images":[r.path.name for r in train],"validation_images":[r.path.name for r in val],
        "templates":templates,"training_crop_counts":{"positive":int((y==1).sum()),"negative":int((y==0).sum())},
        "shared_preparation_seconds":prep_seconds,"initial_validation_scan_wall_seconds":initial_scan_seconds,
        "mined_validation_scan_wall_seconds":mined_scan_seconds,"mining_seconds":mining_seconds,
        "mining_source":source,"mining_images":[r.path.name for r in mining_records],
        "hard_negative_count":len(mining_audit),"matched_mining_metrics":matched_metrics,
        "proposal_diagnostic":diagnostic,"selected_variant":final,"selected_config":bundles[final]["config"],
        "benchmark_images":[r.path.name for r in benchmark_records],"robustness_images":[r.path.name for r in robust_records],
        "validation_summary":validation_rows,"environment":environment_info(),"workers":args.workers,
        "test_evaluated":False,"source_sha256":sources}
    write_json(out/"manifest.json",manifest)
    if args.test:
        completed=out/"TEST_COMPLETED.json"
        if completed.exists():
            log("Final test already completed; reusing saved outputs without another evaluation")
        else:
            test=load_records(root,"test")
            development_ids={r.path.stem for r in train+val}
            if development_ids & {r.path.stem for r in test}:
                raise ValueError("Test/development image-ID collision")
            write_json(out/"FINAL_TEST_STARTED.json",{"frozen_selection_sha256":hashlib.sha256(frozen_path.read_bytes()).hexdigest()})
            # All three variants were frozen before this one final test batch.
            results,test_seconds=parallel_stage(test,bundles,out,"final_test",args.workers)
            raw_test=collect_raw(results,bundles);test_rows=[];test_density=[]
            for name,bundle in bundles.items():
                predictions=predictions_from_raw(raw_test[name],Config(**bundle["config"]))
                write_json(out/f"{name}_test.json",predictions)
                metrics=evaluate_predictions(predictions,test,out/"test_eval",f"{name}_final_test")
                per_image,op=match_predictions(predictions,test,operating[name]["operating_score_floor"])
                per_image.to_csv(out/f"{name}_test_per_image.csv",index=False)
                time_row=timings[timings.variant==name].iloc[0]
                test_rows.append({"variant":name,"split":"test","images":len(test),**metrics,
                    "training_pipeline_seconds":bundle["training_pipeline_seconds"],
                    "mean_inference_seconds":time_row["mean"],"timed_images":int(time_row["count"]),
                    "timing_split":"validation",**op})
                test_density.extend({"variant":name,**row} for row in density_metrics(predictions,test,out,f"{name}_test"))
            pd.DataFrame(test_rows).to_csv(out/"test_summary.csv",index=False)
            pd.DataFrame(test_density).to_csv(out/"test_density.csv",index=False)
            manifest.update({"test_evaluated":True,"test_images":[r.path.name for r in test],
                "test_summary":test_rows,"test_detection_wall_seconds":test_seconds})
            write_json(completed,{"test_images":len(test),"variants":list(bundles),
                "completed_utc":datetime.now(timezone.utc).isoformat()})
            write_json(out/"manifest.json",manifest)
            write_json(out/"m2_final_test.json",json.loads((out/f"{final}_test.json").read_text()))
    write_json(out/"progress.json",{"stage":"complete","run_directory":str(out)})
    log(f"COMPLETE: {out}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--mining-images",type=int,default=160)
    parser.add_argument("--benchmark-images",type=int,default=5)
    parser.add_argument("--robustness-images",type=int,default=20)
    parser.add_argument("--test",action="store_true")
    args=parser.parse_args()
    args.data_root=args.data_root or get_data_root()
    if min(args.workers,args.mining_images,args.benchmark_images)<1 or args.robustness_images<0:
        parser.error("workers, mining-images and benchmark-images must be positive")
    try:
        run(args)
    except Exception:
        args.run_dir.mkdir(parents=True,exist_ok=True)
        write_json(args.run_dir/"progress.json",{"stage":"failed","traceback":traceback.format_exc()})
        raise


if __name__=="__main__":
    main()
