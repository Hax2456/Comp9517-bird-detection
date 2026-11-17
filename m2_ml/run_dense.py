"""Run the revised M2 experiment without changing the original experiment.

Selection uses validation only. Optional test inference is explicitly a follow-up
on the test set already evaluated by the original M2 run, not a fresh holdout.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from m2_ml.detector import load_records,read_rgb,subset,evaluate_predictions,environment_info
from m2_ml.dense_detector import DenseConfig,fit_templates,sample_image,fit_dense,scan_dense,mine_image,dense_predictions,fit_kernel_reranker
from m2_ml.run_full import write_json,log
from utils.data import get_data_root

_MODE=None
_CFG=None
_TEMPLATES=None
_MODELS=None
_CACHE=None
_LIMIT=None


def init(mode,cfg,templates,models,cache):
    global _MODE,_CFG,_TEMPLATES,_MODELS,_CACHE,_LIMIT
    _MODE,_CFG,_TEMPLATES,_MODELS,_CACHE=mode,cfg,templates,models,Path(cache)
    _LIMIT=threadpool_limits(1)


def task(record):
    path=_CACHE/f"{record.path.stem}.joblib"
    if path.exists():return joblib.load(path)
    start=perf_counter();image=read_rgb(record)
    if _MODE=="sample":
        seed=(_CFG.seed+int.from_bytes(hashlib.sha256(record.path.stem.encode()).digest()[:4],"little"))%(2**32)
        result=sample_image(image,record.boxes,_TEMPLATES,_CFG,seed,True)
    elif _MODE=="mine":
        result=mine_image(image,record.boxes,next(iter(_MODELS.values())),_CFG)
    else:
        result=scan_dense(image,_MODELS)
    output=(record.path.stem,result,perf_counter()-start)
    temp=path.with_suffix(".tmp");joblib.dump(output,temp);temp.replace(path)
    return output


def stage(records,mode,models,out,cfg,templates,workers,name):
    cache=out/"cache"/name;cache.mkdir(parents=True,exist_ok=True)
    start,last=perf_counter(),perf_counter();results=[]
    log(f"{name}: {len(records)} images, {workers} workers")
    with ProcessPoolExecutor(max_workers=workers,initializer=init,
        initargs=(mode,cfg,templates,models,str(cache))) as pool:
        for i,item in enumerate(pool.map(task,records,chunksize=1),1):
            results.append(item)
            if i==1 or i==len(records) or perf_counter()-last>=20:
                log(f"{name}: {i}/{len(records)}, {perf_counter()-start:.1f}s")
                write_json(out/"progress.json",{"stage":name,"done":i,"total":len(records)})
                last=perf_counter()
    return results,perf_counter()-start


def unpack(results,names):
    return {name:{image_id:raw[name] for image_id,raw,_ in results} for name in names}


def select(raw,model,name,records,out,cfg):
    rows=[]
    for threshold in (.3,.5):
        predictions=dense_predictions(raw,cfg,threshold)
        metrics=evaluate_predictions(predictions,records,out/"eval",f"{name}_C{model['C']}_nms{threshold}")
        rows.append({"variant":name,"C":model["C"],"nms_iou":threshold,**metrics})
    best=max(rows,key=lambda row:row["AP50"])
    model=dict(model,nms_iou=best["nms_iou"],validation_metrics={k:best[k] for k in best if k.startswith("AP")})
    return model,best,rows


def run(args):
    threadpool_limits(1);warnings.filterwarnings("error",category=ConvergenceWarning)
    cfg=DenseConfig(cell_size=args.cell_size,templates=args.templates)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/"COMPLETE.json").exists():log("Completed revision run; use another directory for a new experiment.");return
    train=subset(load_records(get_data_root(),"train"),args.train_limit,cfg.seed)
    val=subset(load_records(get_data_root(),"val"),args.val_limit,cfg.seed)
    templates=fit_templates(train,cfg)
    protocol={"config":asdict(cfg),"templates":templates,"workers":args.workers,
        "train_images":[r.path.name for r in train],"val_images":[r.path.name for r in val],
        "C_grid":[.01,.1,1.0],"nms_grid":[.3,.5],"regression_alpha":10.0,
        "kernel_reranking":args.kernel,
        "test_policy":"Original test was previously evaluated. Any new test score is a declared follow-up, not a fresh holdout.",
        "environment":environment_info()}
    if (out/"protocol.json").exists():
        old=json.loads((out/"protocol.json").read_text())
        for key in ["config","templates","train_images","val_images","kernel_reranking"]:
            if json.loads(json.dumps(protocol[key]))!=old[key]:raise ValueError("Run configuration changed; choose a new output directory")
    write_json(out/"protocol.json",protocol)
    path=out/"training.joblib"
    if path.exists():X,y,targets,audit,prep_seconds=joblib.load(path)
    else:
        results,prep_seconds=stage(train,"sample",{},out,cfg,templates,args.workers,"training_features")
        X=np.concatenate([data[0] for _,data,_ in results]);y=np.concatenate([data[1] for _,data,_ in results])
        targets=np.concatenate([data[3] for _,data,_ in results])
        audit=[{"image_id":image_id,**data[2]} for image_id,data,_ in results]
        del results
        joblib.dump((X,y,targets,audit,prep_seconds),path)
    pd.DataFrame(audit).to_csv(out/"training_audit.csv",index=False)
    log(f"Aligned training windows: {int(y.sum())} positive, {int((y==0).sum())} negative")
    initial_path=out/"initial_models.joblib"
    if initial_path.exists():initial=joblib.load(initial_path)
    else:
        initial={}
        for color in [False,True]:
            for C in [.01,.1,1.0]:
                name=f"{'color' if color else 'hog'}_C{C}"
                log(f"Fitting {name}")
                initial[name]=fit_dense(X,y,templates,cfg,C,color)
        joblib.dump(initial,initial_path)
    results,scan_seconds=stage(val,"scan",initial,out,cfg,templates,args.workers,"initial_validation")
    raws=unpack(results,initial);del results
    models,chosen,selected_raw,all_trials={},{},{},[]
    for variant,color in [("R1-HOG",False),("R2-HOG-HSV",True)]:
        choices=[]
        for name,b in initial.items():
            if b["color"]!=color:continue
            model,row,trials=select(raws[name],b,variant,val,out,cfg)
            choices.append((model,row,name));all_trials.extend(trials)
        model,row,name=max(choices,key=lambda v:v[1]["AP50"])
        models[variant],chosen[variant],selected_raw[variant]=model,row,raws[name]
        log(f"{variant}: {row}")
    del raws
    source=max(chosen,key=lambda name:chosen[name]["AP50"])
    mined_path=out/"mining.joblib"
    if mined_path.exists():hard,mining_audit,mining_seconds=joblib.load(mined_path)
    else:
        mined,mining_seconds=stage(train,"mine",{source:models[source]},out,cfg,templates,args.workers,"hard_negative_mining")
        pieces=[values[0] for _,values,_ in mined if len(values[0])]
        hard=np.concatenate(pieces) if pieces else np.empty((0,X.shape[1]),dtype=np.float32)
        mining_audit=[{"image_id":image_id,"bbox":box.tolist(),"score":float(score)}
            for image_id,values,_ in mined for box,score in zip(values[1],values[2])]
        if len(hard)>cfg.max_hard:
            ids=np.sort(np.random.default_rng(cfg.seed).choice(len(hard),cfg.max_hard,replace=False))
            hard=hard[ids];mining_audit=[mining_audit[i] for i in ids]
        joblib.dump((hard,mining_audit,mining_seconds),mined_path);del mined
    pd.DataFrame(mining_audit).to_csv(out/"hard_negatives.csv",index=False)
    log(f"Mined {len(hard)} hard negatives from training only; source {source}")
    path=out/"refined_models.joblib"
    if path.exists():refined=joblib.load(path)
    else:
        base=models[source]
        model=fit_dense(np.concatenate([X,hard]),np.concatenate([y,np.zeros(len(hard),dtype=np.int8)]),
                        templates,cfg,base["C"],base["color"])
        reg=Ridge(alpha=10.0).fit(X[y==1,:cfg.hog_dim],targets[y==1])
        box_model=dict(model,regression_weights=reg.coef_.astype(np.float32),regression_bias=reg.intercept_.astype(np.float32))
        refined={"R3-HardNeg":model,"R4-BoxRefine":box_model}
        if args.kernel:
            log("Fitting train-only PCA/Nystroem kernel-SVM reranker")
            reranker,kernel_metadata=fit_kernel_reranker(np.concatenate([X,hard]),
                np.concatenate([y,np.zeros(len(hard),dtype=np.int8)]),cfg)
            refined["R5-KernelSVM"]=dict(base,reranker=reranker,kernel_metadata=kernel_metadata)
        joblib.dump(refined,path)
    results,mined_scan_seconds=stage(val,"scan",refined,out,cfg,templates,args.workers,"refined_validation")
    raws=unpack(results,refined);del results
    for name,model in refined.items():
        model,row,trials=select(raws[name],model,name,val,out,cfg)
        models[name],chosen[name],selected_raw[name]=model,row,raws[name];all_trials.extend(trials)
        log(f"{name}: {row}")
    pd.DataFrame(all_trials).to_csv(out/"validation_search.csv",index=False)
    selected=max(chosen,key=lambda name:chosen[name]["AP50"])
    write_json(out/"FROZEN_REVISION.json",{"selected":selected,"chosen":chosen,
        "test_was_previously_seen":True,"selection":"validation AP50 only"})
    times=[]
    benchmark_records=subset(val,3,cfg.seed)
    for name,model in models.items():
        for r in benchmark_records:
            start=perf_counter();raw=scan_dense(read_rgb(r),{name:model})
            dense_predictions({r.path.stem:raw[name]},cfg,model["nms_iou"])
            times.append({"variant":name,"image_id":r.path.stem,"seconds":perf_counter()-start})
        joblib.dump(model,out/f"{name}.joblib")
    pd.DataFrame(times).to_csv(out/"serial_times.csv",index=False)
    val_rows=[]
    for name,model in models.items():
        predictions=dense_predictions(selected_raw[name],cfg,model["nms_iou"])
        write_json(out/f"{name}_val.json",predictions)
        val_rows.append({**chosen[name],"images":len(val),"mean_inference_seconds":
            np.mean([row["seconds"] for row in times if row["variant"]==name])})
    pd.DataFrame(val_rows).to_csv(out/"validation_summary.csv",index=False)
    manifest={"selected":selected,"source_for_mining":source,"training_windows":len(y),
        "positive_windows":int(y.sum()),"hard_negatives":len(hard),"preparation_wall_seconds":prep_seconds,
        "mining_wall_seconds":mining_seconds,"initial_scan_wall_seconds":scan_seconds,
        "refined_scan_wall_seconds":mined_scan_seconds,"validation":val_rows,
        "test_was_previously_seen":True,"test_re_evaluated":False}
    if args.evaluate_previous_test:
        if args.train_limit or args.val_limit:raise ValueError("Follow-up test requires the full training and validation splits")
        test=load_records(get_data_root(),"test")
        results,_=stage(test,"scan",models,out,cfg,templates,args.workers,"followup_test")
        raw=unpack(results,models);rows=[]
        for name,model in models.items():
            predictions=dense_predictions(raw[name],cfg,model["nms_iou"])
            write_json(out/f"{name}_followup_test.json",predictions)
            metrics=evaluate_predictions(predictions,test,out/"followup_test_eval",name)
            rows.append({"variant":name,"images":len(test),**metrics,"previously_seen_test":True})
        pd.DataFrame(rows).to_csv(out/"followup_test_summary.csv",index=False)
        manifest.update(test_re_evaluated=True,followup_test=rows)
    write_json(out/"manifest.json",manifest);write_json(out/"COMPLETE.json",{"selected":selected})
    write_json(out/"progress.json",{"stage":"complete"});log(f"COMPLETE {out}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--workers",type=int,default=4)
    p.add_argument("--train-limit",type=int);p.add_argument("--val-limit",type=int)
    p.add_argument("--cell-size",type=int,default=6);p.add_argument("--templates",type=int,default=12)
    p.add_argument("--evaluate-previous-test",action="store_true")
    p.add_argument("--kernel",action="store_true",help="include a classical nonlinear SVM reranking ablation")
    run(p.parse_args())


if __name__=="__main__":main()
