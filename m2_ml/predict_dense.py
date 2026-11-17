"""Predict with a saved revised M2 model (no training or model selection).

python -m m2_ml.predict_dense --model <R4-BoxRefine.joblib> --split val --out preds/revised_val.json
Use --limit 3 for a short demo; subset output is not a full-split evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from m2_ml.detector import load_records,read_rgb
from m2_ml.dense_detector import DenseConfig,scan_dense,dense_predictions
from utils.data import get_data_root


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--split",choices=["val","test"],default="val")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--data-root",type=Path)
    p.add_argument("--limit",type=int)
    args=p.parse_args();threadpool_limits(1)
    records=load_records(args.data_root or get_data_root(),args.split)
    total=len(records)
    if args.limit is not None:
        if args.limit<1:p.error("--limit must be positive")
        records=records[:args.limit]
    model=joblib.load(args.model);cfg=DenseConfig(**model["config"])
    predictions=[];timings=[]
    for record in records:
        start=perf_counter();raw=scan_dense(read_rgb(record),{"model":model})
        predictions.extend(dense_predictions({record.path.stem:raw["model"]},cfg,model["nms_iou"]))
        timings.append(perf_counter()-start)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(predictions,allow_nan=False))
    metadata={"split":args.split,"images":len(records),"full_split":len(records)==total,
        "image_ids":[r.path.stem for r in records],"mean_seconds":float(np.mean(timings)),
        "per_image_seconds":timings,"model":str(args.model),"nms_iou":model["nms_iou"]}
    args.out.with_suffix(".metadata.json").write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))


if __name__=="__main__":main()
