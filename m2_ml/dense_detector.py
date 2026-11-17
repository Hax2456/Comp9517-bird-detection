"""Revised traditional detector: aligned dense HOG + optional HSV + linear SVM.

Unlike the original crop classifier, training examples are sampled from the exact
feature maps/search windows used at inference. The outer window includes context;
the predicted box is its central region. No neural features or pretrained weights.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import euclidean_distances
from skimage.color import rgb2gray, rgb2hsv

from m2_ml.detector import Config, hog_descriptor, iou_one_to_many, nms


@dataclass
class DenseConfig:
    seed: int = 42
    cell_size: int = 6
    window_cells: int = 8
    context_cells: int = 1
    orientations: int = 9
    hsv_bins: int = 8
    templates: int = 12
    positives_per_bird: int = 4
    negatives_per_image: int = 40
    positive_iou: float = 0.5
    negative_iou: float = 0.2
    negative_coverage: float = 0.3
    hard_per_image: int = 32
    max_hard: int = 40000
    batch_size: int = 1024
    candidate_cap: int = 6000
    max_detections: int = 300
    score_floor: float = -1.5

    @property
    def window_pixels(self):
        return self.cell_size*self.window_cells

    @property
    def inner_pixels(self):
        return self.cell_size*(self.window_cells-2*self.context_cells)

    @property
    def hog_dim(self):
        return (self.window_cells-1)**2*4*self.orientations


def fit_templates(records, cfg):
    sizes=np.concatenate([r.boxes[:,2:] for r in records if len(r.boxes)])
    clusters=KMeans(n_clusters=min(cfg.templates,len(sizes)),n_init=10,random_state=cfg.seed)
    centers=clusters.fit(np.log(np.maximum(sizes,4))).cluster_centers_
    templates=np.unique(np.maximum(4,np.rint(np.exp(centers))).astype(int),axis=0)
    return sorted([tuple(map(int,v)) for v in templates],key=lambda v:v[0]*v[1])


def prepare_level(image, template, cfg, color=True):
    """Resize/pad once, compute HOG blocks and cell-integral HSV histograms."""
    height,width=image.shape[:2]
    tw,th=template
    if tw>width or th>height:
        return None
    lw=max(cfg.inner_pixels,int(round(width*cfg.inner_pixels/tw)))
    lh=max(cfg.inner_pixels,int(round(height*cfg.inner_pixels/th)))
    level=np.asarray(Image.fromarray(image).resize((lw,lh),Image.Resampling.BILINEAR))
    c=cfg.cell_size;p=cfg.context_cells*c
    extra_x=(-lw)%c;extra_y=(-lh)%c
    padded=np.pad(level,((p,p+extra_y),(p,p+extra_x),(0,0)),mode="reflect")
    gray=rgb2gray(padded)
    hcfg=Config(cell_size=c,block_cells=2,orientations=cfg.orientations)
    ny,nx=padded.shape[0]//c,padded.shape[1]//c
    blocks=hog_descriptor(gray,hcfg).reshape(ny-1,nx-1,2,2,cfg.orientations).astype(np.float32)
    b=cfg.window_cells-1
    # A view: no matrix of every flattened window is materialised here.
    windows=np.lib.stride_tricks.sliding_window_view(blocks,(b,b),axis=(0,1))
    windows=windows.transpose(0,1,5,6,2,3,4)
    integral=None
    if color:
        hsv=rgb2hsv(padded.astype(np.float32)/255.0)
        bins=np.minimum((hsv*cfg.hsv_bins).astype(np.int32),cfg.hsv_bins-1)
        ids=(np.arange(padded.shape[0])[:,None]//c*nx+
             np.arange(padded.shape[1])[None,:]//c).ravel()
        channels=[]
        for k in range(3):
            counts=np.bincount(ids*cfg.hsv_bins+bins[...,k].ravel(),
                               minlength=ny*nx*cfg.hsv_bins)
            channels.append(counts.reshape(ny,nx,cfg.hsv_bins))
        counts=np.concatenate(channels,axis=2).astype(np.float32)
        integral=np.pad(counts.cumsum(0).cumsum(1),((1,0),(1,0),(0,0)))
    return {"windows":windows,"integral":integral,"sx":lw/width,"sy":lh/height,
            "width":width,"height":height,"template":template,"padded":padded}


def window_boxes(level, ys, xs, cfg):
    xs=np.asarray(xs);ys=np.asarray(ys)
    x=xs*cfg.cell_size/level["sx"];y=ys*cfg.cell_size/level["sy"]
    w=np.minimum(cfg.inner_pixels/level["sx"],level["width"]-x)
    h=np.minimum(cfg.inner_pixels/level["sy"],level["height"]-y)
    return np.column_stack([x,y,w,h]).astype(np.float32)


def window_features(level,ys,xs,cfg,color=True):
    ys=np.asarray(ys,dtype=int);xs=np.asarray(xs,dtype=int)
    shape=level["windows"][ys,xs].reshape(len(ys),cfg.hog_dim)
    if not color:
        return np.ascontiguousarray(shape)
    integral=level["integral"];w=cfg.window_cells
    hist=(integral[ys+w,xs+w]-integral[ys,xs+w]-integral[ys+w,xs]+integral[ys,xs])
    hist/=cfg.window_pixels**2
    return np.concatenate([shape,hist],axis=1).astype(np.float32)


def overlap_matrix(boxes,gt):
    if not len(gt):return np.empty((len(boxes),0),dtype=np.float32)
    return np.column_stack([iou_one_to_many(g,boxes) for g in gt])


def background_mask(boxes,gt,cfg):
    if not len(gt):return np.ones(len(boxes),dtype=bool)
    boxes=np.asarray(boxes);gt=np.asarray(gt)
    lo=np.maximum(boxes[:,None,:2],gt[None,:,:2])
    hi=np.minimum(boxes[:,None,:2]+boxes[:,None,2:],gt[None,:,:2]+gt[None,:,2:])
    inter=np.maximum(hi-lo,0).prod(2)
    iou=inter/np.maximum(boxes[:,None,2]*boxes[:,None,3]+gt[None,:,2]*gt[None,:,3]-inter,1e-8)
    coverage=inter/np.maximum(gt[None,:,2]*gt[None,:,3],1e-8)
    return (iou.max(1)<cfg.negative_iou)&(coverage.max(1)<cfg.negative_coverage)


def box_targets(boxes,gt):
    boxes=np.asarray(boxes);gt=np.asarray(gt)
    center=boxes[:,:2]+boxes[:,2:]/2
    target_center=gt[:,:2]+gt[:,2:]/2
    return np.column_stack([(target_center-center)/boxes[:,2:],np.log(gt[:,2:]/boxes[:,2:])]).astype(np.float32)


def refine_boxes(boxes,delta,width,height):
    boxes=np.asarray(boxes);delta=np.asarray(delta)
    center=boxes[:,:2]+boxes[:,2:]/2+np.clip(delta[:,:2],-.5,.5)*boxes[:,2:]
    size=boxes[:,2:]*np.exp(np.clip(delta[:,2:],-.7,.7))
    low=np.clip(center-size/2,[0,0],[width-1,height-1])
    high=np.clip(center+size/2,low+1,[width,height])
    return np.column_stack([low,high-low]).astype(np.float32)


def sample_image(image,gt,templates,cfg,seed,return_targets=False):
    """GT-matched jittered positives and random negatives on actual HOG maps."""
    rng=np.random.default_rng(seed)
    positives=[[] for _ in gt];negatives=[]
    for index,template in enumerate(templates):
        level=prepare_level(image,template,cfg,color=True)
        if level is None:continue
        rows,cols=level["windows"].shape[:2]
        for j,g in enumerate(gt):
            # Only nearby dimensions can reach IoU .5; skip incompatible scales.
            gw,gh=g[2:];tw,th=template
            if min(gw,tw)*min(gh,th)/(gw*gh+tw*th-min(gw,tw)*min(gh,th))<.45:continue
            center_x=(g[0]+g[2]/2)*level["sx"]
            center_y=(g[1]+g[3]/2)*level["sy"]
            cx=int(round((center_x-cfg.inner_pixels/2)/cfg.cell_size))
            cy=int(round((center_y-cfg.inner_pixels/2)/cfg.cell_size))
            xs,ys=np.meshgrid(np.arange(cx-1,cx+2),np.arange(cy-1,cy+2))
            xs,ys=xs.ravel(),ys.ravel()
            valid=(xs>=0)&(xs<cols)&(ys>=0)&(ys<rows)
            xs,ys=xs[valid],ys[valid]
            if not len(xs):continue
            boxes=window_boxes(level,ys,xs,cfg)
            overlaps=iou_one_to_many(g,boxes)
            keep=np.flatnonzero(overlaps>=cfg.positive_iou)
            if len(keep):
                X=window_features(level,ys[keep],xs[keep],cfg)
                targets=box_targets(boxes[keep],np.repeat(g[None,:],len(keep),axis=0))
                for score,features,target in zip(overlaps[keep],X,targets):
                    positives[j].append((float(score),features,target))
        # Oversample per level, then choose a fixed budget across the image.
        count=max(8,int(np.ceil(cfg.negatives_per_image/len(templates)*4)))
        xs=rng.integers(cols,size=count);ys=rng.integers(rows,size=count)
        boxes=window_boxes(level,ys,xs,cfg)
        keep=np.flatnonzero(background_mask(boxes,gt,cfg))
        if len(keep):negatives.extend(window_features(level,ys[keep],xs[keep],cfg))
    features=[];labels=[];targets=[];covered=0
    for candidates in positives:
        if not candidates:continue
        covered+=1
        candidates.sort(key=lambda v:-v[0])
        # Keep the best alignment and sample other valid positions/scales.
        chosen=[0]
        if len(candidates)>1:
            chosen.extend((rng.choice(len(candidates)-1,min(cfg.positives_per_bird-1,len(candidates)-1),replace=False)+1).tolist())
        for k in chosen:
            features.append(candidates[k][1]);labels.append(1);targets.append(candidates[k][2])
    if negatives:
        indices=rng.choice(len(negatives),min(cfg.negatives_per_image,len(negatives)),replace=False)
        features.extend(negatives[k] for k in indices);labels.extend([0]*len(indices))
        targets.extend([np.full(4,np.nan,dtype=np.float32)]*len(indices))
    dimension=cfg.hog_dim+3*cfg.hsv_bins
    X=np.stack(features) if features else np.empty((0,dimension),dtype=np.float32)
    result=(X,np.asarray(labels,dtype=np.int8),{"birds":len(gt),"birds_with_positive_windows":covered,
        "positive_windows":int(np.sum(np.asarray(labels)==1)),"negative_windows":int(np.sum(np.asarray(labels)==0))})
    return (*result,np.asarray(targets,dtype=np.float32).reshape(-1,4)) if return_targets else result


def fit_dense(X,y,templates,cfg,C,color,balance=False):
    start=perf_counter()
    # HOG is already L2-Hys normalised, HSV histograms are L1 normalised.
    # Avoid amplifying rare gradient bins via per-dimension standardisation.
    X=X if color else X[:,:cfg.hog_dim]
    model=LinearSVC(C=C,dual="auto",class_weight="balanced" if balance else None,max_iter=20000,random_state=cfg.seed)
    model.fit(X,y)
    return {"svm":model,"weights":model.coef_[0].astype(np.float32),
        "bias":np.float32(model.intercept_[0]),"color":color,"C":C,
        "templates":templates,"config":asdict(cfg),"fit_seconds":perf_counter()-start,
        "samples":len(y)}


def fit_kernel_reranker(X,y,cfg):
    """Train-only PCA/Nystroem RBF map followed by a linear SVM in kernel space."""
    start=perf_counter()
    pca=PCA(n_components=min(96,X.shape[1],len(X)-1),svd_solver="randomized",random_state=cfg.seed)
    reduced=pca.fit_transform(X)
    ids=np.random.default_rng(cfg.seed).choice(len(X),min(1024,len(X)),replace=False)
    distances=euclidean_distances(reduced[ids],squared=True)
    gamma=1/(2*float(np.median(distances[distances>0])))
    feature_map=Nystroem(kernel="rbf",gamma=gamma,n_components=min(512,len(X)),random_state=cfg.seed)
    mapped=feature_map.fit_transform(reduced)
    svm=LinearSVC(C=1.0,dual="auto",class_weight="balanced",max_iter=20000,random_state=cfg.seed).fit(mapped,y)
    return make_pipeline(pca,feature_map,svm),{"gamma":gamma,"components":feature_map.n_components,
                                            "fit_seconds":perf_counter()-start}


def scan_dense(image,bundles,collect_features=False,collect_floor=None):
    """Shared dense maps, batched SVM scores; optionally return mined descriptors."""
    cfg=DenseConfig(**next(iter(bundles.values()))["config"])
    templates=next(iter(bundles.values()))["templates"]
    color=any(b["color"] or "reranker" in b for b in bundles.values()) or collect_features
    outputs={name:{"boxes":[],"scores":[],"features":[]} for name in bundles}
    def trim(result):
        if not result["scores"]:return
        scores=np.concatenate(result["scores"])
        boxes=np.concatenate(result["boxes"])
        order=np.argsort(-scores,kind="stable")[:cfg.candidate_cap]
        result["boxes"]=[boxes[order]];result["scores"]=[scores[order]]
        if result["features"]:
            result["features"]=[np.concatenate(result["features"])[order]]
    for template in templates:
        level=prepare_level(image,template,cfg,color)
        if level is None:continue
        rows,cols=level["windows"].shape[:2]
        for start in range(0,rows*cols,cfg.batch_size):
            ids=np.arange(start,min(rows*cols,start+cfg.batch_size))
            ys,xs=np.divmod(ids,cols)
            X=window_features(level,ys,xs,cfg,color)
            boxes=window_boxes(level,ys,xs,cfg)
            for name,bundle in bundles.items():
                values=X if bundle["color"] else X[:,:cfg.hog_dim]
                scores=values@bundle["weights"]+bundle["bias"]
                floor=(0.0 if collect_floor is None else collect_floor) if collect_features else cfg.score_floor
                keep=scores>=floor
                if not np.any(keep):continue
                selected_boxes=boxes[keep]
                if "regression_weights" in bundle:
                    delta=X[keep,:cfg.hog_dim]@bundle["regression_weights"].T+bundle["regression_bias"]
                    selected_boxes=refine_boxes(selected_boxes,delta,image.shape[1],image.shape[0])
                outputs[name]["boxes"].append(selected_boxes);outputs[name]["scores"].append(scores[keep])
                if collect_features or "reranker" in bundle:outputs[name]["features"].append(X[keep])
                if sum(len(v) for v in outputs[name]["scores"])>2*cfg.candidate_cap:
                    trim(outputs[name])
        for result in outputs.values():trim(result)
    for name,result in outputs.items():
        boxes=np.concatenate(result["boxes"]) if result["boxes"] else np.empty((0,4),dtype=np.float32)
        scores=np.concatenate(result["scores"]) if result["scores"] else np.empty(0,dtype=np.float32)
        order=np.argsort(-scores,kind="stable")[:cfg.candidate_cap]
        result["boxes"],result["scores"]=boxes[order],scores[order]
        if collect_features or "reranker" in bundles[name]:
            dim=cfg.hog_dim+3*cfg.hsv_bins
            X=np.concatenate(result["features"]) if result["features"] else np.empty((0,dim),dtype=np.float32)
            result["features"]=X[order]
            if "reranker" in bundles[name] and len(order):
                result["scores"]=bundles[name]["reranker"].decision_function(result["features"]).astype(np.float32)
                new_order=np.argsort(-result["scores"],kind="stable")
                result["boxes"],result["scores"]=result["boxes"][new_order],result["scores"][new_order]
                result["features"]=result["features"][new_order]
        if not collect_features:del result["features"]
    return outputs


def dense_predictions(raw,cfg,nms_iou=.3,floor=None):
    predictions=[]
    for image_id,result in raw.items():
        boxes,scores=result["boxes"],result["scores"]
        floor=cfg.score_floor if floor is None else floor
        valid=scores>=floor;boxes,scores=boxes[valid],scores[valid]
        keep=nms(boxes,scores,nms_iou,cfg.max_detections)
        predictions.extend({"image_id":image_id,"bbox":boxes[k].astype(float).tolist(),"score":float(scores[k])}
                           for k in keep)
    return predictions


def mine_image(image,gt,bundle,cfg):
    raw=scan_dense(image,{"model":bundle},collect_features=True)["model"]
    keep=np.flatnonzero(background_mask(raw["boxes"],gt,cfg))
    boxes,scores=raw["boxes"][keep],raw["scores"][keep]
    chosen=nms(boxes,scores,.3,cfg.hard_per_image)
    return raw["features"][keep[chosen]],boxes[chosen],scores[chosen]
