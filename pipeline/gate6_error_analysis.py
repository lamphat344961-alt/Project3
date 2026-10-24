#!/usr/bin/env python3
"""PAAC-VN Gate 6 — pre-specified 6-pair TN1 error analysis (descriptive only).

Primary descriptive view: mapping-free co-assignment rates for each frozen
KMeans partition, with exact same-genre and random-pair baselines.
Auxiliary view: ACC-optimal Hungarian mapped confusion, using the same mapping
procedure as Gate 4 and reconciled against official ACC/macro-F1 first.

Grid: 8 TN1 representations × 2 regimes × 10 outer seeds × 6 fixed genre pairs.
No bootstrap, no p-values, no TOST, no new genre pairs, no reclustering.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

TN1_N=23883
K=49
SEEDS=np.arange(42,52,dtype=np.int32)
DATA_HASH="0cf5e72c00f1392e9f7e83ef2fc282e8deb9e81bc6e5d332ce5d761a41f7c52b"
MODEL_HASH="bdc8a5f2f72d699ff76dbedf5ceca015d5f2c2d2ff531c6289d900fd4b3e01d6"
REPS=("tfidf_lsa","phobert_base_v2","videberta_base","sup_simcse_phobert",
      "bkai_vietnamese_biencoder","multilingual_e5_large","bge_m3","qwen3_embedding_0_6b")
TRUNCS=("char_budget","native")
PAIRS=(
    ("SPORTS","GAME_SPORTS"),
    ("EDUCATION","GAME_EDUCATIONAL"),
    ("MUSIC_AND_AUDIO","GAME_MUSIC"),
    ("TOOLS","PRODUCTIVITY"),
    ("ENTERTAINMENT","VIDEO_PLAYERS"),
    ("BUSINESS","FINANCE"),
)


def sha256(path:Path,chunk:int=4<<20)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(chunk),b""):h.update(b)
    return h.hexdigest()


def comb2(n):
    n=np.asarray(n,dtype=np.int64)
    return n*(n-1)//2


def contingency(genre:np.ndarray,pred:np.ndarray):
    genres,gi=np.unique(genre.astype(str),return_inverse=True)
    clusters,ci=np.unique(pred,return_inverse=True)
    if len(genres)!=K or len(clusters)!=K:
        raise RuntimeError(f"Expected 49 genres/clusters; got {len(genres)}/{len(clusters)}")
    c=np.zeros((K,K),dtype=np.int64)
    np.add.at(c,(gi,ci),1)
    return genres,clusters,c,gi,ci


def hungarian_mapping_and_scores(genres,clusters,c):
    rows,cols=linear_sum_assignment(c,maximize=True)
    mapping={clusters[col]:genres[row] for row,col in zip(rows,cols)}
    matched=c[rows,cols].astype(np.float64)
    rs=c.sum(axis=1)[rows].astype(np.float64)
    cs=c.sum(axis=0)[cols].astype(np.float64)
    acc=float(matched.sum()/c.sum())
    f1=float(np.mean(2*matched/(rs+cs)))
    return mapping,acc,f1


def exact_coassignment(c:np.ndarray,genres:np.ndarray,clusters:np.ndarray,a:str,b:str):
    ia=int(np.where(genres==a)[0][0]); ib=int(np.where(genres==b)[0][0])
    na=int(c[ia].sum()); nb=int(c[ib].sum())
    if na<2 or nb<2:
        raise RuntimeError(f"Insufficient support for {a}/{b}: {na}/{nb}")
    # P(same predicted cluster | one app from A, one from B), all A×B pairs.
    cross_num=int(np.sum(c[ia].astype(np.int64)*c[ib].astype(np.int64)))
    cross_den=na*nb
    p_cross=cross_num/cross_den
    # Exact within-genre co-assignment for each genre, then pair-count weighted pool.
    same_a_num=int(np.sum(comb2(c[ia]))); same_a_den=int(comb2(na))
    same_b_num=int(np.sum(comb2(c[ib]))); same_b_den=int(comb2(nb))
    p_same_a=same_a_num/same_a_den; p_same_b=same_b_num/same_b_den
    p_same_pool=(same_a_num+same_b_num)/(same_a_den+same_b_den)
    # Exact random-pair baseline over all unordered distinct app pairs in this partition.
    sizes=c.sum(axis=0).astype(np.int64)
    random_num=int(np.sum(comb2(sizes))); random_den=int(comb2(int(c.sum())))
    p_random=random_num/random_den
    return {
        "support_a":na,"support_b":nb,
        "cross_coassignment_count":cross_num,"cross_pair_count":cross_den,
        "cross_coassignment_rate":float(p_cross),
        "same_a_coassignment_rate":float(p_same_a),
        "same_b_coassignment_rate":float(p_same_b),
        "same_genre_pooled_baseline":float(p_same_pool),
        "random_pair_baseline":float(p_random),
        "cross_minus_same_genre":float(p_cross-p_same_pool),
        "cross_minus_random":float(p_cross-p_random),
    }


def mapped_confusion(genre:np.ndarray,pred:np.ndarray,mapping:dict,a:str,b:str):
    mapped=np.array([mapping[x] for x in pred],dtype=object)
    ma=genre.astype(str)==a; mb=genre.astype(str)==b
    ca=int(np.sum(ma & (mapped==b))); cb=int(np.sum(mb & (mapped==a)))
    na=int(ma.sum()); nb=int(mb.sum())
    return {"support_a":na,"support_b":nb,
            "a_to_b_count":ca,"a_to_b_rate":float(ca/na),
            "b_to_a_count":cb,"b_to_a_rate":float(cb/nb)}


def official_metric(metrics,rep,tr,seed,metric):
    q=metrics[(metrics.experiment=="tn1")&(metrics.section=="TN1_EXTERNAL")&
              (metrics.representation==rep)&(metrics.truncation==tr)&
              (metrics.language=="vi")&(metrics.algorithm=="kmeans_main")&
              (metrics.run=="main")&(metrics.seed==seed)&(metrics.metric==metric)]
    if len(q)!=1:
        raise RuntimeError(f"Missing/duplicate Gate4 metric {rep}/{tr}/{seed}/{metric}: {len(q)}")
    return float(q.value.iloc[0])


def summarize(df:pd.DataFrame,group_cols:list[str],metric_cols:list[str])->pd.DataFrame:
    rows=[]
    for key,g in df.groupby(group_cols,sort=False):
        if not isinstance(key,tuple):key=(key,)
        row=dict(zip(group_cols,key)); row["n_seeds"]=len(g)
        for m in metric_cols:
            v=g[m].to_numpy(dtype=float)
            row[m+"__mean"]=float(np.mean(v)); row[m+"__std"]=float(np.std(v,ddof=1))
            row[m+"__min"]=float(np.min(v)); row[m+"__max"]=float(np.max(v))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo",type=Path,default=Path("."))
    ap.add_argument("--out-dir",type=Path,default=Path("results/gate6/error_analysis"))
    args=ap.parse_args(); repo=args.repo.resolve()
    out=args.out_dir if args.out_dir.is_absolute() else repo/args.out_dir
    if out.exists():raise FileExistsError(f"Refusing overwrite: {out}")
    if sha256(repo/"data_v1/benchmark_v1.parquet")!=DATA_HASH:raise RuntimeError("Frozen TN1 hash mismatch")
    if sha256(repo/"pipeline/models.yaml")!=MODEL_HASH:raise RuntimeError("Frozen models.yaml hash mismatch")
    emeta=json.loads((repo/"results/evaluation/evaluation_meta.json").read_text(encoding="utf-8"))
    metrics_p=repo/"results/evaluation/metrics_long.csv"
    if sha256(metrics_p)!=emeta["metrics_long_sha256"]:raise RuntimeError("metrics_long hash mismatch")
    metrics=pd.read_csv(metrics_p)
    ds=pd.read_parquet(repo/"data_v1/benchmark_v1.parquet",columns=["appId","genreId"])
    genre=ds.genreId.astype(str).to_numpy()
    present=set(genre)
    missing=sorted({x for p in PAIRS for x in p}-present)
    if missing:raise RuntimeError(f"Pre-specified genres absent: {missing}")

    co_rows=[]; map_rows=[]; reconciliation=[]; label_hashes={}
    for rep in REPS:
        for tr in TRUNCS:
            lp=repo/"results/labels"/f"tn1__{rep}__vi__{tr}__kmeans_main.npz"
            label_hashes[lp.name]=sha256(lp)
            with np.load(lp,allow_pickle=False) as z:
                seeds=np.asarray(z["seeds"],dtype=np.int32); labels=np.asarray(z["labels"],dtype=np.int32)
            if not np.array_equal(seeds,SEEDS) or labels.shape!=(10,TN1_N):
                raise RuntimeError(f"{lp.name}: seeds/labels shape mismatch")
            for i,seed in enumerate(seeds):
                pred=labels[i]
                genres,clusters,c,_,_=contingency(genre,pred)
                mapping,acc,f1=hungarian_mapping_and_scores(genres,clusters,c)
                off_acc=official_metric(metrics,rep,tr,int(seed),"acc_hungarian")
                off_f1=official_metric(metrics,rep,tr,int(seed),"macro_f1_hungarian")
                err_acc=abs(acc-off_acc); err_f1=abs(f1-off_f1)
                if err_acc>1e-12 or err_f1>1e-12:
                    raise RuntimeError(f"Hungarian reconciliation failed {rep}/{tr}/{seed}: "
                                       f"ACC {acc} vs {off_acc}; F1 {f1} vs {off_f1}")
                reconciliation.append({"representation":rep,"regime":tr,"seed":int(seed),
                                       "acc_recomputed":acc,"acc_official":off_acc,"acc_abs_error":err_acc,
                                       "macro_f1_recomputed":f1,"macro_f1_official":off_f1,"macro_f1_abs_error":err_f1})
                for a,b in PAIRS:
                    co=exact_coassignment(c,genres,clusters,a,b)
                    co_rows.append({"representation":rep,"regime":tr,"seed":int(seed),
                                    "genre_a":a,"genre_b":b,**co})
                    mc=mapped_confusion(genre,pred,mapping,a,b)
                    map_rows.append({"representation":rep,"regime":tr,"seed":int(seed),
                                     "genre_a":a,"genre_b":b,**mc,
                                     "acc_reconciled":True,"macro_f1_reconciled":True})
            print(f"[C3] {rep}/{tr} complete",flush=True)

    co=pd.DataFrame(co_rows); mp=pd.DataFrame(map_rows); rec=pd.DataFrame(reconciliation)
    if len(co)!=960 or len(mp)!=960 or len(rec)!=160:
        raise RuntimeError(f"Unexpected row counts co={len(co)} mapped={len(mp)} rec={len(rec)}")
    co_summary=summarize(co,["representation","regime","genre_a","genre_b"],
                         ["cross_coassignment_rate","same_a_coassignment_rate","same_b_coassignment_rate",
                          "same_genre_pooled_baseline","random_pair_baseline","cross_minus_same_genre","cross_minus_random"])
    mp_summary=summarize(mp,["representation","regime","genre_a","genre_b"],["a_to_b_rate","b_to_a_rate"])
    if len(co_summary)!=96 or len(mp_summary)!=96:raise RuntimeError("Expected 96 summary rows")

    out.mkdir(parents=True,exist_ok=False)
    paths={
        "coassignment_raw":out/"gate6_coassignment_6pairs.csv",
        "coassignment_summary":out/"gate6_coassignment_6pairs_summary.csv",
        "mapped_raw":out/"gate6_mapped_confusion_6pairs.csv",
        "mapped_summary":out/"gate6_mapped_confusion_6pairs_summary.csv",
        "reconciliation":out/"gate6_hungarian_reconciliation.csv",
    }
    co.to_csv(paths["coassignment_raw"],index=False); co_summary.to_csv(paths["coassignment_summary"],index=False)
    mp.to_csv(paths["mapped_raw"],index=False); mp_summary.to_csv(paths["mapped_summary"],index=False)
    rec.to_csv(paths["reconciliation"],index=False)
    manifest={
        "project":"PAAC-VN","phase":"C3_GATE6_ERROR_ANALYSIS","created_at_utc":datetime.now(timezone.utc).isoformat(),
        "design":"descriptive_only","primary_view":"mapping-free exact co-assignment",
        "auxiliary_view":"ACC-optimal Hungarian mapped confusion",
        "genre_pairs":[list(x) for x in PAIRS],"representations":list(REPS),"regimes":list(TRUNCS),
        "seeds":SEEDS.tolist(),"expected_raw_rows":960,"expected_summary_rows":96,
        "coassignment_definitions":{
            "cross":"sum_c n_Ac*n_Bc / (n_A*n_B)",
            "same_a":"sum_c C(n_Ac,2) / C(n_A,2)",
            "same_b":"sum_c C(n_Bc,2) / C(n_B,2)",
            "same_genre_pooled":"pair-count-weighted pooling of within-A and within-B pairs",
            "random":"sum_c C(n_c,2) / C(n,2), exact over all unordered distinct app pairs",
        },
        "inference":"NONE: no bootstrap, p-value, Holm, TOST or new genre-pair selection",
        "source_hashes":{
            "tn1_dataset":sha256(repo/"data_v1/benchmark_v1.parquet"),
            "models_yaml":sha256(repo/"pipeline/models.yaml"),
            "metrics_long":sha256(metrics_p),
            "evaluation_meta":sha256(repo/"results/evaluation/evaluation_meta.json"),
            "script":sha256(Path(__file__).resolve()),
        },
        "kmeans_label_sha256":label_hashes,
        "outputs":{k:{"path":v.relative_to(repo).as_posix(),"sha256":sha256(v),"rows":int(len(pd.read_csv(v)))} for k,v in paths.items()},
        "canonical_inputs_modified":False,
        "gate6_cannot_change_main_conclusions":True,
    }
    m=out/"gate6_error_analysis_manifest.json"
    with m.open("x",encoding="utf-8") as f:json.dump(manifest,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(f"C3 COMPLETE: {out}")
    print(f"manifest_sha256={sha256(m)}")

if __name__=="__main__":main()
