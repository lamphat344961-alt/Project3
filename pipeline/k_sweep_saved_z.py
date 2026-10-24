#!/usr/bin/env python3
"""PAAC-VN Gate6 supplementary k-sweep: ONLY TN1 TF-IDF char_budget saved Z.
No new linkage, no new embedding, no clustering refit; no main-k selection.
"""
import argparse
import json
import os
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster

from evaluate_reviewed import sha256, cosine_unit, geometry_scores

STEM='tn1__tfidf_lsa__vi__char_budget'
KS=(30,40,49,60,80)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,default=Path('.'))
    a=p.parse_args();root=a.repo.resolve()
    meta=json.loads((root/'results/evaluation/evaluation_meta.json').read_text())
    scores=pd.read_csv(root/'results/evaluation/metrics_long.csv')
    if sha256(root/'results/evaluation/metrics_long.csv')!=meta['metrics_long_sha256']:
        raise ValueError('Gate4 metric table SHA mismatch')
    top=scores[(scores.section=='TN1_EXTERNAL')&(scores.algorithm=='agglomerative')&(scores.metric=='ami')]
    winner=top.sort_values('value',ascending=False).iloc[0]
    if (winner.representation,winner.truncation)!=('tfidf_lsa','char_budget'):
        raise ValueError('Frozen k-sweep selected config does not match Gate4 argmax')
    manifest=pd.read_csv(root/'results/audit/clustering_manifest.csv')
    row=manifest[(manifest.experiment=='tn1')&(manifest.representation=='tfidf_lsa')
                 &(manifest.truncation=='char_budget')&(manifest.language=='vi')]
    if len(row)!=1:raise ValueError('Missing frozen Gate3 manifest source')
    path=root/'results/labels'/f'{STEM}__linkage.npy'
    labels_path=root/'results/labels'/f'{STEM}__agglomerative.npy'
    if sha256(path)!=row.iloc[0].linkage_sha256 or sha256(labels_path)!=row.iloc[0].agglomerative_sha256:
        raise ValueError('Gate3 labels/linkage SHA mismatch')
    npy=root/'embeddings/tn1'/f'{STEM}.npy'
    if sha256(npy)!=meta['sources'][STEM]['sha256_npy_current']:
        raise ValueError('Embedding current SHA differs from Gate4')
    X=np.load(npy,mmap_mode='r');Z=np.load(path,mmap_mode='r')
    if X.shape[0]!=23883 or Z.shape!=(23882,4):raise ValueError('Unexpected frozen array shapes')
    unit=cosine_unit(X)
    result=[]
    for k in KS:
        labs=fcluster(np.array(Z,dtype=np.float64,order='C',copy=True),k,criterion='maxclust')
        count=np.unique(labs,return_counts=True)[1]
        vals=geometry_scores(unit,labs,2048)
        result.append({'requested_k':k,'observed_clusters':len(count),
                      'largest_cluster_fraction':int(count.max())/len(labs),
                      'singletons':int(np.count_nonzero(count==1)),
                      'smallest_cluster':int(count.min()),**vals})
        if k==49:
            old=np.load(labels_path,mmap_mode='r')
            if not np.array_equal(labs,old):
                raise ValueError('k49 partition differs from frozen Gate4 labels')
    out=root/'results/figures'
    out.mkdir(parents=True,exist_ok=True)
    csv=out/'k_sweep_tfidf_char_budget.csv'
    png=out/'k_sweep_tfidf_char_budget.png'
    man=out/'k_sweep_tfidf_char_budget_manifest.json'
    if any(p.exists() for p in [csv,png,man]):raise FileExistsError('Existing k-sweep output; refusing overwrite')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    cols=('silhouette_cosine','davies_bouldin','calinski_harabasz')
    for ax,col in zip(axes,cols):
        ax.plot([r['requested_k'] for r in result],[r[col] for r in result],marker='o')
        ax.set_xlabel('Requested k (saved linkage)');ax.set_title(col)
    fig.tight_layout()
    # stage before publish
    pd.DataFrame(result).to_csv(csv.with_suffix('.partial.csv'),index=False)
    fig.savefig(png.with_suffix('.partial.png'),dpi=160);plt.close(fig)
    os.replace(csv.with_suffix('.partial.csv'),csv)
    os.replace(png.with_suffix('.partial.png'),png)
    with man.open('x') as f:json.dump({'selected':STEM,'reason':'highest frozen TN1 Agglo AMI at k=49',
       'created_at_utc':datetime.now(timezone.utc).isoformat(),
       'evaluation_metrics_sha256':meta['metrics_long_sha256'],
       'linkage_sha256':sha256(path),'source_npy_sha256':sha256(npy),
       'reported_rows':result},f,indent=2)
    print('One supplementary k-sweep figure:',png)

if __name__=='__main__':main()
