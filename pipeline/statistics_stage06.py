"""N05 fixed-label app-composition dispersion and descriptive Stage06 contrasts.

B200/f0.80/PCG64 root20261023; no replacement, refit, reranking, bootstrap,
p-values, population confidence intervals or iid resampling of partition pairs.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import adjusted_mutual_info_score,adjusted_rand_score

from pipeline import embed
from pipeline.ami_cached import expected_cell_table,score_pair
from pipeline.labse_stage03 import load_contract,load_inputs

ROOT=Path(__file__).resolve().parents[1]
STAGE05=ROOT/'results/stage05/combined_20261025_001'
B=200
F=0.80
SEED=20261023


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def make_subsets(sizes,b=B):
    """One PCG64 stream: all TN1 draws, then all TN2 draws, sorted physical rows."""
    rng=np.random.Generator(np.random.PCG64(SEED))
    return {exp:np.stack([np.sort(rng.choice(sizes[exp],int(np.floor(F*sizes[exp])),replace=False)).astype(np.int32) for _ in range(b)]) for exp in ('tn1','tn2')}


def summaries(values):
    a=np.asarray(values,dtype=float)
    defined=np.isfinite(a)
    if not defined.any():
        return dict(n_defined=0,n_undefined=len(a),mean=np.nan,sd=np.nan,q025=np.nan,q50=np.nan,q975=np.nan,minimum=np.nan,maximum=np.nan)
    v=a[defined]
    return dict(n_defined=int(defined.sum()),n_undefined=int((~defined).sum()),mean=float(v.mean()),sd=float(v.std(ddof=1)) if len(v)>1 else np.nan,
        q025=float(np.quantile(v,.025)),q50=float(np.quantile(v,.5)),q975=float(np.quantile(v,.975)),minimum=float(v.min()),maximum=float(v.max()))


def load_sources():
    receipt=read(STAGE05/'execution_receipt.json')
    require(receipt['status']=='STAGE05_COMPLETE_VALIDATED','Stage05 not validated')
    for name,digest in receipt['artifacts_sha256'].items():
        require(embed.sha256(STAGE05/name)==digest,'Stage05 table changed: '+name)
    frames=load_inputs(load_contract())
    index=pd.read_csv(STAGE05/'artifact_index.csv')
    require(len(index)==34,'Expected34 clustering cells')
    labels=[]
    for row in index.itertuples():
        path=ROOT/row.labels_path
        require(embed.sha256(path)==row.labels_sha256,'Labels changed')
        vector=ROOT/row.embedding_path
        require(embed.sha256(vector)==row.embedding_sha256 and embed.sha256(vector.with_suffix('.rows.parquet'))==row.rows_sha256 and embed.sha256(vector.with_suffix('.json'))==row.metadata_sha256,'Frozen embedding/sidecar changed')
        z=np.load(path,allow_pickle=False)
        a=z['labels']
        require(a.shape==(10,len(frames[row.experiment])) and z['seeds'].tolist()==list(range(42,52)) and int(z['k'])==49 and int(z['n_init'])==10,'Frozen label protocol')
        require(a.min()>=0 and a.max()<49,'Cluster label range')
        labels.append(a)
    return frames,index,labels


def pair_groups(index):
    groups=[]
    paired=index[index.experiment=='tn2']
    for (rep,condition),rows in paired.groupby(['representation','condition'],sort=True):
        require(set(rows.language)=={'en','vi'},'Paired views')
        groups.append((rep,condition,int(rows[rows.language=='en'].index[0]),int(rows[rows.language=='vi'].index[0])))
    require(len(groups)==9,'Expected9 paired conditions')
    return groups


def compute_labels(out,frames,index,labels,subsets):
    taxonomy=np.full((B,34,10,2),np.nan)
    groups=pair_groups(index)
    pair_values=np.full((B,9,190,2),np.nan)
    benchmark={}
    checks=[]
    for exp in ('tn1','tn2'):
        rows=index[index.experiment==exp]
        ids=list(rows.index)
        genre=pd.Categorical(frames[exp].genreId.astype(str)).codes
        packed=np.vstack([genre,*[labels[i] for i in ids]]).astype(np.int32)
        slots={i:np.arange(1+j*10,1+(j+1)*10) for j,i in enumerate(ids)}
        n=len(subsets[exp][0])
        print('MARGINALS '+exp,flush=True)
        margins=np.empty((B,len(packed),49),dtype=np.int64)
        offsets=49*np.arange(len(packed),dtype=np.int64)[:,None]
        for draw,ii in enumerate(subsets[exp]):
            margins[draw]=np.bincount((packed[:,ii]+offsets).ravel(),minlength=len(packed)*49).reshape(len(packed),49)
        print('EXACT EMI CACHE '+exp+' max count '+str(margins.max()),flush=True)
        before=time.perf_counter()
        cache=expected_cell_table(n,margins.ravel())
        cache_seconds=time.perf_counter()-before
        with (out/(exp+'_emi_cell_cache.npy')).open('xb') as handle:
            np.save(handle,cache)
        start=time.perf_counter()
        for draw,ii in enumerate(subsets[exp]):
            sampled=packed[:,ii]
            def score(a,b):
                return score_pair(sampled[a],sampled[b],cache,margins[draw,a],margins[draw,b])
            for i in ids:
                for seed,slot in enumerate(slots[i]):
                    taxonomy[draw,i,seed]=score(0,slot)
            if exp=='tn2':
                for g,(_,_,en,vi) in enumerate(groups):
                    tasks=[(a,b) for a in slots[en] for b in slots[vi]]
                    tasks.extend((slots[i][a],slots[i][b]) for i in (en,vi) for a in range(10) for b in range(a+1,10))
                    pair_values[draw,g]=[score(a,b) for a,b in tasks]
            if draw==0:
                # Check diverse real cohort pairs before continuing the full computation.
                tasks=[(0,int(slots[i][s])) for i in (ids[0],ids[len(ids)//2],ids[-1]) for s in (0,9)]
                tasks.extend([(1,2),(1,len(packed)-1)])
                for a,b in tasks:
                    actual=score(a,b)
                    expected=(adjusted_mutual_info_score(sampled[a],sampled[b]),adjusted_rand_score(sampled[a],sampled[b]))
                    error=np.abs(np.asarray(actual)-expected)
                    require(error[0]<=1e-8 and error[1]<=1e-10,'Cached AMI/ARI real-data oracle mismatch')
                    checks.append(dict(experiment=exp,partition_a=a,partition_b=b,ami_error=float(error[0]),ari_error=float(error[1])))
            if draw==2:
                elapsed=time.perf_counter()-start
                benchmark[exp]=dict(first_three_draw_seconds=elapsed,projected_200_draw_seconds=elapsed*B/3,emi_cache_seconds=cache_seconds,
                    n_apps_per_draw=n,max_marginal_count=int(margins.max()))
                print('BENCHMARK '+exp+' '+json.dumps(benchmark[exp]),flush=True)
            if (draw+1)%10==0:
                print('DRAWS '+exp+' '+str(draw+1)+'/'+str(B),flush=True)
        with (out/(exp+'_label_dispersion.npz')).open('xb') as handle:
            np.savez_compressed(handle,taxonomy=taxonomy[:,ids],cell_indices=np.asarray(ids),
                pair_metrics=pair_values if exp=='tn2' else np.empty((B,0,190,2)),metric_names=np.array(['ami','ari']))
    embed.write_json_create_only(out/'numerical_oracle_checks.json',dict(status='EXACT_CACHED_AMI_ORACLE_PASS',checks=checks,tolerance_ami=1e-8,tolerance_ari=1e-10,benchmark=benchmark))
    return taxonomy,pair_values,groups,benchmark


def paired_folder(rep,condition):
    if rep=='labse':
        return ROOT/'results/stage05/labse_20261025_001'
    if rep=='multilingual_e5_large':
        return ROOT/'results/stage05/e5_tn2_20261025_001'/condition
    return ROOT/'results/stage05/remaining_eval_20261025_001'/(rep+'__'+condition)


def check_parent_file(folder,path):
    parent=folder
    while not (parent/'execution_receipt.json').exists():
        parent=parent.parent
        require(parent!=ROOT.parent,'No source execution receipt')
    receipt=read(parent/'execution_receipt.json')
    require(embed.sha256(path)==receipt['artifacts_sha256'][path.relative_to(parent).as_posix()],'Per-app artifact changed')


def fixed_app_metrics(out,frames,groups,subsets):
    retrieval=np.full((B,9,2,3),np.nan)
    correspondence=np.full((B,9,3),np.nan)
    for g,(rep,condition,_,_) in enumerate(groups):
        folder=paired_folder(rep,condition)
        path=folder/'retrieval_per_query.jsonl'
        check_parent_file(folder,path)
        queries=pd.read_json(path,lines=True)
        for d,direction in enumerate(('EN_TO_VI','VI_TO_EN')):
            rows=queries[queries.direction==direction]
            require(rows.query_app_id.tolist()==frames['tn2'].appId.tolist() and (rows.target_count==13657).all(),'Fixed full-catalogue query support')
            values=rows[['expected_recall_at_1','expected_recall_at_10','expected_reciprocal_rank']].to_numpy()
            retrieval[:,g,d]=np.asarray([values[ii].mean(axis=0) for ii in subsets['tn2']])
        path=folder/'paired_correspondence.parquet'
        check_parent_file(folder,path)
        pairs=pd.read_parquet(path)
        require(pairs.appId.tolist()==frames['tn2'].appId.tolist(),'Correspondence ID support')
        values=pairs[['matched_cosine','negative_cosine_median','delta_cosine']].to_numpy()
        correspondence[:,g]=np.asarray([values[ii].mean(axis=0) for ii in subsets['tn2']])
    table=pd.read_csv(STAGE05/'probe_all.csv')
    probes=np.full((B,len(table),2),np.nan)
    support=np.zeros((B,len(table)),dtype=np.int32)
    missing=np.zeros_like(support)
    for j,row in enumerate(table.itertuples()):
        exp=row.cell.split(':')[0]
        rep=row.representation
        folder=ROOT/'results/stage05'/('labse_probe_20261025_001' if rep=='labse' else 'e5_probe_20261025_001' if rep=='multilingual_e5_large' else 'remaining_probe_20261025_001')
        name=row.cell.replace(':','__') if rep in ('labse','multilingual_e5_large') else rep+'__'+row.cell.replace(':','__')
        path=folder/(name+'.test_predictions.parquet')
        check_parent_file(folder,path)
        pred=pd.read_parquet(path)
        ids={v:i for i,v in enumerate(frames[exp].appId)}
        positions=np.asarray([ids[v] for v in pred.appId],dtype=np.int64)
        classes=sorted(frames[exp].genreId.unique())
        y=pd.Categorical(pred.genreId,categories=classes).codes.astype(np.int64)
        p=pd.Categorical(pred.prediction,categories=classes).codes.astype(np.int64)
        require((y>=0).all() and (p>=0).all(),'Probe fixed label domain')
        for draw,ii in enumerate(subsets[exp]):
            include=np.zeros(len(frames[exp]),dtype=bool)
            include[ii]=True
            keep=include[positions]
            count=int(keep.sum())
            support[draw,j]=count
            if not count:
                missing[draw,j]=49
                continue
            cm=np.bincount(y[keep]*49+p[keep],minlength=49*49).reshape(49,49)
            denom=cm.sum(axis=0)+cm.sum(axis=1)
            f1=np.divide(2*np.diag(cm),denom,out=np.zeros(49,dtype=float),where=denom>0)
            probes[draw,j]=[f1.mean(),np.trace(cm)/count]
            missing[draw,j]=int((cm.sum(axis=1)==0).sum())
    with (out/'fixed_per_app_dispersion.npz').open('xb') as handle:
        np.savez_compressed(handle,retrieval=retrieval,correspondence=correspondence,probe=probes,probe_test_support=support,probe_missing_true_classes=missing)
    return retrieval,correspondence,probes,support,missing,table


def rank_summary(values,points,metadata,group_columns,metric):
    result=[]
    for _,rows in metadata.groupby(group_columns,sort=True):
        ids=rows.index.to_numpy()
        a=values[:,ids]
        valid=np.isfinite(a).all(axis=1)
        ranks=np.asarray([rankdata(-v,method='average') for v in a[valid]]) if valid.any() else np.empty((0,len(ids)))
        top=(a[valid]==np.max(a[valid],axis=1)[:,None]).astype(float) if valid.any() else np.empty((0,len(ids)))
        if valid.any():
            top/=top.sum(axis=1)[:,None]
        point_ranks=rankdata(-points[ids],method='average')
        for j,i in enumerate(ids):
            result.append(dict(**metadata.loc[i].to_dict(),metric=metric,point_rank=float(point_ranks[j]),
                rank_mean=float(ranks[:,j].mean()) if valid.any() else np.nan,rank_min=float(ranks[:,j].min()) if valid.any() else np.nan,rank_max=float(ranks[:,j].max()) if valid.any() else np.nan,
                share_draws_top_tie_adjusted=float(top[:,j].mean()) if valid.any() else np.nan,n_defined=int(valid.sum()),n_undefined=int((~valid).sum())))
    return result


def summarize(out,frames,index,taxonomy,pairs,groups,retrieval,correspondence,probes,support,missing,probe_table):
    point=pd.read_csv(STAGE05/'taxonomy_all_by_seed.csv')
    means=taxonomy.mean(axis=2)
    rows=[]
    point_tax=[]
    metadata=[]
    for i,cell in index.iterrows():
        name=':'.join([cell.experiment,cell.language,cell.condition])
        original=point[(point.representation==cell.representation)&(point.cell==name)]
        point_tax.append(float(original.ami.mean()))
        metadata.append(dict(experiment=cell.experiment,representation=cell.representation,language=cell.language,condition=cell.condition))
        for m,metric in enumerate(('ami','ari')):
            rows.append(dict(**metadata[-1],metric=metric,point=float(original[metric].mean()),**summaries(means[:,i,m])))
    pd.DataFrame(rows).to_csv(out/'taxonomy_app_dispersion.csv',index=False)
    rank=pd.DataFrame(metadata)
    rank_summary_rows=rank_summary(means[:,:,0],np.asarray(point_tax),rank,['experiment','language','condition'],'ami')
    pd.DataFrame(rank_summary_rows).to_csv(out/'taxonomy_rank_stability.csv',index=False)
    point_pairs=pd.read_csv(STAGE05/'partition_all.csv')
    components=np.stack([pairs[:,:,:100,0].mean(axis=2),pairs[:,:,100:145,0].mean(axis=2),pairs[:,:,145:,0].mean(axis=2)],axis=2)
    gap=components[:,:,0]-(components[:,:,1]+components[:,:,2])/2
    component_rows=[]
    for g,(rep,condition,_,_) in enumerate(groups):
        original=point_pairs[(point_pairs.representation==rep)&(point_pairs.condition==condition)].iloc[0]
        for j,key in enumerate(('C','W_EN','W_VI','S')):
            values=components[:,g,j] if j<3 else gap[:,g]
            component_rows.append(dict(representation=rep,condition=condition,component=key,point=float(original[key]),**summaries(values)))
    pd.DataFrame(component_rows).to_csv(out/'partition_app_dispersion.csv',index=False)
    original_retrieval=pd.read_csv(STAGE05/'retrieval_all.csv')
    ret_rows=[]
    cor_rows=[]
    point_cor=[]
    ret_rank_values=[]
    ret_rank_points=[]
    ret_rank_meta=[]
    for g,(rep,condition,_,_) in enumerate(groups):
        point_cor=read(paired_folder(rep,condition)/'correspondence_summary.json')
        for m,key in enumerate(('matched_cosine','negative_cosine_median','delta_cosine')):
            cor_rows.append(dict(representation=rep,condition=condition,metric=key,point=point_cor[key]['mean'],**summaries(correspondence[:,g,m])))
        for d,direction in enumerate(('EN_TO_VI','VI_TO_EN')):
            original=original_retrieval[(original_retrieval.representation==rep)&(original_retrieval.condition==condition)&(original_retrieval.direction==direction)].iloc[0]
            for m,(key,field) in enumerate([('recall_at_1','recall_at_1_primary'),('recall_at_10','recall_at_10_supplementary'),('mrr','mrr_supplementary')]):
                ret_rows.append(dict(representation=rep,condition=condition,direction=direction,metric=key,candidate_catalogue_size=13657,point=float(original[field]),**summaries(retrieval[:,g,d,m])))
            ret_rank_values.append(retrieval[:,g,d,0])
            ret_rank_points.append(float(original['recall_at_1_primary']))
            ret_rank_meta.append(dict(representation=rep,condition=condition,direction=direction))
    pd.DataFrame(ret_rows).to_csv(out/'retrieval_fixed_catalogue_query_dispersion.csv',index=False)
    pd.DataFrame(cor_rows).to_csv(out/'correspondence_app_dispersion.csv',index=False)
    pd.DataFrame(rank_summary(np.asarray(ret_rank_values).T,np.asarray(ret_rank_points),pd.DataFrame(ret_rank_meta),['condition','direction'],'recall_at_1')).to_csv(out/'retrieval_rank_stability.csv',index=False)
    probe_rows=[]
    probe_meta=[]
    for j,row in enumerate(probe_table.itertuples()):
        exp,lang,condition=row.cell.split(':')
        probe_meta.append(dict(experiment=exp,representation=row.representation,language=lang,condition=condition))
        for m,metric in enumerate(('test_macro_f1','test_accuracy')):
            probe_rows.append(dict(**probe_meta[-1],metric=metric,point=float(getattr(row,metric)),test_support_min=int(support[:,j].min()),test_support_max=int(support[:,j].max()),
                draws_missing_true_classes=int((missing[:,j]>0).sum()),**summaries(probes[:,j,m])))
    pd.DataFrame(probe_rows).to_csv(out/'probe_fixed_model_test_app_dispersion.csv',index=False)
    pd.DataFrame(rank_summary(probes[:,:,0],probe_table.test_macro_f1.to_numpy(),pd.DataFrame(probe_meta),['experiment','language','condition'],'test_macro_f1')).to_csv(out/'probe_rank_stability.csv',index=False)
    contrast=[]
    seed_contrast=[]
    for (exp,rep,lang),members in index.groupby(['experiment','representation','language']):
        by={r.condition:i for i,r in members.iterrows()}
        if exp=='tn1':
            left,right=by['char_budget'],by['native']
            for m,metric in enumerate(('ami','ari')):
                a=point[(point.representation==rep)&(point.cell=='tn1:vi:char_budget')].set_index('seed')[metric]
                b=point[(point.representation==rep)&(point.cell=='tn1:vi:native')].set_index('seed')[metric]
                delta=a-b
                seed_contrast.extend(dict(experiment=exp,representation=rep,contrast='char950_minus_native',metric=metric,seed=int(seed),value=float(value)) for seed,value in delta.items())
                contrast.append(dict(experiment=exp,representation=rep,contrast='char950_minus_native',metric=metric,point=float(delta.mean()),**summaries(means[:,left,m]-means[:,right,m])))
    for rep,condition,left,right in groups:
        for m,metric in enumerate(('ami','ari')):
            a=point[(point.representation==rep)&(point.cell=='tn2:en:'+condition)].set_index('seed')[metric]
            b=point[(point.representation==rep)&(point.cell=='tn2:vi:'+condition)].set_index('seed')[metric]
            delta=a-b
            contrast.append(dict(experiment='tn2',representation=rep,condition=condition,contrast='EN_minus_VI_taxonomy',metric=metric,point=float(delta.mean()),**summaries(means[:,left,m]-means[:,right,m])))
            seed_contrast.extend(dict(experiment='tn2',representation=rep,condition=condition,contrast='EN_minus_VI_taxonomy',metric=metric,seed=int(seed),value=float(value)) for seed,value in delta.items())
    pd.DataFrame(contrast).to_csv(out/'paired_descriptive_contrasts.csv',index=False)
    pd.DataFrame(seed_contrast).to_csv(out/'matched_seed_contrasts.csv',index=False)
    strata=pd.read_csv(STAGE05/'provenance_all_by_seed.csv')
    profile=frames['tn1'].assign(provenance=frames['tn1'].text_source.map({'machine':'translation-flagged','human':'not-flagged'}).fillna('unknown'))
    profile.groupby(['provenance','genreId']).size().rename('n_apps').to_csv(out/'provenance_genre_support.csv')
    strata.groupby(['representation','cell','provenance'])[['ami','ari']].agg(['mean','std','min','max']).to_csv(out/'provenance_seed_summary.csv')
    return dict(n_label_scores=B*(34*10+9*190),n_taxonomy_cells=34,n_paired_conditions=9,n_probe_cells=26,
        undefined_taxonomy=int((~np.isfinite(taxonomy)).sum()),undefined_partition=int((~np.isfinite(pairs)).sum()),undefined_probe=int((~np.isfinite(probes)).sum()),undefined_retrieval=int((~np.isfinite(retrieval)).sum()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    args=parser.parse_args()
    require(args.run_id and all(c.isalnum() or c in '_-' for c in args.run_id),'Invalid run ID')
    out=ROOT/'results/stage06'/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    receipt=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        code_sha256={p:embed.sha256(ROOT/p) for p in ['pipeline/statistics_stage06.py','pipeline/ami_cached.py']},packages=embed.package_versions(),
        protocol=dict(B=B,fraction=F,sample_size_rule='floor(0.8*N)',sampling_unit='app ID',replace=False,rng='PCG64',root_seed=SEED,stream_order='TN1 all200, then TN2 all200',fixed_embeddings=True,fixed_partitions=True,fixed_probe_models=True),
        stage05_receipt_sha256=embed.sha256(STAGE05/'execution_receipt.json'),interpretation='Descriptive finite-catalogue app-composition dispersion only; no population CI/p-values/confirmatory tests/causal claims. Retrieval samples queries against the fixed full13657 candidate catalogue; probe uses sampled heldout-app intersections conditional on fitted models. Provenance uses existing fixed-partition seed summaries, not additional stratified sampling.')
    try:
        frames,index,labels=load_sources()
        subsets=make_subsets({exp:len(frame) for exp,frame in frames.items()})
        with (out/'sampled_app_indices.npz').open('xb') as handle:
            np.savez_compressed(handle,**subsets)
        taxonomy,pairs,groups,benchmark=compute_labels(out,frames,index,labels,subsets)
        retrieval,correspondence,probes,support,missing,probe_table=fixed_app_metrics(out,frames,groups,subsets)
        receipt.update(summarize(out,frames,index,taxonomy,pairs,groups,retrieval,correspondence,probes,support,missing,probe_table),benchmark=benchmark,status='STAGE06_DESCRIPTIVE_ANALYSIS_COMPLETE')
        receipt['pair_groups']=[dict(representation=r,condition=c,en_cell_index=e,vi_cell_index=v) for r,c,e,v in groups]
    except Exception as exc:
        receipt.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        receipt['elapsed_seconds']=time.perf_counter()-start
        receipt['artifacts_sha256']={p.name:embed.sha256(p) for p in out.iterdir() if p.is_file()}
        embed.write_json_create_only(out/'execution_receipt.json',receipt)
    print(json.dumps(dict(status=receipt['status'],elapsed_seconds=receipt['elapsed_seconds'],receipt=str(out/'execution_receipt.json'))),flush=True)


if __name__=='__main__':
    main()
