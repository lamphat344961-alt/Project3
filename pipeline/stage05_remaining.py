"""Reuse verified historical recipes and complete Stage05 without GPU inference."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,adjusted_mutual_info_score,adjusted_rand_score
from sklearn.preprocessing import Normalizer
from threadpoolctl import threadpool_limits

from pipeline import embed,evaluate_reviewed as metrics
from pipeline.evaluate_stage05 import ROOT,evaluate_pair,provenance_labels
from pipeline.labse_stage03 import load_contract,load_inputs
from pipeline.probe_stage05 import fit_select

NEURAL=('phobert_base_v2','sup_simcse_phobert','bkai_vietnamese_biencoder','bge_m3','qwen3_embedding_0_6b')
REMAINING=('tfidf_lsa',*NEURAL)
PROPOSAL=ROOT/'results/stage05/probe_split_proposal_20261025_001/split_proposal.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


class Legacy:
    def __init__(self):
        manifest,self.manifest_path=metrics.validate_manifest(ROOT)
        self.ev=metrics.Evidence(ROOT,manifest)
        self.old=read(ROOT/'results/evaluation/evaluation_meta.json')
        require(embed.sha256(self.manifest_path)==self.old['clustering_manifest_sha256'],'Historical manifest changed')
        self.manifest=manifest[manifest.representation.isin(REMAINING)]
        require(len(self.manifest)==24,'Expected24 remaining cells')
        self.frames=load_inputs(load_contract())
        self.specs=embed.load_models(ROOT/'pipeline/models.yaml')['representations']
        self.lineage={}

    def get(self,exp,rep,lang,condition):
        require(rep in REMAINING,'Excluded model requested')
        obj=self.ev.get(exp,rep,lang,condition,load_X=True)
        key=obj['key']
        current=self.ev.embedding_provenance[key]
        prior=self.old['sources'][key]
        require(current['sha256_npy_current']==prior['sha256_npy_current'] and current['sha256_json']==prior['sha256_json'],'Historical embedding changed: '+key)
        spec=self.specs[rep]
        require(all(obj['meta'][field]==spec[field] for field in ('revision','hf_name','pooling','needs_word_segmentation','clustering_prefix')),'Recipe mismatch: '+key)
        self.lineage[key]=dict(**current,rows_sha256=embed.sha256(ROOT/'embeddings'/exp/(key+'.rows.parquet')),
            main_labels_sha256=embed.sha256(ROOT/'results/labels'/(key+'__kmeans_main.npz')),
            labels_reused=True,original_compute_dtype=obj['meta'].get('model_compute_dtype','not_recorded'))
        return obj


def lexical_features(corpora,spec,train_masks=None):
    """Fit shared lexical space on declared support, return frozen full transforms."""
    texts=[text for corpus in corpora for text in corpus]
    mask=None if train_masks is None else np.concatenate(train_masks)
    fit_texts=texts if mask is None else [t for t,m in zip(texts,mask) if m]
    vec=embed.make_tfidf_vectorizer(spec)
    sparse=vec.fit_transform(fit_texts)
    cfg=spec['svd']
    svd=TruncatedSVD(n_components=int(cfg['n_components']),random_state=int(cfg['random_state']))
    if mask is None:
        dense=svd.fit_transform(sparse).astype(np.float32)
    else:
        svd.fit(sparse)
        dense=svd.transform(vec.transform(texts)).astype(np.float32)
    dense=Normalizer(copy=False).fit_transform(dense).astype(np.float32,copy=False)
    result=[]
    pos=0
    for corpus in corpora:
        result.append(dense[pos:pos+len(corpus)])
        pos+=len(corpus)
    return result,vec,svd


def lexical_texts(frames,spec,exp,condition):
    langs=('vi',) if exp=='tn1' else ('en','vi')
    corpora=[]
    for lang in langs:
        field='description_clean' if lang=='vi' else 'description_original_clean'
        prepared,_=embed.prepare_tfidf_texts(frames[exp][field].tolist(),spec,condition,
            950 if condition=='char_budget' else None,512 if condition=='token_budget' else None)
        corpora.append(prepared)
    return langs,corpora


def lexical_audit(out,legacy):
    evidence=[]
    for exp,conditions in [('tn1',('char_budget','native')),('tn2',('native','token_budget'))]:
        for condition in conditions:
            print('LEXICAL RECONSTRUCTION '+exp+' '+condition,flush=True)
            langs,corpora=lexical_texts(legacy.frames,legacy.specs['tfidf_lsa'],exp,condition)
            arrays,vec,svd=lexical_features(corpora,legacy.specs['tfidf_lsa'])
            folder=out/(exp+'__'+condition)
            folder.mkdir()
            joblib.dump(dict(vectorizer=vec,svd=svd),folder/'reconstructed_fit.joblib',compress=3)
            for lang,reconstructed in zip(langs,arrays):
                obj=legacy.get(exp,'tfidf_lsa',lang,condition)
                maximum=float(np.max(np.abs(reconstructed-obj['X'])))
                evidence.append(dict(cell=obj['key'],max_abs_difference=maximum,threshold=1e-5,
                    passed=maximum<=1e-5,fit_scope='full cohort; TN2 concatenated EN+VI in one shared space',
                    reconstruction_model_sha256=embed.sha256(folder/'reconstructed_fit.joblib')))
                require(maximum<=1e-5,'LEXICAL_RECONSTRUCTION_MISMATCH:'+obj['key']+':'+str(maximum))
    embed.write_json_create_only(out/'lexical_lineage.json',dict(status='LEXICAL_RECONSTRUCTION_PASS',cells=evidence,
        interpretation='Reconstructed pinned algorithm/config/full-cohort support; matched historical matrices. Original serialized fit states were absent; reconstructed states are explicitly identified, not backdated. Transductive unsupervised use only.'))
    return dict(status='LEXICAL_RECONSTRUCTION_PASS',cells=evidence)


def evaluate(out,legacy,lexical_receipt):
    require(read(lexical_receipt)['status']=='LEXICAL_RECONSTRUCTION_PASS','Lexical lineage required')
    alignment=[]
    strata=[]
    for row in legacy.manifest.itertuples():
        obj=legacy.get(row.experiment,row.representation,row.language,row.truncation)
        print('REUSE LABELS '+obj['key'],flush=True)
        y=legacy.ev.datasets[row.experiment]['y']
        for seed,pred in zip(range(42,52),obj['main']):
            require(len(np.unique(pred))==49,'49 clusters required')
            alignment.append(dict(representation=row.representation,cell=':'.join([row.experiment,row.language,row.truncation]),seed=seed,n_apps=len(y),**metrics.external_scores(y,pred)))
            if row.experiment=='tn1':
                groups=provenance_labels(legacy.frames['tn1'])
                for group in ('translation-flagged','not-flagged','unknown'):
                    mask=(groups==group).to_numpy()
                    strata.append(dict(representation=row.representation,cell='tn1:vi:'+row.truncation,seed=seed,provenance=group,
                        n_apps=int(mask.sum()),ami=float(adjusted_mutual_info_score(y[mask],pred[mask])),ari=float(adjusted_rand_score(y[mask],pred[mask]))))
    pd.DataFrame(alignment).to_csv(out/'taxonomy_by_seed.csv',index=False)
    pd.DataFrame(strata).to_csv(out/'provenance_taxonomy_by_seed.csv',index=False)
    for rep in ('tfidf_lsa','bge_m3','qwen3_embedding_0_6b'):
        for condition in ('native','token_budget'):
            en=legacy.get('tn2',rep,'en',condition)
            vi=legacy.get('tn2',rep,'vi',condition)
            folder=out/(rep+'__'+condition)
            folder.mkdir()
            print('PAIRED '+rep+' '+condition,flush=True)
            evaluate_pair(folder,en['X'],vi['X'],en['main'],vi['main'],legacy.frames['tn2'])
    return dict(status='REMAINING_RECIPES_EVALUATED',n_cells=24,n_reused_kmeans_runs=240,n_new_kmeans_runs=0,
        n_taxonomy_rows=len(alignment),n_provenance_rows=len(strata),n_partition_pair_rows=1140,n_retrieval_queries=163884,
        lexical_audit_sha256=embed.sha256(lexical_receipt))


def export_probe(out,rep,cell,x,frame,folds,classes):
    aligned=frame[['appId']].merge(folds[['appId','split']],on='appId',how='left',validate='one_to_one')
    require(aligned.appId.tolist()==frame.appId.tolist() and not aligned.split.isna().any(),'Fold alignment')
    split=aligned.split.to_numpy()
    y=frame.genreId.to_numpy()
    print('PROBE '+rep+' '+cell,flush=True)
    model,pred,summary=fit_select(x,y,split,classes)
    base=out/(rep+'__'+cell.replace(':','__'))
    pd.DataFrame(dict(appId=frame.appId[split=='test'],genreId=y[split=='test'],prediction=pred)).to_parquet(base.with_suffix('.test_predictions.parquet'),index=False)
    precision,recall,f1,support=precision_recall_fscore_support(y[split=='test'],pred,labels=classes,zero_division=0)
    pd.DataFrame(dict(genreId=classes,precision=precision,recall=recall,f1=f1,support=support)).to_csv(base.with_suffix('.per_class.csv'),index=False)
    pd.DataFrame(confusion_matrix(y[split=='test'],pred,labels=classes),index=classes,columns=classes).to_csv(base.with_suffix('.confusion.csv'))
    with base.with_suffix('.model.npz').open('xb') as handle:
        np.savez_compressed(handle,coef=model.coef_,intercept=model.intercept_,classes=model.classes_.astype(str))
    summary.update(representation=rep,cell=cell,test_n=int((split=='test').sum()))
    embed.write_json_create_only(base.with_suffix('.json'),summary)
    return summary


def probe(out,legacy,approved_by,approval_reference):
    require(approved_by.strip() and approval_reference.strip(),'Approved probe contract required')
    proposal=read(PROPOSAL)
    for name,digest in proposal['artifacts_sha256'].items():
        require(embed.sha256(PROPOSAL.parent/name)==digest,'Fold hash changed')
    folds=pd.read_parquet(PROPOSAL.parent/'proposed_app_folds.parquet')
    require(folds.groupby('developerId').split.nunique().max()==1 and folds.groupby('component').split.nunique().max()==1,'Group leakage')
    classes=sorted(legacy.frames['tn1'].genreId.unique())
    summaries=[]
    for rep in NEURAL:
        cells=[('tn1','vi','char_budget'),('tn1','vi','native')]
        if rep in ('bge_m3','qwen3_embedding_0_6b'):
            cells.extend([('tn2','en','native'),('tn2','vi','native')])
        for exp,lang,condition in cells:
            obj=legacy.get(exp,rep,lang,condition)
            summaries.append(export_probe(out,rep,':'.join([exp,lang,condition]),obj['X'],legacy.frames[exp],folds,classes))
    # Inductive lexical baseline: learn vocabulary/IDF/SVD on training apps only.
    # Freeze those transforms; classifier may refit on train+validation after C selection.
    for exp,conditions in [('tn1',('char_budget','native')),('tn2',('native',))]:
        frame=legacy.frames[exp]
        aligned=frame[['appId']].merge(folds[['appId','split']],on='appId',how='left',validate='one_to_one')
        train=(aligned.split=='train').to_numpy()
        for condition in conditions:
            langs,corpora=lexical_texts(legacy.frames,legacy.specs['tfidf_lsa'],exp,condition)
            arrays,vec,svd=lexical_features(corpora,legacy.specs['tfidf_lsa'],[train]*len(langs))
            folder=out/('lexical_train_only__'+exp+'__'+condition)
            folder.mkdir()
            joblib.dump(dict(vectorizer=vec,svd=svd),folder/'fit.joblib',compress=3)
            embed.write_json_create_only(folder/'fit_app_ids.json',frame.appId[train].tolist())
            embed.write_json_create_only(folder/'fit_metadata.json',dict(fit_scope='train apps only; joint EN+VI training views for TN2',
                n_fit_apps=int(train.sum()),n_fit_texts=int(train.sum())*len(langs),vocabulary_size=len(vec.vocabulary_),
                fit_app_ids_sha256=embed.ordered_ids_sha256(frame.appId[train]),split_proposal_sha256=embed.sha256(PROPOSAL)))
            for lang,x in zip(langs,arrays):
                with (folder/('features__'+lang+'.npy')).open('xb') as handle:
                    np.save(handle,x)
                summaries.append(export_probe(out,'tfidf_lsa_train_only',':'.join([exp,lang,condition]),x,frame,folds,classes))
    pd.DataFrame([{k:v for k,v in s.items() if k!='tuning'} for s in summaries]).to_csv(out/'probe_summary.csv',index=False)
    return dict(status='REMAINING_18_PROBES_COMPLETE',cells=summaries,n_probe_cells=len(summaries),
        approved_by=approved_by,approval_reference=approval_reference,split_proposal_sha256=embed.sha256(PROPOSAL),
        scope='14 frozen-neural probes,4 train-only lexical probes; no T512 probe/PCA/learning curve')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['lexical_audit','evaluate','probe'])
    parser.add_argument('run_id')
    parser.add_argument('--lexical-receipt',type=Path)
    parser.add_argument('--approved-by',default='')
    parser.add_argument('--approval-reference',default='')
    args=parser.parse_args()
    require(args.run_id and all(c.isalnum() or c in '_-' for c in args.run_id),'Invalid run ID')
    out=ROOT/'results/stage05'/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    receipt=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        mode=args.mode,packages=embed.package_versions(),code_sha256={p:embed.sha256(ROOT/p) for p in ['pipeline/stage05_remaining.py','pipeline/evaluate_stage05.py','pipeline/probe_stage05.py','pipeline/evaluate_reviewed.py','pipeline/embed.py','pipeline/models.yaml']})
    start=time.perf_counter()
    legacy=None
    try:
        legacy=Legacy()
        with threadpool_limits(limits=4):
            if args.mode=='lexical_audit':
                receipt.update(lexical_audit(out,legacy))
            elif args.mode=='evaluate':
                require(args.lexical_receipt is not None,'Lexical receipt required')
                receipt.update(evaluate(out,legacy,args.lexical_receipt))
            else:
                receipt.update(probe(out,legacy,args.approved_by,args.approval_reference))
    except Exception as exc:
        receipt.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        receipt['elapsed_seconds']=time.perf_counter()-start
        receipt['input_lineage']=legacy.lineage if legacy is not None else {}
        receipt['artifacts_sha256']={p.relative_to(out).as_posix():embed.sha256(p) for p in out.rglob('*') if p.is_file()}
        embed.write_json_create_only(out/'execution_receipt.json',receipt)
    print(json.dumps(dict(status=receipt['status'],receipt=str(out/'execution_receipt.json'))),flush=True)


if __name__=='__main__':
    main()
