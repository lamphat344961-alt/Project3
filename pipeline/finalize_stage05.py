"""Verify remaining outputs and assemble the Stage05 scientific result tables."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,adjusted_mutual_info_score,adjusted_rand_score
from sklearn.preprocessing import Normalizer
from threadpoolctl import threadpool_limits

from pipeline import embed,evaluate_reviewed as metrics
from pipeline.evaluate_stage05 import provenance_labels,agreement_rows
from pipeline.stage05_remaining import ROOT,Legacy,PROPOSAL,lexical_texts,REMAINING

RESULTS=ROOT/'results/stage05'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def checked_receipt(folder,status):
    receipt=read(folder/'execution_receipt.json')
    require(receipt['status']==status,'Run not complete: '+str(folder))
    for name,digest in receipt['artifacts_sha256'].items():
        require(embed.sha256(folder/name)==digest,'Artifact changed: '+str(folder/name))
    return receipt


def verify_queries(folder,frame):
    summary=read(folder/'retrieval_summary.json')
    queries=pd.read_json(folder/'retrieval_per_query.jsonl',lines=True)
    require(len(queries)==27314 and set(queries.direction)=={'EN_TO_VI','VI_TO_EN'},'Query coverage')
    require(summary['app_ids_sha256']==embed.ordered_ids_sha256(frame.appId),'Retrieval cohort')
    harmonic=np.r_[0.,np.cumsum(1./np.arange(1,13658))]
    for direction,rows in queries.groupby('direction',sort=False):
        require(rows.query_app_id.tolist()==frame.appId.tolist() and (rows.positive_target_app_id==rows.query_app_id).all() and (rows.target_count==13657).all(),'Query IDs/catalogue')
        lo,hi=rows.rank_min.to_numpy(),rows.rank_max.to_numpy()
        ties=hi-lo+1
        require((lo>=1).all() and (hi<=13657).all() and (ties>0).all() and np.array_equal(ties,rows.tie_group_size),'Rank intervals')
        for metric,column,k in [('recall_at_1_primary','expected_recall_at_1',1),('recall_at_10_supplementary','expected_recall_at_10',10),('mrr_supplementary','expected_reciprocal_rank',None)]:
            expected=(harmonic[hi]-harmonic[lo-1])/ties if k is None else np.minimum(ties,np.maximum(0,k-(lo-1)))/ties
            require(np.allclose(expected,rows[column],rtol=0,atol=1e-12),'Tie metric arithmetic')
            require(abs(rows[column].mean()-summary['aggregate'][direction][metric])<=1e-12,'Retrieval aggregate')
    return summary


def verify_remaining_evaluation(legacy,folder):
    receipt=checked_receipt(folder,'REMAINING_RECIPES_EVALUATED')
    table=pd.read_csv(folder/'taxonomy_by_seed.csv')
    require(len(table)==240 and not table.duplicated(['representation','cell','seed']).any(),'Remaining taxonomy coverage')
    for row in legacy.manifest.itertuples():
        obj=legacy.get(row.experiment,row.representation,row.language,row.truncation)
        cell=':'.join([row.experiment,row.language,row.truncation])
        rows=table[(table.representation==row.representation)&(table.cell==cell)].set_index('seed')
        require(sorted(rows.index.tolist())==list(range(42,52)),'Seed coverage')
        for seed,labels in zip(range(42,52),obj['main']):
            require(all(abs(rows.loc[seed,k]-v)<=1e-12 for k,v in metrics.external_scores(legacy.ev.datasets[row.experiment]['y'],labels).items()),'Taxonomy recomputation')
    strata=pd.read_csv(folder/'provenance_taxonomy_by_seed.csv')
    require(len(strata)==360 and not strata.duplicated(['representation','cell','seed','provenance']).any(),'Provenance coverage')
    groups=provenance_labels(legacy.frames['tn1'])
    for (rep,cell),rows in strata.groupby(['representation','cell']):
        obj=legacy.get('tn1',rep,'vi',cell.split(':')[2])
        for row in rows.itertuples():
            mask=(groups==row.provenance).to_numpy()
            true=legacy.ev.datasets['tn1']['y'][mask]
            pred=obj['main'][int(row.seed)-42][mask]
            require(int(row.n_apps)==mask.sum() and abs(row.ami-adjusted_mutual_info_score(true,pred))<=1e-12 and abs(row.ari-adjusted_rand_score(true,pred))<=1e-12,'Provenance recomputation')
    for rep in ('tfidf_lsa','bge_m3','qwen3_embedding_0_6b'):
        for condition in ('native','token_budget'):
            pair_folder=folder/(rep+'__'+condition)
            en=legacy.get('tn2',rep,'en',condition)
            vi=legacy.get('tn2',rep,'vi',condition)
            stored=pd.read_csv(pair_folder/'partition_pairs.csv')
            expected=pd.DataFrame(agreement_rows(en['main'],vi['main']))
            require(len(stored)==190 and np.array_equal(stored[['component','seed','seed_other']],expected[['component','seed','seed_other']]),'Partition pair indices')
            require(np.allclose(stored[['ami','ari','nmi']],expected[['ami','ari','nmi']],rtol=0,atol=1e-12),'Partition metric recomputation')
            ps=read(pair_folder/'partition_summary.json')
            for role in ('C','W_EN','W_VI'):
                require(abs(ps[role]-stored[stored.component==role].ami.mean())<=1e-12,'Partition summary')
            require(abs(ps['S']-(ps['C']-(ps['W_EN']+ps['W_VI'])/2))<=1e-12,'S formula')
            pairs=pd.read_parquet(pair_folder/'paired_correspondence.parquet')
            require(pairs.appId.tolist()==legacy.frames['tn2'].appId.tolist() and np.allclose(pairs.delta_cosine,pairs.matched_cosine-pairs.negative_cosine_median,rtol=0,atol=1e-12),'Cosine mapping/delta')
            verify_queries(pair_folder,legacy.frames['tn2'])
    return receipt


def verify_remaining_probes(legacy,folder):
    receipt=checked_receipt(folder,'REMAINING_18_PROBES_COMPLETE')
    require(len(receipt['cells'])==18 and receipt['split_proposal_sha256']==embed.sha256(PROPOSAL),'Probe scope/split')
    proposal=read(PROPOSAL)
    for name,digest in proposal['artifacts_sha256'].items():
        require(embed.sha256(PROPOSAL.parent/name)==digest,'Split changed')
    folds=pd.read_parquet(PROPOSAL.parent/'proposed_app_folds.parquet')
    require(folds.groupby('component').split.nunique().max()==1 and folds.groupby('developerId').split.nunique().max()==1,'Group leakage')
    classes=sorted(legacy.frames['tn1'].genreId.unique())
    lexical_cache={}
    for result in receipt['cells']:
        rep,cell=result['representation'],result['cell']
        exp,lang,condition=cell.split(':')
        frame=legacy.frames[exp]
        split=frame[['appId']].merge(folds[['appId','split']],on='appId',how='left',validate='one_to_one').split.to_numpy()
        test=split=='test'
        y=frame.genreId.to_numpy()
        if rep=='tfidf_lsa_train_only':
            feature_folder=folder/('lexical_train_only__'+exp+'__'+condition)
            if (exp,condition) not in lexical_cache:
                fit_ids=read(feature_folder/'fit_app_ids.json')
                require(fit_ids==frame.appId[split=='train'].tolist(),'Lexical fit support excludes heldout')
                states=joblib.load(feature_folder/'fit.joblib')
                langs,corpora=lexical_texts(legacy.frames,legacy.specs['tfidf_lsa'],exp,condition)
                train_texts=[text for corpus in corpora for text,fit in zip(corpus,split=='train') if fit]
                expected=embed.make_tfidf_vectorizer(legacy.specs['tfidf_lsa']).fit(train_texts)
                require(expected.vocabulary_==states['vectorizer'].vocabulary_ and np.allclose(expected.idf_,states['vectorizer'].idf_,rtol=0,atol=1e-12),'Lexical vocabulary/IDF is train-only')
                for view,texts in zip(langs,corpora):
                    actual=np.load(feature_folder/('features__'+view+'.npy'),mmap_mode='r',allow_pickle=False)
                    replay=Normalizer(copy=False).fit_transform(states['svd'].transform(states['vectorizer'].transform(texts)).astype(np.float32))
                    require(actual.shape==(len(frame),300) and np.isfinite(actual).all() and np.max(np.abs(replay-actual))<=1e-5,'Lexical feature replay')
                lexical_cache[(exp,condition)]=True
            x=np.load(feature_folder/('features__'+lang+'.npy'),mmap_mode='r',allow_pickle=False)
        else:
            x=legacy.get(exp,rep,lang,condition)['X']
        base=folder/(rep+'__'+cell.replace(':','__'))
        predictions=pd.read_parquet(base.with_suffix('.test_predictions.parquet'))
        require(predictions.appId.tolist()==frame.appId[test].tolist() and predictions.genreId.tolist()==y[test].tolist(),'Probe test mapping')
        require([r['C'] for r in result['tuning']]==proposal['proposed_probe']['C_grid'],'Probe C grid')
        require(result['selected_C']==max(result['tuning'],key=lambda r:r['validation_macro_f1'])['C'],'Validation-only C selection')
        saved=np.load(base.with_suffix('.model.npz'),allow_pickle=False)
        require(saved['classes'].tolist()==classes and saved['coef'].shape==(49,x.shape[1]),'Probe model classes/dim')
        replay=saved['classes'][np.argmax(np.asarray(x[test],dtype=np.float64)@saved['coef'].T+saved['intercept'],axis=1)]
        require(np.array_equal(replay,predictions.prediction),'Saved model prediction replay')
        require(abs(f1_score(y[test],replay,labels=classes,average='macro',zero_division=0)-result['test_macro_f1'])<=1e-12 and abs(accuracy_score(y[test],replay)-result['test_accuracy'])<=1e-12,'Probe metrics')
        cm=pd.read_csv(base.with_suffix('.confusion.csv'),index_col=0)
        require(np.array_equal(cm,confusion_matrix(y[test],replay,labels=classes)),'Probe confusion')
        pc=pd.read_csv(base.with_suffix('.per_class.csv'))
        require(pc.genreId.tolist()==classes and np.array_equal(pc.support.to_numpy(),cm.sum(axis=1).to_numpy()) and abs(pc.f1.mean()-result['test_macro_f1'])<=1e-12,'Probe per-class support/F1')
    return receipt


def combine(out,remaining,probe):
    tables=[]
    batches=[('labse','labse_20261025_001','LABSE_CLUSTERING_RETRIEVAL_CORRESPONDENCE_COMPLETE'),
             ('multilingual_e5_large','e5_tn1_20261025_001','E5_TN1_CLUSTERING_PROVENANCE_COMPLETE'),
             ('multilingual_e5_large','e5_tn2_20261025_001','E5_TN2_CLUSTERING_RETRIEVAL_CORRESPONDENCE_COMPLETE')]
    provenance=[]
    for rep,run_id,status in batches:
        checked_receipt(RESULTS/run_id,status)
        table=pd.read_csv(RESULTS/run_id/'taxonomy_by_seed.csv')
        table['representation']=rep
        tables.append(table)
        if (RESULTS/run_id/'provenance_taxonomy_by_seed.csv').exists():
            strata=pd.read_csv(RESULTS/run_id/'provenance_taxonomy_by_seed.csv')
            strata['representation']=rep
            provenance.append(strata)
    tables.append(pd.read_csv(remaining/'taxonomy_by_seed.csv'))
    provenance.append(pd.read_csv(remaining/'provenance_taxonomy_by_seed.csv'))
    table=pd.concat(tables,ignore_index=True)
    expected={('tn1',rep,'vi',condition) for rep in (*REMAINING,'multilingual_e5_large','labse') for condition in ('char_budget','native')}
    expected|={('tn2',rep,lang,condition) for rep in ('tfidf_lsa','multilingual_e5_large','bge_m3','qwen3_embedding_0_6b') for lang in ('en','vi') for condition in ('native','token_budget')}
    expected|={('tn2','labse',lang,'native') for lang in ('en','vi')}
    actual={(r.cell.split(':')[0],r.representation,r.cell.split(':')[1],r.cell.split(':')[2]) for r in table.itertuples()}
    require(actual==expected and len(table)==340 and not table.duplicated(['representation','cell','seed']).any(),'Complete34-cell taxonomy coverage')
    require(all(sorted(g.seed.tolist())==list(range(42,52)) for _,g in table.groupby(['representation','cell'])),'Ten seeds per cell')
    table.to_csv(out/'taxonomy_all_by_seed.csv',index=False)
    summary=table.groupby(['representation','cell'])[['ami','ari']].agg(['mean','std','min','max'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.to_csv(out/'taxonomy_summary.csv')
    strata=pd.concat(provenance,ignore_index=True)
    require(len(strata)==480,'All provenance rows')
    strata.to_csv(out/'provenance_all_by_seed.csv',index=False)
    old_probe=RESULTS/'validation_probes_20261025_001'
    old_receipt=read(old_probe/'validation_receipt.json')
    require(old_receipt['status']=='EIGHT_LABSE_E5_PROBES_VALIDATION_PASS','Prior probe validation')
    for name,digest in old_receipt['artifacts_sha256'].items():
        require(embed.sha256(old_probe/name)==digest,'Prior probe validation artifact changed')
    for run_id,digest in old_receipt['run_receipts_sha256'].items():
        r=read(RESULTS/run_id/'execution_receipt.json')
        require(embed.sha256(RESULTS/run_id/'execution_receipt.json')==digest,'Prior probe receipt changed')
        for name,h in r['artifacts_sha256'].items():
            require(embed.sha256(RESULTS/run_id/name)==h,'Prior probe model changed')
    old=pd.read_csv(old_probe/'probe_comparison.csv').rename(columns={'model':'representation','majority_macro_f1':'majority_test_macro_f1'})
    old['representation']=old.representation.replace({'e5':'multilingual_e5_large'})
    new=pd.read_csv(probe/'probe_summary.csv')
    probes=pd.concat([old,new],ignore_index=True)
    require(len(probes)==26 and not probes.duplicated(['representation','cell']).any(),'Complete26 probe cells')
    expected_probes={(rep,'tn1:vi:'+condition) for rep in ('labse','multilingual_e5_large','tfidf_lsa_train_only','phobert_base_v2','sup_simcse_phobert','bkai_vietnamese_biencoder','bge_m3','qwen3_embedding_0_6b') for condition in ('char_budget','native')}
    expected_probes|={(rep,'tn2:'+lang+':native') for rep in ('labse','multilingual_e5_large','tfidf_lsa_train_only','bge_m3','qwen3_embedding_0_6b') for lang in ('en','vi')}
    require(set(zip(probes.representation,probes.cell))==expected_probes,'Probe recipe/cohort coverage')
    probes.to_csv(out/'probe_all.csv',index=False)
    retrieval=[]
    partition=[]
    for rep,folder,conditions in [('labse',RESULTS/'labse_20261025_001',('native',)),('multilingual_e5_large',RESULTS/'e5_tn2_20261025_001',('native','token_budget')),*[(rep,remaining,('native','token_budget')) for rep in ('tfidf_lsa','bge_m3','qwen3_embedding_0_6b')]]:
        for condition in conditions:
            pair_folder=folder if rep=='labse' else folder/condition if rep=='multilingual_e5_large' else folder/(rep+'__'+condition)
            for direction,values in read(pair_folder/'retrieval_summary.json')['aggregate'].items():
                require(values['direction']==direction,'Retrieval direction label')
                retrieval.append(dict(representation=rep,condition=condition,**values))
            ps=read(pair_folder/'partition_summary.json')
            partition.append(dict(representation=rep,condition=condition,**ps))
    require(len(retrieval)==18 and len(partition)==9,'All paired evaluation coverage')
    pd.DataFrame(retrieval).to_csv(out/'retrieval_all.csv',index=False)
    pd.DataFrame(partition).to_csv(out/'partition_all.csv',index=False)
    index=[]
    for exp,rep,lang,condition in sorted(expected):
        stem='__'.join([exp,rep,lang,condition])
        if rep=='labse':
            base=ROOT/'results/labse_stage03'/('labse_local_20261024_204813_'+'_'.join([exp,lang,condition]))/stem
            label=RESULTS/'labse_20261025_001'/(':'.join([exp,lang,condition]).replace(':','__')+'__kmeans_main.npz')
        elif rep=='multilingual_e5_large':
            run_id='kg_20261025_022156_432511_tn1_034318_693554' if exp=='tn1' else 'kg_20261025_022156_432511_tn2_022259_065085'
            base=ROOT/'results/e5_stage02'/run_id/stem
            label=RESULTS/('e5_'+exp+'_20261025_001')/(':'.join([exp,lang,condition]).replace(':','__')+'__kmeans_main.npz')
        else:
            base=ROOT/'embeddings'/exp/stem
            label=ROOT/'results/labels'/(stem+'__kmeans_main.npz')
        meta=read(base.with_suffix('.json'))
        index.append(dict(experiment=exp,representation=rep,language=lang,condition=condition,
            embedding_path=base.with_suffix('.npy').relative_to(ROOT).as_posix(),embedding_sha256=embed.sha256(base.with_suffix('.npy')),
            rows_sha256=embed.sha256(base.with_suffix('.rows.parquet')),metadata_sha256=embed.sha256(base.with_suffix('.json')),
            labels_path=label.relative_to(ROOT).as_posix(),labels_sha256=embed.sha256(label),
            dimension=meta.get('embedding_dim',meta.get('recipe',{}).get('dim',meta.get('checkpoint',{}).get('config_summary',{}).get('dimension'))),
            revision=meta.get('revision',meta.get('recipe',{}).get('revision')),k=49,n_init=10,seeds='42-51'))
    pd.DataFrame(index).to_csv(out/'artifact_index.csv',index=False)
    return dict(n_taxonomy_cells=34,n_taxonomy_seed_rows=340,n_provenance_rows=480,n_probe_cells=26,n_paired_conditions=9,
        n_total_retrieval_queries=245826,n_total_partition_pairs=1710,
        interpretations=['Recipe/input-condition comparisons, not causal model-quality effects.','TN2 entirely translation-flagged; no extrapolation to independently authored translations.','S and seed SD descriptive; no iid pair bootstrap or population confidence interval.','TF-IDF historical clustering/retrieval transductive; probe TF-IDF train-only fit.','Historical neural compute precision not recorded in some metadata; no assumption all compute FP32.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    parser.add_argument('--remaining-run',default='remaining_eval_20261025_001')
    parser.add_argument('--probe-run',default='remaining_probe_20261025_001')
    args=parser.parse_args()
    require(all(v and all(c.isalnum() or c in '_-' for c in v) for v in [args.run_id,args.remaining_run,args.probe_run]),'Invalid run ID')
    out=RESULTS/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    remaining,probe=RESULTS/args.remaining_run,RESULTS/args.probe_run
    receipt=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),code_sha256=embed.sha256(Path(__file__)))
    try:
        with threadpool_limits(limits=4):
            legacy=Legacy()
            lexical=RESULTS/'lexical_lineage_20261025_001'
            checked_receipt(lexical,'LEXICAL_RECONSTRUCTION_PASS')
            lineage=read(lexical/'lexical_lineage.json')
            require(len(lineage['cells'])==6 and all(c['passed'] and c['max_abs_difference']<=1e-5 for c in lineage['cells']),'Lexical shared-space lineage')
            print('VERIFY REMAINING EVALUATION',flush=True)
            verify_remaining_evaluation(legacy,remaining)
            print('VERIFY REMAINING PROBES',flush=True)
            verify_remaining_probes(legacy,probe)
            receipt.update(combine(out,remaining,probe))
        parents=[remaining/'execution_receipt.json',probe/'execution_receipt.json',RESULTS/'lexical_lineage_20261025_001/execution_receipt.json',RESULTS/'validation_probes_20261025_001/validation_receipt.json']
        parents.extend(RESULTS/run/'execution_receipt.json' for run in ('labse_20261025_001','e5_tn1_20261025_001','e5_tn2_20261025_001'))
        receipt.update(status='STAGE05_COMPLETE_VALIDATED',input_receipts_sha256={str(p.relative_to(ROOT)):embed.sha256(p) for p in parents})
    except Exception as exc:
        receipt.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        receipt['artifacts_sha256']={p.name:embed.sha256(p) for p in out.iterdir() if p.is_file()}
        embed.write_json_create_only(out/'execution_receipt.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
