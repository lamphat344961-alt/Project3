#!/usr/bin/env python3
"""PAAC-VN Gate 5; fixed-label, paired-app bootstrap (NO embedding/recluster).

Modes: --mode pilot (mandatory exact Gate4 point checks), --mode prepare (create
shared immutable RNG inputs and run_config once), --mode full (B=1000, per-config
independent workers, resumable atomic chunks), --mode finalize (requires ALL
expected blocks before summarizing).
Source: amended CLAUDE.md Gate5 addendum. Never overwrites canonical inputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.metrics import adjusted_mutual_info_score as ami

B = 1000
SEEDS = tuple(range(42, 52))
REPS1 = ('bge_m3','bkai_vietnamese_biencoder','multilingual_e5_large',
         'phobert_base_v2','qwen3_embedding_0_6b','sup_simcse_phobert',
         'tfidf_lsa','videberta_base')
REPS2 = ('bge_m3','multilingual_e5_large','qwen3_embedding_0_6b','tfidf_lsa')
TR1 = ('char_budget','native')
TR2 = ('token_budget','native')
LEADS = ('multilingual_e5_large','bge_m3','qwen3_embedding_0_6b')
DATA_HASH = {'tn1':'0cf5e72c00f1392e9f7e83ef2fc282e8deb9e81bc6e5d332ce5d761a41f7c52b',
             'tn2':'9208bfc42fc66b45c19496af34088cb742d1ea2e7cca25998e139b16c7e77493'}
MODELS_HASH = 'bdc8a5f2f72d699ff76dbedf5ceca015d5f2c2d2ff531c6289d900fd4b3e01d6'
TOL = 1e-8   # check against exact stored double AMI, not arbitrary inferential cutoff
MARKER = 'PAAC-GATE5-AMENDMENT-2026-10-04'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for blk in iter(lambda:f.read(4<<20),b''): h.update(blk)
    return h.hexdigest()


def rd(path):
    with Path(path).open(encoding='utf8') as f: return json.load(f)


def write_json_new(path, content):
    path=Path(path)
    # Exclusive file creation; do not silently overwrite existing reports.
    with path.open('x',encoding='utf8') as f: json.dump(content,f,ensure_ascii=False,indent=2,allow_nan=False)


def atomic_npz_new(path, **arrays):
    path=Path(path)
    if path.exists(): raise FileExistsError(path)
    temp=path.with_suffix('.partial.npz')
    if temp.exists(): raise FileExistsError(temp)
    try:
        # Windows: fsync() on a read-only CRT descriptor raises EBADF.
        # Save, flush and fsync on the SAME writable handle, then close.
        with temp.open('wb') as f:
            np.savez_compressed(f,**arrays)
            f.flush()
            os.fsync(f.fileno())
        # no writers expected, but make accidental overwrites fail
        if path.exists(): raise FileExistsError(path)
        os.replace(temp,path)
    finally:
        if temp.exists(): temp.unlink()


def file_meta(repo):
    epath=repo/'results/evaluation/evaluation_meta.json'
    e=rd(epath)
    assert e['gate']==4 and e['status']=='COMPLETE' and e['n_metrics_rows']==30128
    paths={'metrics':repo/'results/evaluation/metrics_long.csv',
           'pairs':repo/'results/evaluation/tn2_a_app_pairs.parquet',
           'cluster_manifest':repo/'results/audit/clustering_manifest.csv',
           'evaluation_meta':epath,
           'model':repo/'pipeline/models.yaml',
           'tn1':repo/'data_v1/benchmark_v1.parquet',
           'tn2':repo/'data_v1/paired_en.parquet'}
    actual={key:sha(p) for key,p in paths.items()}
    expect={'metrics':e['metrics_long_sha256'], 'pairs':e['tn2_a_app_pairs_sha256'],
            'cluster_manifest':e['clustering_manifest_sha256'],
            'model':MODELS_HASH, **DATA_HASH}
    for k,h in expect.items():
        if actual[k]!=h: raise ValueError(f'INPUT SHA MISMATCH {k}: {actual[k]} vs {h}')
    if e.get('dataset_sha256')!=DATA_HASH: raise ValueError('Gate 4 dataset metadata mismatch')
    return e,actual


def amendment(repo):
    p=repo/'CLAUDE.md'
    src=p.read_text(encoding='utf8')
    if src.count(MARKER)!=1: raise RuntimeError('Gate 5 amendment missing or ambiguous in CLAUDE.md')
    # Timestamp must be an already-past explicit date; reject full run from future-dated addendum.
    import re
    m=re.search(r'PAAC-GATE5-AMENDMENT-2026-10-04[^\n]*\n\s*\*\*Approved/amended at:\*\*\s*`([^`]+)`',src)
    if not m: raise RuntimeError('Gate5 amendment timestamp missing')
    stamped=datetime.fromisoformat(m.group(1))
    if stamped.tzinfo is None or stamped.astimezone(timezone.utc)>=datetime.now(timezone.utc):
        raise RuntimeError('Amendment must precede execution')
    return {'sha256':sha(p),'timestamp':m.group(1)}


def key(exp,rep,lang,tr):return f'{exp}__{rep}__{lang}__{tr}'


# A configuration is one experiment × representation × truncation. TN2 uses
# both languages and the 190 fixed partition pairs within that one block.
ALL_CONFIGS = tuple((exp, rep, tr)
    for exp,reps,trs in (('tn1',REPS1,TR1),('tn2',REPS2,TR2))
    for rep in reps for tr in trs)
WEIGHT_KEYS = tuple((exp,rep,tr,lang)
    for exp,rep,tr in ALL_CONFIGS
    for lang in (('vi',) if exp=='tn1' else ('en','vi')))


def config_name(exp,rep,tr):
    return f'{exp}__{rep}__{tr}'


def weight_name(exp,rep,tr,lang):
    return f'seed_weights__{exp}__{rep}__{tr}__{lang}.npy'


def parse_configs(tokens):
    """Accept comma/space-separated canonical IDs; never silently skip typos."""
    if not tokens:
        return ALL_CONFIGS
    requested = [s.strip() for token in tokens for s in token.split(',') if s.strip()]
    if not requested or 'all' in requested:
        if requested not in ([], ['all']):
            raise ValueError("Use --configs all OR explicit config IDs")
        return ALL_CONFIGS
    names={config_name(*t):t for t in ALL_CONFIGS}
    if len(requested)!=len(set(requested)):
        raise ValueError('Duplicate config in --configs')
    unknown=set(requested)-set(names)
    if unknown:
        raise ValueError(f'Unknown --configs: {sorted(unknown)}; use experiment__representation__truncation')
    return tuple(names[s] for s in requested)


def get_expected(df,**conditions):
    x=df
    for col,val in conditions.items():
        if val is None: x=x[x[col].isna()]
        else: x=x[x[col]==val]
    if len(x)!=1:
        raise RuntimeError(f'Gate4 lookup expected one row got {len(x)}: {conditions}')
    return float(x.iloc[0]['value'])


class Inputs:
    def __init__(self, repo):
        self.repo=repo
        self.e,self.files=file_meta(repo)
        self.df=pd.read_csv(repo/'results/evaluation/metrics_long.csv')
        if len(self.df)!=30128: raise ValueError('Incorrect metrics_long row count')
        m=pd.read_csv(repo/'results/audit/clustering_manifest.csv')
        if len(m)!=32 or (m['status']!='COMPLETE').any(): raise ValueError('Gate3 manifest not complete')
        self.manifest={key(r.experiment,r.representation,r.language,r.truncation):r
                       for r in m.itertuples(index=False)}
        self.y={};self.ids={};self.genre={}
        for exp,filename,n in [('tn1','benchmark_v1.parquet',23883),('tn2','paired_en.parquet',13657)]:
            data=pd.read_parquet(repo/'data_v1'/filename,columns=['appId','genreId'])
            if len(data)!=n or data.genreId.nunique()!=49 or data.appId.isna().any() or data.appId.duplicated().any():
                raise ValueError(f'Invalid canonical {exp} identity/genre')
            self.y[exp]=pd.Categorical(data.genreId.astype(str)).codes.astype(np.int32)
            self.ids[exp]=data.appId.astype(str).to_numpy()
            self.genre[exp]=data.genreId.astype(str).to_numpy()
        self.labels={}
        for exp,reps,trs,langs in (('tn1',REPS1,TR1,('vi',)),('tn2',REPS2,TR2,('en','vi'))):
            for rep in reps:
                for tr in trs:
                    for lang in langs:
                        name=key(exp,rep,lang,tr)
                        source=self.e['sources'][name]
                        base=repo/'embeddings'/exp/name
                        if sha(base.with_suffix('.json'))!=source['sha256_json']:
                            raise ValueError('Embedding sidecar JSON changed: '+name)
                        meta=rd(base.with_suffix('.json'))
                        expected_row_hash=hashlib.sha256('\n'.join(self.ids[exp]).encode()).hexdigest()
                        if (meta.get('dataset_sha256')!=DATA_HASH[exp]
                                or meta.get('models_yaml_sha256')!=MODELS_HASH
                                or meta.get('app_ids_sha256')!=expected_row_hash):
                            raise ValueError('Metadata inconsistent: '+name)
                        rows=pd.read_parquet(base.with_suffix('.rows.parquet'),columns=['row_idx','appId','genreId'])
                        if (not np.array_equal(rows.row_idx.to_numpy(),np.arange(len(self.y[exp])))
                                or not np.array_equal(rows.appId.astype(str).to_numpy(),self.ids[exp])
                                or not np.array_equal(rows.genreId.astype(str).to_numpy(),self.genre[exp])):
                            raise ValueError('Row sidecar misalignment: '+name)
                        audit=self.manifest[name]
                        lbl=repo/'results/labels'/f'{name}__kmeans_main.npz'
                        if sha(lbl)!=audit.kmeans_main_sha256: raise ValueError('Label SHA changed: '+name)
                        with np.load(lbl,allow_pickle=False) as z:
                            x=z['labels']
                            if (x.shape!=(10,len(self.y[exp])) or list(z['seeds'])!=list(SEEDS)
                                    or int(z['n_init'])!=10 or int(z['k'])!=49):
                                raise ValueError('Frozen KMeans protocol mismatch: '+name)
                            self.labels[name]=np.asarray(x,dtype=np.int32)
        pairs=pd.read_parquet(repo/'results/evaluation/tn2_a_app_pairs.parquet')
        if len(pairs)!=8*13657: raise ValueError('TN2-A expected 109256 rows')
        self.delta={}
        for rep in REPS2:
            for tr in TR2:
                rows=pairs[(pairs.representation==rep)&(pairs.truncation==tr)].sort_values('row_idx')
                if (len(rows)!=13657 or not np.array_equal(rows.row_idx.to_numpy(),np.arange(13657))
                        or not np.array_equal(rows.appId.astype(str).to_numpy(),self.ids['tn2'])
                        or not np.array_equal(rows.genreId.astype(str).to_numpy(),self.genre['tn2'])):
                    raise ValueError('TN2-A row mismatch: '+rep+' '+tr)
                if not np.allclose(rows.matched_cosine.to_numpy()-rows.negative_cosine_median.to_numpy(),
                                   rows.delta_cosine.to_numpy(),atol=2e-7,rtol=0):
                    raise ValueError('TN2-A row arithmetic invalid')
                self.delta[(rep,tr)]=rows.delta_cosine.to_numpy(dtype=np.float64)

    def point_checks(self):
        """Cross-check every main seed in TN1/C, 190 TN2-B scores per config, TN2-A means.
        Pilot exactness must precede any full bootstrap. No optimization or selection.
        """
        df=self.df; records=[]
        def test(name,v,recorded):
            diff=abs(v-recorded)
            records.append((name,v,recorded,diff))
        for rep in REPS1:
            for tr in TR1:
                x=self.labels[key('tn1',rep,'vi',tr)]
                for i,s in enumerate(SEEDS):
                    val=get_expected(df,section='TN1_EXTERNAL',representation=rep,truncation=tr,
                                     language='vi',algorithm='kmeans_main',run='main',seed=float(s),metric='ami')
                    test(f'TN1/{rep}/{tr}/s{s}',ami(self.y['tn1'],x[i]),val)
        for rep in REPS2:
            for tr in TR2:
                en=self.labels[key('tn2',rep,'en',tr)]
                vi=self.labels[key('tn2',rep,'vi',tr)]
                for lang,labels in [('en',en),('vi',vi)]:
                    for i,s in enumerate(SEEDS):
                        val=get_expected(df,section='TN2_C',representation=rep,truncation=tr,
                                         language=lang,algorithm='kmeans_main',run='main',seed=float(s),metric='ami')
                        test(f'TN2-C/{rep}/{tr}/{lang}/s{s}',ami(self.y['tn2'],labels[i]),val)
                for i,s in enumerate(SEEDS):
                    for j,t in enumerate(SEEDS):
                        val=get_expected(df,section='TN2_B',representation=rep,truncation=tr,
                           language='en_vs_vi',algorithm='kmeans_main',run='cross_language_all_pairs',
                           seed=float(s),seed_other=float(t),metric='ami')
                        test(f'TN2-B/cross/{rep}/{tr}/{s}/{t}',ami(en[i],vi[j]),val)
                for lang,labels in [('en',en),('vi',vi)]:
                    for i,s in enumerate(SEEDS):
                        for j in range(i+1,10):
                            t=SEEDS[j]
                            val=get_expected(df,section='TN2_B',representation=rep,truncation=tr,
                                language=lang,algorithm='kmeans_main',run='within_language_all_pairs',
                                seed=float(s),seed_other=float(t),metric='ami')
                            test(f'TN2-B/within/{rep}/{tr}/{lang}/{s}/{t}',ami(labels[i],labels[j]),val)
                stat=get_expected(df,section='TN2_A',representation=rep,truncation=tr,
                                  language='en_vs_vi',algorithm='embedding_pair',run='paired_apps',
                                  metric='delta_cosine',pair_role='mean')
                test(f'TN2-A/{rep}/{tr}',float(np.mean(self.delta[(rep,tr)])),stat)
        errors=sorted((r for r in records if r[3]>TOL or not np.isfinite(r[3])),key=lambda x:-x[3])
        return {'status':'PASS' if not errors else 'SOURCE_DISCREPANCY',
                'n_checks':len(records),'max_abs_error':max(x[3] for x in records),
                'tolerance':TOL,
                'discrepancies':[{'id':i,'observed':v,'gate4':g,'error':d} for i,v,g,d in errors[:100]]}


def rng_plan():
    """Stable, explicit SeedSequence(42) child allocation across processes.

    0/1: TN1/TN2 shared app index matrices.
    2..: independent seed-weight matrices for EVERY config and language,
         allocated in the fixed order of WEIGHT_KEYS (never --configs order).
    """
    return np.random.SeedSequence(42).spawn(2+len(WEIGHT_KEYS))


def rng_description():
    return {
        'algorithm':'numpy.default_rng(SeedSequence(42).spawn(...))',
        'root_seed':42,'seedsequence_children':2+len(WEIGHT_KEYS),
        'app_child_indices':{'tn1':0,'tn2':1},
        'seed_weight_child_indices':{weight_name(*k):i+2 for i,k in enumerate(WEIGHT_KEYS)},
        'seed_draw':'10 independent categorical draws among 10 seeds per replicate, stored as integer counts',
        'independence':'separate SeedSequence child for every experiment/configuration/language',
    }


def create_rng_files(out):
    """ONLY --mode prepare creates RNG files. Never called by --mode full."""
    rs=rng_plan()
    specs=[('indices_tn1.npy',(B,23883),rs[0]),('indices_tn2.npy',(B,13657),rs[1])]
    for filename,shape,ss in specs:
        p=out/filename
        if not p.exists():
            r=np.random.default_rng(ss)
            partial=p.with_suffix('.partial.npy')
            if partial.exists(): raise FileExistsError(partial)
            arr=np.lib.format.open_memmap(partial,mode='w+',dtype=np.int32,shape=shape)
            try:
                for i in range(B):
                    arr[i]=r.integers(0,shape[1],size=shape[1],dtype=np.int32)
                arr.flush()
            finally:
                arr._mmap.close()
                del arr
            os.replace(partial,p)
    for i,k in enumerate(WEIGHT_KEYS):
        p=out/weight_name(*k)
        if p.exists():continue
        r=np.random.default_rng(rs[i+2])
        choices=r.integers(0,10,size=(B,10))
        mat=np.stack([np.bincount(row,minlength=10) for row in choices]).astype(np.int8)
        partial=p.with_suffix('.partial.npy')
        if partial.exists(): raise FileExistsError(partial)
        np.save(partial,mat,allow_pickle=False)
        os.replace(partial,p)


def rng_fingerprints(out):
    return {'indices_sha256':{e:sha(out/f'indices_{e}.npy') for e in ('tn1','tn2')},
            'seed_weights_sha256':{weight_name(*k):sha(out/weight_name(*k)) for k in WEIGHT_KEYS},
            'rng_derivation':rng_description()}


def load_rng(out,expected=None):
    """Read-only consumer; enforce run_config hashes before any computation."""
    observed=rng_fingerprints(out)
    if expected is not None and observed!=expected:
        raise RuntimeError('Shared RNG inputs/SeedSequence plan differ from run_config')
    indices={k:np.load(out/f'indices_{k}.npy',mmap_mode='r') for k in ('tn1','tn2')}
    weights={k:np.load(out/weight_name(*k),mmap_mode='r') for k in WEIGHT_KEYS}
    try:
        for e,n in (('tn1',23883),('tn2',13657)):
            x=indices[e]
            if x.shape!=(B,n) or x.dtype!=np.int32 or x.min()<0 or x.max()>=n:
                raise ValueError('Invalid resampling matrix: '+e)
        for k,w in weights.items():
            if w.shape!=(B,10) or np.any(w.sum(axis=1)!=10) or np.any(w<0):
                raise ValueError('Invalid seed weights: '+str(k))
    except Exception:
        close_mmaps(indices,weights)
        raise
    return indices,weights


def close_mmaps(*groups):
    """Windows-safe resource release (especially for TemporaryDirectory tests)."""
    for group in groups:
        for array in group.values():
            mm=getattr(array,'_mmap',None)
            if mm is not None and not mm.closed: mm.close()


def block_file(out,exp,rep,tr,lo,hi):
    return out/'blocks'/f'{exp}__{rep}__{tr}__{lo:04d}_{hi:04d}.npz'


def run_blocks(out, inp, indices, args, configs=None):
    """Worker: write only selected blocks; never creates RNG/run_config."""
    (out/'blocks').mkdir(exist_ok=True)
    selected=set(ALL_CONFIGS if configs is None else configs)
    pairs=np.array([(i,j) for i in range(10) for j in range(i+1,10)],dtype=np.int8)
    for rep in REPS1:
        for tr in TR1:
            if ('tn1',rep,tr) not in selected:continue
            labels=inp.labels[key('tn1',rep,'vi',tr)]
            for lo in range(0,B,args.tn1_block):
                hi=min(B,lo+args.tn1_block)
                path=block_file(out,'tn1',rep,tr,lo,hi)
                if path.exists(): continue
                a=np.empty((hi-lo,10),dtype=np.float64)
                for b in range(lo,hi):
                    ii=indices['tn1'][b]
                    yy=inp.y['tn1'][ii]
                    a[b-lo]=[ami(yy,l[ii]) for l in labels]
                atomic_npz_new(path,replicate=np.arange(lo,hi,dtype=np.int32),ami_vi=a)
                print('[TN1]',rep,tr,lo,hi,flush=True)
    for rep in REPS2:
        for tr in TR2:
            if ('tn2',rep,tr) not in selected:continue
            en=inp.labels[key('tn2',rep,'en',tr)]
            vi=inp.labels[key('tn2',rep,'vi',tr)]
            for lo in range(0,B,args.tn2_block):
                hi=min(B,lo+args.tn2_block)
                path=block_file(out,'tn2',rep,tr,lo,hi)
                if path.exists(): continue
                n=hi-lo
                ea=np.empty((n,10));va=np.empty((n,10)); cross=np.empty((n,10,10))
                within_e=np.empty((n,45));within_v=np.empty((n,45))
                for b in range(lo,hi):
                    ii=indices['tn2'][b]; yy=inp.y['tn2'][ii]; idx=b-lo
                    a=en[:,ii];c=vi[:,ii]
                    for s in range(10):
                        ea[idx,s]=ami(yy,a[s]);va[idx,s]=ami(yy,c[s])
                        for t in range(10): cross[idx,s,t]=ami(a[s],c[t])
                    for k,(s,t) in enumerate(pairs):
                        within_e[idx,k]=ami(a[s],a[t]);within_v[idx,k]=ami(c[s],c[t])
                atomic_npz_new(path,replicate=np.arange(lo,hi,dtype=np.int32),
                               ami_en=ea,ami_vi=va,cross=cross,
                               within_en=within_e,within_vi=within_v)
                print('[TN2-B CHECKPOINT]',rep,tr,lo,hi,flush=True)


def load_blocks(out,exp,rep,tr,block):
    parts={}; seen=np.zeros(B,dtype=np.int8)
    for lo in range(0,B,block):
        hi=min(B,lo+block)
        path=block_file(out,exp,rep,tr,lo,hi)
        if not path.exists(): raise RuntimeError('INCOMPLETE BLOCK: '+str(path))
        with np.load(path,allow_pickle=False) as z:
            if not np.array_equal(z['replicate'],np.arange(lo,hi)):raise ValueError('Block replicate mismatch')
            names=('ami_vi',) if exp=='tn1' else ('ami_en','ami_vi','cross','within_en','within_vi')
            if not parts:
                shapes={'ami_en':(B,10),'ami_vi':(B,10),'cross':(B,10,10),'within_en':(B,45),'within_vi':(B,45)}
                parts={name:np.empty(shapes[name]) for name in names}
            for name in names:
                parts[name][lo:hi]=z[name]  # retain any undefined values for explicit reporting
            seen[lo:hi]+=1
    if not np.all(seen==1):raise ValueError('Replicate coverage is not exactly once')
    return parts


def ci(a,alpha):
    return np.quantile(a,[alpha/2,1-alpha/2]).tolist()


def percentile_p(a,null=0,side='two'):
    """CI inversion via empirical percentile tails; NOT null-calibrated test.
    Boundary decisions may differ at B=1000 rank; recorded 0 means <1/B resolution, not zero probability.
    """
    a=np.asarray(a)
    if side=='greater': return float(np.mean(a<=null))
    if side=='less': return float(np.mean(a>=null))
    if side=='two': return float(min(1.,2*min(np.mean(a<=null),np.mean(a>=null))))
    raise ValueError(side)


def tost_p(a,epsilon=.01):
    return max(percentile_p(a,-epsilon,'greater'),percentile_p(a,epsilon,'less'))


def tn2b_weighted_seed_sensitivity(d,we,wv):
    """TN2-B secondary metric, independent EN and VI multinomial seed weights.

    Cross pairs use outer product of EN and VI weights. Within-language
    comparisons only use the 45 distinct original seed pairs. Unavailable
    within information becomes NaN, never an artificial self-similarity of 1.
    """
    jj=np.array([(i,j) for i in range(10) for j in range(i+1,10)])
    we=np.asarray(we,dtype=np.float64)
    wv=np.asarray(wv,dtype=np.float64)
    cross=np.einsum('bij,bi,bj->b',d['cross'],we,wv)/100
    pe=we[:,jj[:,0]]*we[:,jj[:,1]]
    pv=wv[:,jj[:,0]]*wv[:,jj[:,1]]
    de=pe.sum(axis=1); dv=pv.sum(axis=1)
    within=np.full(len(we),np.nan)
    mask=(de>0)&(dv>0)
    within[mask]=(.5*np.sum(d['within_en'][mask]*pe[mask],axis=1)/de[mask]
                  +.5*np.sum(d['within_vi'][mask]*pv[mask],axis=1)/dv[mask])
    return cross-within


def holm(vals):
    vals=np.asarray(vals,dtype=float)
    if not np.isfinite(vals).all() or np.any((vals<0)|(vals>1)):raise ValueError('Invalid p')
    ix=np.argsort(vals,kind='stable'); adj=np.empty(len(ix),dtype=float)
    step=np.maximum.accumulate(np.minimum(1,(len(ix)-np.arange(len(ix)))*vals[ix]))
    adj[ix]=step
    return adj


def summarize(out,inp,weights,args):
    """All full estimates from primary app CI, secondary seed reweighting only."""
    tn1={(r,t):load_blocks(out,'tn1',r,t,args.tn1_block)['ami_vi'] for r in REPS1 for t in TR1}
    tn2={(r,t):load_blocks(out,'tn2',r,t,args.tn2_block) for r in REPS2 for t in TR2}
    rows=[]; vectors={}; unweighted={}; sensitive={}
    for r in REPS1:
        x=tn1[(r,'native')]-tn1[(r,'char_budget')]
        vectors[('TN1-A',r,'native_minus_char')]=x
    for t in TR1:
        for r in REPS1:
            if r=='tfidf_lsa':continue
            vectors[('TN1-B',r,t)]=tn1[(r,t)]-tn1[('tfidf_lsa',t)]
        for a,b in ((LEADS[0],LEADS[1]),(LEADS[0],LEADS[2]),(LEADS[1],LEADS[2])):
            vectors[('TOP3',a+'_minus_'+b,t)]=tn1[(a,t)]-tn1[(b,t)]
    sample=np.load(out/'indices_tn2.npy',mmap_mode='r')
    try:
        for r in REPS2:
            for t in TR2:
                d=tn2[(r,t)]
                vectors[('TN2-C',r,t)]=d['ami_en']-d['ami_vi']
                unweighted[('TN2-B',r,t)]=d['cross'].mean(axis=(1,2))-(d['within_en'].mean(axis=1)+d['within_vi'].mean(axis=1))/2
                # Primary app-only TN2-A (existing per-app data, no pair resampling).
                a=inp.delta[(r,t)]
                unweighted[('TN2-A',r,t)]=np.array([a[ii].mean() for ii in sample])
    finally:
        sample._mmap.close()  # allow TemporaryDirectory deletion on Windows
    for k,v in vectors.items():
        unweighted[k]=v.mean(axis=1)
        fam,r,t=k
        if fam=='TN1-A':
            wn=weights[('tn1',r,'native','vi')]
            wc=weights[('tn1',r,'char_budget','vi')]
            sensitive[k]=((tn1[(r,'native')]*wn).sum(axis=1)
                          -(tn1[(r,'char_budget')]*wc).sum(axis=1))/10
        elif fam=='TN1-B':
            wa=weights[('tn1',r,t,'vi')]
            wb=weights[('tn1','tfidf_lsa',t,'vi')]
            sensitive[k]=((tn1[(r,t)]*wa).sum(axis=1)
                          -(tn1[('tfidf_lsa',t)]*wb).sum(axis=1))/10
        elif fam=='TOP3':
            a,b=r.split('_minus_')
            wa=weights[('tn1',a,t,'vi')]
            wb=weights[('tn1',b,t,'vi')]
            sensitive[k]=((tn1[(a,t)]*wa).sum(axis=1)
                          -(tn1[(b,t)]*wb).sum(axis=1))/10
        elif fam=='TN2-C':
            d=tn2[(r,t)]
            we=weights[('tn2',r,t,'en')]
            wv=weights[('tn2',r,t,'vi')]
            sensitive[k]=((d['ami_en']*we).sum(axis=1)
                          -(d['ami_vi']*wv).sum(axis=1))/10
        else:raise ValueError('Unhandled seedwise family '+fam)
    # Independent EN/VI weights are applied inside the TN2-B helper.
    for r in REPS2:
        for t in TR2:
            k=('TN2-B',r,t);d=tn2[(r,t)]
            sensitive[k]=tn2b_weighted_seed_sensitivity(
                d,weights[('tn2',r,t,'en')],weights[('tn2',r,t,'vi')])
    # full-data point estimates derived from Gate 4 metrics, after pilot verification.
    df=inp.df
    def full_ami(exp,rep,tr,lang,seed):
        section='TN1_EXTERNAL' if exp=='tn1' else 'TN2_C'
        return get_expected(df,section=section,representation=rep,truncation=tr,
                            language=lang,algorithm='kmeans_main',run='main',seed=float(seed),metric='ami')
    def gmean(exp,rep,tr,lang):return np.mean([full_ami(exp,rep,tr,lang,s) for s in SEEDS])
    def point(k):
        fam,r,t=k
        if fam=='TN1-A':return gmean('tn1',r,'native','vi')-gmean('tn1',r,'char_budget','vi')
        if fam=='TN1-B':return gmean('tn1',r,t,'vi')-gmean('tn1','tfidf_lsa',t,'vi')
        if fam=='TOP3':
            a,b=r.split('_minus_');return gmean('tn1',a,t,'vi')-gmean('tn1',b,t,'vi')
        if fam=='TN2-C':return gmean('tn2',r,t,'en')-gmean('tn2',r,t,'vi')
        if fam=='TN2-A':return get_expected(df,section='TN2_A',representation=r,truncation=t,
                         language='en_vs_vi',algorithm='embedding_pair',run='paired_apps',
                         metric='delta_cosine',pair_role='mean')
        if fam=='TN2-B':
            vals=[]
            for lang,run in [('en_vs_vi','cross_language_all_pairs'),('en','within_language_all_pairs'),('vi','within_language_all_pairs')]:
                q=df[(df.section=='TN2_B')&(df.representation==r)&(df.truncation==t)&(df.language==lang)
                     &(df.algorithm=='kmeans_main')&(df.run==run)&(df.metric=='ami')]['value'].to_numpy()
                if len(q)!=(100 if lang=='en_vs_vi' else 45):raise ValueError('TN2B aggregate incomplete')
                vals.append(q.mean())
            return vals[0]-(vals[1]+vals[2])/2
        raise ValueError(k)
    for k,arr in unweighted.items():
        fam,r,t=k
        valid=np.isfinite(arr)
        sens=sensitive.get(k)
        sens_valid=np.isfinite(sens) if sens is not None else np.zeros(B,bool)
        if sens is not None and sens_valid.sum()<B:
            # Never silently discard undefined sensitivity replicates.
            print('WARNING: sensitivity undefined replicates:',k,int(B-sens_valid.sum()),flush=True)
        eq=fam in ('TN1-A','TN2-C','TOP3')
        # Undefined primary replicate invalidates this ENTIRE inferential comparison:
        # do NOT cherry-pick finite draws and present a nominal 1000-replicate CI.
        q95=ci(arr,.05) if valid.all() else [np.nan,np.nan]
        q90=ci(arr,.10) if valid.all() else [np.nan,np.nan]
        p=(percentile_p(arr,0,'greater' if fam=='TN2-A' else 'two')
           if valid.all() and fam in ('TN1-A','TN2-A','TN2-B','TN2-C') else np.nan)
        e=tost_p(arr,.01) if valid.all() and eq else np.nan
        row={'family':fam,'comparison':r,'regime':t,'point_estimate':point(k),
             'boot_mean':float(arr.mean()) if valid.all() else np.nan,'ci95_lo':q95[0],'ci95_hi':q95[1],
             'ci90_lo':q90[0],'ci90_hi':q90[1],
             'p_percentile':p,'tost_p_001':e,
             'tost_90ci_inside_001':bool(q90[0]>-.01 and q90[1]<.01) if eq and valid.all() else None,
            'tost_p_0005_sensitivity':tost_p(arr,.005) if eq and valid.all() else np.nan,
            'tost_90ci_inside_0005_sensitivity':bool(q90[0]>-.005 and q90[1]<.005) if eq and valid.all() else None,
             'role':'exploratory' if fam in ('TN1-B','TOP3') else 'primary',
             'replicates_valid':int(valid.sum()),
             'replicates_undefined':int(B-valid.sum()),
             'status':'VALID' if valid.all() else 'UNDEFINED_REPLICATE',
             'sensitivity_status':'UNDEFINED_REPLICATE' if sens is not None and not sens_valid.all() else 'VALID',
             'sensitivity_ci95_lo':float(ci(sens[sens_valid],.05)[0]) if sens is not None and sens_valid.any() else np.nan,
             'sensitivity_ci95_hi':float(ci(sens[sens_valid],.05)[1]) if sens is not None and sens_valid.any() else np.nan,
             'sensitivity_ci90_lo':float(ci(sens[sens_valid],.10)[0]) if sens is not None and sens_valid.any() else np.nan,
             'sensitivity_ci90_hi':float(ci(sens[sens_valid],.10)[1]) if sens is not None and sens_valid.any() else np.nan,
             'sensitivity_tost_p_001':tost_p(sens[sens_valid],.01) if sens is not None and sens_valid.any() and eq else np.nan,
             'sensitivity_valid':int(sens_valid.sum()) if sens is not None else 0,
             'sensitivity_undefined':int(B-sens_valid.sum()) if sens is not None else 0}
        rows.append(row)
    summary=pd.DataFrame(rows).sort_values(['family','regime','comparison']).reset_index(drop=True)
    # Auxiliary Wilcoxon across ten fixed KMeans seeds. Not 10 independent datasets.
    from scipy.stats import wilcoxon
    summary['wilcoxon_seed10_aux_p']=np.nan
    for idx,row in summary[summary.family=='TN2-C'].iterrows():
        r,t=row['comparison'],row['regime']
        seq=np.array([full_ami('tn2',r,t,'en',s)-full_ami('tn2',r,t,'vi',s) for s in SEEDS])
        if np.allclose(seq,0,atol=0,rtol=0):
            summary.loc[idx,'wilcoxon_seed10_aux_p']=1.0
        else:
            summary.loc[idx,'wilcoxon_seed10_aux_p']=float(wilcoxon(seq,alternative='two-sided',
                                                                    zero_method='wilcox',method='auto').pvalue)
    for col, family in [('p_percentile','TN1-A'),('p_percentile','TN2-A'),('p_percentile','TN2-B'),('p_percentile','TN2-C'),
                        ('tost_p_001','TN1-A'),('tost_p_001','TN2-C')]:
        mask=summary.family==family
        field='holm_difference_p' if col=='p_percentile' else 'holm_tost_p'
        fam_p=summary.loc[mask,col].to_numpy()
        # Fail closed: if one member is undefined, no family-wise inference.
        summary.loc[mask,field]=holm(fam_p) if np.isfinite(fam_p).all() else np.nan
    summary['holm_difference_reject_005']=summary['holm_difference_p']<.05
    summary['holm_equivalence_reject_005']=summary['holm_tost_p']<.05
    summary.loc[~summary.family.isin(['TN1-A','TN2-C']),'holm_equivalence_reject_005']=False
    # Long table: all KMeans main AMI by configuration × seed × replicate.
    long=[]
    for (r,t),v in tn1.items():
        for i,s in enumerate(SEEDS):
            long.append(pd.DataFrame({'experiment':'tn1','representation':r,'truncation':t,
                         'language':'vi','seed':s,'replicate':np.arange(B),'ami':v[:,i]}))
    for (r,t),v in tn2.items():
        for lang,mat in [('en',v['ami_en']),('vi',v['ami_vi'])]:
            for i,s in enumerate(SEEDS):
                long.append(pd.DataFrame({'experiment':'tn2','representation':r,'truncation':t,
                         'language':lang,'seed':s,'replicate':np.arange(B),'ami':mat[:,i]}))
    seedwise=pd.concat(long,ignore_index=True)
    assert len(seedwise)==320000
    # Replicate-level paired differences retained (8*4+14+6=52).
    replicates=pd.concat([pd.DataFrame({'family':k[0],'comparison':k[1],'regime':k[2],
                            'replicate':np.arange(B),'estimate':v}) for k,v in unweighted.items()],ignore_index=True)
    assert len(replicates)==52*B
    return summary,seedwise,replicates


def audit_info(inp,am,script):
    return {'project':'PAAC-VN','phase':'Gate5','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'method':'app paired bootstrap on fixed partitions, AMI sklearn arithmetic; percentile CI inversion p',
        'inference_plan':'fixed before uncertainty estimation; NOT preregistered before point estimates',
        'n_bootstrap':B,'seed':42,'seeds':list(SEEDS),'amendment':am,'script_sha256':script,
        'source_sha256':inp.files,'source_embedded_fingerprints':{k:dict(sha256_npy_current=v['sha256_npy_current']) for k,v in inp.e['sources'].items()},
        'versions':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn_version,'platform':platform.platform()},
        'familywise':'Holm per 8-member primary difference family; TOST separately per TN1-A / TN2-C',
        'SESOI':{'AMI_primary':0.01,'sensitivity_illustrative':0.005},
        'limitations':['bootstrap samples apps with replacement, holds partitions fixed',
                       'AMI for INDIVIDUAL configurations shows upward app-duplicate bias, previously observed about +0.013 TN1 and +0.023 TN2; user-reported descriptive diagnostics, not independently re-measured here',
                       'paired differences previously showed about <=0.002 residual bootstrap bias; user-reported descriptive diagnostic, not a universal guarantee',
                       'do NOT publish or interpret per-configuration AMI confidence intervals; inferential output restricted to paired contrasts',
                       'seed reweighting is sensitivity heuristic, not independent datasets',
                       'seed sensitivity weights independent by configuration and language, paired app indices shared',
                       'percentile CI inversion gives approximate empirical p, not null-calibrated',
                       'top3 chosen after viewing Gate4 point estimates: exploratory',
                       'cross and within seed pair observations are dependent']}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,default=Path('.'))
    p.add_argument('--mode',required=True,choices=('pilot','prepare','full','finalize'))
    p.add_argument('--configs',nargs='+',default=None,
                   help='Only for --mode full. One or more IDs, e.g. tn1__bge_m3__char_budget '
                        'tn2__bge_m3__native; comma separated is also accepted. Default: all.')
    p.add_argument('--tn1-block',type=int,default=50)
    p.add_argument('--tn2-block',type=int,default=10)
    args=p.parse_args()
    if args.tn1_block<1 or args.tn2_block<1:raise ValueError('block sizes must be positive')
    if args.configs is not None and args.mode!='full':
        raise ValueError('--configs is only meaningful with --mode full')
    configs=parse_configs(args.configs)
    repo=args.repo.resolve();out=repo/'results/statistics/gate5'
    am=amendment(repo)
    script=sha(Path(__file__))
    if args.mode=='pilot':
        out.mkdir(parents=True,exist_ok=True)
        try:
            inp=Inputs(repo)
            check=inp.point_checks()
        except (ValueError, RuntimeError, FileNotFoundError, KeyError) as exc:
            report={'phase':'Gate5_PILOT','created_at_utc':datetime.now(timezone.utc).isoformat(),
                    'gate5_status':'SOURCE_DISCREPANCY',
                    'check':{'status':'SOURCE_DISCREPANCY','error_type':type(exc).__name__,
                             'detail':str(exc)},
                    'script_sha256':script,'amendment':am}
            write_json_new(out/'pilot.json',report)
            raise RuntimeError('Pilot SOURCE_DISCREPANCY; report written to pilot.json') from exc
        report={'phase':'Gate5_PILOT','created_at_utc':datetime.now(timezone.utc).isoformat(),
                'amendment':am,'script_sha256':script,'source_sha256':inp.files,
                'check':check,'gate5_status':'PILOT_PASS' if check['status']=='PASS' else 'SOURCE_DISCREPANCY'}
        write_json_new(out/'pilot.json',report)
        print(json.dumps(report,indent=2))
        if check['status']!='PASS':raise RuntimeError('SOURCE_DISCREPANCY: full blocked')
        return
    inp=Inputs(repo)
    pilot_path=out/'pilot.json'
    if not pilot_path.exists():raise RuntimeError('Pilot required first: --mode pilot')
    pilot=rd(pilot_path)
    if not (pilot['gate5_status']=='PILOT_PASS' and pilot['amendment']==am
            and pilot['script_sha256']==script and pilot['source_sha256']==inp.files):
        raise RuntimeError('Pilot invalid: source/script/amendment changed or not PASS')
    def expected_spec():
        return {'B':B,'seed':42,'tn1_block':args.tn1_block,'tn2_block':args.tn2_block,
                **rng_fingerprints(out),'script_sha256':script,
                'amendment':am,'source_sha256':inp.files}
    manifest_path=out/'run_config.json'
    if args.mode=='prepare':
        # One-time serial action: processes using --configs MUST NOT initialize
        # random sampling matrices, preventing race and order-dependent RNG.
        if manifest_path.exists():
            spec=rd(manifest_path)
            if spec!=expected_spec():raise RuntimeError('Existing run_config/RNG mismatch')
            print('PREPARE VERIFIED (immutable inputs unchanged)')
        else:
            create_rng_files(out)
            spec=expected_spec()
            indices,weights=load_rng(out,{k:spec[k] for k in ('indices_sha256','seed_weights_sha256','rng_derivation')})
            close_mmaps(indices,weights)
            write_json_new(manifest_path,spec)
            print('PREPARE COMPLETE — shared indices/weights and run_config frozen')
        return
    if not manifest_path.exists():
        raise RuntimeError('Run --mode prepare after a PASS pilot, BEFORE --mode full/finalize')
    spec=rd(manifest_path)
    if (spec['script_sha256']!=script or spec['amendment']!=am or spec['source_sha256']!=inp.files
            or spec['B']!=B or spec['seed']!=42 or spec['rng_derivation']!=rng_description()):
        raise RuntimeError('Sources, script, or RNG plan differ from prepared run_config')
    if (args.tn1_block,args.tn2_block)!=(spec['tn1_block'],spec['tn2_block']) and args.mode=='full':
        raise RuntimeError('Block sizes must match prepared run_config')
    args.tn1_block=spec['tn1_block'];args.tn2_block=spec['tn2_block']
    rng_expected={k:spec[k] for k in ('indices_sha256','seed_weights_sha256','rng_derivation')}
    if args.mode=='full':
        indices,weights=load_rng(out,rng_expected)
        try:
            run_blocks(out,inp,indices,args,configs=configs)
        finally:
            close_mmaps(indices,weights)
        print('SELECTED CONFIG BLOCKS GENERATED. Finalize ONLY after ALL configs complete.')
        return
    # --mode finalize refuses to infer missing configurations from --configs.
    # Validate every shared RNG file against run_config, then every expected
    # checkpoint via load_blocks before any finished output is published.
    indices,weights=load_rng(out,rng_expected)
    close_mmaps(indices)
    if any((out/f).exists() for f in ('bootstrap_summary.csv','bootstrap_seedwise_ami.parquet','bootstrap_replicates.parquet','statistical_audit.json','gate5_manifest.json')):
        raise FileExistsError('Gate5 output already exists; refusing overwrite')
    try:
        summary,seedwise,replicates=summarize(out,inp,weights,args)
    finally:
        close_mmaps(weights)
    # Stage all outputs to temporary names, publish after every numerical check.
    summary.to_csv(out/'bootstrap_summary.csv.partial',index=False)
    seedwise.to_parquet(out/'bootstrap_seedwise_ami.parquet.partial',index=False)
    replicates.to_parquet(out/'bootstrap_replicates.parquet.partial',index=False)
    aud=audit_info(inp,am,script)
    aud['status']='COMPLETE_WITH_LIMITATIONS';aud['hypotheses_n']=len(summary)
    aud['undefined_sensitivity_count']=int(summary.sensitivity_undefined.sum())
    aud['undefined_primary_count']=int(summary.replicates_undefined.sum())
    aud['status_counts']=summary.status.value_counts().to_dict()
    if aud['undefined_primary_count']:
        aud['status']='INCOMPLETE_UNDEFINED_REPLICATE'
    aud['pilot_check']=pilot['check'];aud['point_estimate_validation']='PILOT_PASS'
    write_json_new(out/'statistical_audit.json.partial',aud)
    for name in ('bootstrap_summary.csv','bootstrap_seedwise_ami.parquet','bootstrap_replicates.parquet','statistical_audit.json'):
        os.replace(out/(name+'.partial'),out/name)
    manifest={**aud,'generated_sha256':{name:sha(out/name) for name in (
        'bootstrap_summary.csv','bootstrap_seedwise_ami.parquet','bootstrap_replicates.parquet','statistical_audit.json')},
        'bootstrap_index_sha256':spec['indices_sha256'],
        'seed_weights_sha256':spec['seed_weights_sha256'],
        'rng_derivation':spec['rng_derivation'],
        'checkpoint_block_sha256':{p.name:sha(p) for p in sorted((out/'blocks').glob('*.npz'))}}
    write_json_new(out/'gate5_manifest.json',manifest)
    print('Gate5 finalized: ',out,'n comparisons',len(summary))

if __name__=='__main__': main()
