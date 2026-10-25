"""Prepare a reviewable group-safe probe split proposal. Never trains a probe."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from pipeline import embed, make_paired
from pipeline.labse_stage03 import load_contract, load_inputs

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    args = parser.parse_args()
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):
        raise ValueError('Invalid run ID')
    out = ROOT / 'results/stage05' / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    frames = load_inputs(load_contract())
    df = frames['tn1'].sort_values('appId', kind='stable').reset_index(drop=True)
    parent = np.arange(len(df))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a,b)] = min(a,b)

    def merge_values(values):
        seen = {}
        for i, value in enumerate(values):
            if pd.isna(value) or not str(value).strip():
                continue
            key = str(value)
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i

    if df.developerId.isna().any() or df.developerId.astype(str).str.strip().eq('').any():
        raise ValueError('Missing developer ID: cannot silently treat as developer singleton')
    merge_values(df.developerId)
    merge_values(df.dup_group_id)
    print('Grouping exact texts across both views', flush=True)
    seen_texts = {}
    for field in ('description_clean', 'description_original_clean'):
        for i, value in enumerate(df[field]):
            if not isinstance(value, str) or not value.strip():
                continue
            key = value.strip()
            if key in seen_texts:
                union(i, seen_texts[key])
            else:
                seen_texts[key] = i
    # Use the existing deterministic candidate algorithm and frozen threshold,
    # including EN outside TN2, rather than assuming TN2 dedup covers TN1.
    component_sizes = {}
    for field in ('description_clean', 'description_original_clean'):
        print('Near-duplicate candidates: ' + field, flush=True)
        roots = make_paired.near_dup_groups(df[field].fillna('').astype(str).tolist())
        for i, root in enumerate(roots):
            union(i, int(root))
        _, counts = np.unique(roots, return_counts=True)
        component_sizes[field] = dict(n_components=len(counts), non_singleton_components=int((counts>1).sum()), max_component=int(counts.max()))
    groups = np.array([find(i) for i in range(len(df))])
    folds = np.full(len(df), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (_, test) in enumerate(splitter.split(np.zeros(len(df)), df.genreId, groups)):
        folds[test] = fold
    assignment = np.where(folds == 0, 'test', np.where(folds == 1, 'validation', 'train'))
    table = df[['appId','genreId','developerId']].copy()
    table['component'], table['fold'], table['split'] = groups, folds, assignment
    assert table.groupby('developerId').split.nunique().max() == 1
    assert table.groupby('component').split.nunique().max() == 1
    assert set(frames['tn2'].appId).issubset(set(table.appId))
    paired = frames['tn2'][['appId','genreId']].merge(table[['appId','split','component','fold']], on='appId', how='left', validate='one_to_one')
    support = []
    for cohort, data in [('tn1',table),('tn2',paired)]:
        counts = pd.crosstab(data.genreId, data.split).reindex(columns=['train','validation','test'],fill_value=0)
        counts.to_csv(out / (cohort + '_class_support.csv'))
        support.append(dict(cohort=cohort, split_counts=data.split.value_counts().to_dict(),
            n_genres=len(counts), all_49_classes_present_in_each_split=bool(len(counts)==49 and (counts>0).all().all()),
            minimum_class_support={c:int(counts[c].min()) for c in counts}))
    table.to_parquet(out / 'proposed_app_folds.parquet', index=False)
    paired.to_parquet(out / 'proposed_tn2_folds.parquet', index=False)
    _, group_counts = np.unique(groups,return_counts=True)
    receipt = dict(status='PROPOSED_NOT_AUTHORIZED_FOR_PROBE', created_at_utc=datetime.now(timezone.utc).isoformat(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        code_sha256={p:embed.sha256(ROOT/p) for p in ['pipeline/prepare_stage05_probe.py','pipeline/make_paired.py']},
        dataset_sha256={k:embed.sha256(ROOT/v['path']) for k,v in load_contract()['datasets'].items()},
        n_apps=len(df), developer_missing=0, n_developers=int(df.developerId.nunique()),
        n_components=len(group_counts), max_component_apps=int(group_counts.max()), near_duplicate_components=component_sizes,
        grouping='Union of developerId, historical VI dup IDs, exact cleaned texts across EN/VI, and per-view MinHash-LSH components on all TN1 apps; TN2 inherits by app ID',
        near_duplicate_method={'threshold':make_paired.NEAR_DUP_JACCARD,'num_perm':make_paired.NUM_PERM,'seed':make_paired.SEED,'shingle':make_paired.SHINGLE,'scope':'MinHash-estimated Jaccard among LSH candidates; not exhaustive exact all-pairs'},
        proposed_split='StratifiedGroupKFold(5,shuffle=True,random_state=42); fold0 test, fold1 validation, folds2-4 train; approximate 60/20/20 before inspecting probe scores',
        class_support=support,
        proposed_probe={'primary_cohort':'TN1 VI both approved input conditions','estimator':'L2 LogisticRegression','solver':'lbfgs','max_iter':2000,'C_grid':[0.01,0.1,1.0,10.0,100.0],'selection':'validation Macro-F1, ties choose smaller C; refit train+validation then evaluate test once','preprocessing':'Frozen L2 embeddings; no scaler/PCA; all learned preprocessing for lexical baselines train-only','secondary_cohort':'TN2 EN/VI native; same app folds, separate view models','primary_metric':'Macro-F1 over fixed 49 labels; accuracy/per-class F1/confusion secondary'},
        limitations=['Candidate-based near-duplicate detection cannot prove absence of all semantic duplicates.','Split ratios and tuning contract require author approval; no classifier fit or test scores generated.','Lexical train-only baseline and learning curve/PCA sensitivities are not implemented by this preparation script.'])
    receipt['artifacts_sha256']={p.name:embed.sha256(p) for p in out.iterdir() if p.is_file()}
    embed.write_json_create_only(out/'split_proposal.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()
