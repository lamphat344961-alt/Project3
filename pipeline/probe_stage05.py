"""Frozen LaBSE/E5 logistic probe using the approved prepared split."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from threadpoolctl import threadpool_limits
from pipeline import embed
from pipeline.evaluate_stage05 import ROOT, load_validated, load_validated_e5_tn1, load_validated_e5_tn2, VALIDATION, E5_VALIDATION, E5_TN1_VALIDATION

C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def fit_select(x, y, split, class_labels):
    """Only validation labels choose C; test is first used after final fit."""
    train = split == 'train'
    validation = split == 'validation'
    test = split == 'test'
    if not (train.any() and validation.any() and test.any()):
        raise ValueError('All three splits must be nonempty')
    scores = []
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        for c in C_GRID:
            model = LogisticRegression(C=c, solver='lbfgs', max_iter=2000, random_state=42)
            model.fit(x[train], y[train])
            predicted = model.predict(x[validation])
            scores.append(dict(C=c, validation_macro_f1=float(f1_score(y[validation],predicted,labels=class_labels,average='macro',zero_division=0))))
        # Ordered grid + stable max => smaller C wins exact validation ties.
        chosen = max(scores,key=lambda r:r['validation_macro_f1'])['C']
        final = LogisticRegression(C=chosen, solver='lbfgs', max_iter=2000, random_state=42)
        final.fit(x[~test], y[~test])
        pred = final.predict(x[test])
    labels, counts = np.unique(y[~test], return_counts=True)
    majority = np.repeat(labels[np.argmax(counts)], test.sum())
    return final, pred, dict(selected_C=chosen, tuning=scores,
        test_macro_f1=float(f1_score(y[test],pred,labels=class_labels,average='macro',zero_division=0)),
        test_accuracy=float(accuracy_score(y[test],pred)),
        majority_test_macro_f1=float(f1_score(y[test],majority,labels=class_labels,average='macro',zero_division=0)),
        majority_test_accuracy=float(accuracy_score(y[test],majority)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    parser.add_argument('--model',choices=['labse','e5'],default='labse')
    parser.add_argument('--split-proposal', type=Path, required=True)
    parser.add_argument('--approved-by', required=True)
    parser.add_argument('--approval-reference', required=True)
    args = parser.parse_args()
    if not args.approved_by.strip() or not args.approval_reference.strip():
        raise ValueError('Explicit author approval required')
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):
        raise ValueError('Invalid run ID')
    proposal = json.loads(args.split_proposal.read_text(encoding='utf-8'))
    if proposal['status'] != 'PROPOSED_NOT_AUTHORIZED_FOR_PROBE' or proposal['proposed_probe']['C_grid'] != list(C_GRID):
        raise ValueError('Unsupported split/tuning contract')
    if not all(s['all_49_classes_present_in_each_split'] for s in proposal['class_support']):
        raise ValueError('Missing class support')
    for name, digest in proposal['artifacts_sha256'].items():
        if embed.sha256(args.split_proposal.parent/name) != digest:
            raise ValueError('Split artifact hash mismatch: '+name)
    folds = pd.read_parquet(args.split_proposal.parent/'proposed_app_folds.parquet')
    if folds.appId.duplicated().any() or folds.groupby('component').split.nunique().max() != 1 or folds.groupby('developerId').split.nunique().max() != 1:
        raise ValueError('Split leaks developer/component')
    if args.model=='labse':
        inputs, frames = load_validated()
        input_validation = [VALIDATION]
    else:
        inputs,frames = load_validated_e5_tn1()
        paired,_ = load_validated_e5_tn2()
        # Approved probe scope: TN1 VI char/native; TN2 EN/VI native.
        # No additional T512 classifier runs or inferred probe artifacts.
        inputs.update({k:v for k,v in paired.items() if k.endswith(':native')})
        input_validation = [E5_TN1_VALIDATION,E5_VALIDATION]
    for exp, expected in proposal['dataset_sha256'].items():
        from pipeline.labse_stage03 import load_contract
        if embed.sha256(ROOT/load_contract()['datasets'][exp]['path']) != expected:
            raise ValueError('Split dataset hash mismatch')
    out = ROOT/'results/stage05'/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    receipt = dict(status='RUNNING', approved_by=args.approved_by, approval_reference=args.approval_reference,
        split_proposal_sha256=embed.sha256(args.split_proposal), code_sha256=embed.sha256(Path(__file__)),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), packages=embed.package_versions(),
        model=args.model,input_validation_sha256={p.relative_to(ROOT).as_posix():embed.sha256(p) for p in input_validation},
        scope='Four frozen '+args.model+' cells: TN1 VI char/native, TN2 EN/VI native; majority baseline; no T512 probe, lexical baseline, PCA sensitivity or learning curve implied')
    summaries = []
    try:
        classes = sorted(frames['tn1'].genreId.unique())
        for cell, x in inputs.items():
            exp = cell.split(':')[0]
            frame = frames[exp]
            aligned = frame[['appId','genreId']].merge(folds[['appId','split']],on='appId',how='left',validate='one_to_one')
            if aligned.split.isna().any() or aligned.appId.tolist() != frame.appId.tolist():
                raise ValueError('Split alignment error')
            y, split = frame.genreId.to_numpy(), aligned.split.to_numpy()
            print('PROBE '+cell,flush=True)
            with threadpool_limits(limits=4):
                model, predictions, summary = fit_select(x,y,split,classes)
            summary['cell'] = cell
            summaries.append(summary)
            base = out/cell.replace(':','__')
            pd.DataFrame({'appId':frame.appId[split=='test'],'genreId':y[split=='test'],'prediction':predictions}).to_parquet(base.with_suffix('.test_predictions.parquet'),index=False)
            precision, recall, f1, support = precision_recall_fscore_support(y[split=='test'],predictions,labels=classes,zero_division=0)
            pd.DataFrame({'genreId':classes,'precision':precision,'recall':recall,'f1':f1,'support':support}).to_csv(base.with_suffix('.per_class.csv'),index=False)
            pd.DataFrame(confusion_matrix(y[split=='test'],predictions,labels=classes),index=classes,columns=classes).to_csv(base.with_suffix('.confusion.csv'))
            with base.with_suffix('.model.npz').open('xb') as handle:
                np.savez_compressed(handle,coef=model.coef_,intercept=model.intercept_,classes=model.classes_)
            embed.write_json_create_only(base.with_suffix('.json'),summary)
        receipt.update(status='FOUR_'+args.model.upper()+'_PROBES_COMPLETE',cells=summaries)
    except Exception as exc:
        receipt.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',cells=summaries)
        raise
    finally:
        receipt['artifacts_sha256']={p.name:embed.sha256(p) for p in out.iterdir() if p.is_file()}
        embed.write_json_create_only(out/'execution_receipt.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()
