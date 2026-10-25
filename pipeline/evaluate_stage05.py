"""Stage05 evaluation of validated LaBSE outputs, without embedding inference."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from itertools import combinations, product
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from threadpoolctl import threadpool_limits

from pipeline import cluster, embed, evaluate_reviewed as metrics, g1c_retrieval
from pipeline.labse_stage03 import load_contract, load_inputs

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / 'results/labse_stage03/validation_local_001/validation_receipt.json'
INPUT_PREFIX = 'labse_local_20261024_204813_'


def write_json(path, value):
    embed.write_json_create_only(path, value)


def agreement_rows(en, vi):
    rows = []
    for role, pairs in (
        ('C', product(range(10), repeat=2)),
        ('W_EN', combinations(range(10), 2)),
        ('W_VI', combinations(range(10), 2)),
    ):
        for i, j in pairs:
            a, b = (en[i], vi[j]) if role == 'C' else ((en[i], en[j]) if role == 'W_EN' else (vi[i], vi[j]))
            rows.append(dict(component=role, seed=cluster.SEEDS[i], seed_other=cluster.SEEDS[j], **metrics.partition_similarity(a, b)))
    return rows


def summarize_agreement(rows):
    means = {role: float(np.mean([r['ami'] for r in rows if r['component'] == role])) for role in ('C', 'W_EN', 'W_VI')}
    return dict(**means, S=means['C'] - (means['W_EN'] + means['W_VI']) / 2,
                cross_pairs=100, within_pairs_per_language=45,
                interpretation='Descriptive seed-relative EN-VI partition gap, not a language/translation causal effect')


def provenance_labels(frame):
    # Source flags describe collection provenance, not verified human authorship.
    return frame.text_source.map({'machine': 'translation-flagged', 'human': 'not-flagged'}).fillna('unknown')


def load_validated():
    validation = json.loads(VALIDATION.read_text(encoding='utf-8'))
    if validation['status'] != 'FOUR_CELLS_LOCAL_VALIDATION_PASS':
        raise ValueError('LaBSE validation is not PASS')
    contract = load_contract()
    frames = load_inputs(contract)
    inputs = {}
    for item in validation['validated_cells']:
        cell = item['cell']
        exp, lang, condition = cell.split(':')
        folder = ROOT / 'results/labse_stage03' / (INPUT_PREFIX + cell.replace(':', '_'))
        if embed.sha256(folder / 'execution_receipt.json') != item['execution_receipt_sha256']:
            raise ValueError('Input receipt changed: ' + cell)
        for name, digest in item['artifacts_sha256'].items():
            if embed.sha256(folder / name) != digest:
                raise ValueError('Input artifact changed: ' + name)
        base = folder / '__'.join([exp, 'labse', lang, condition])
        x = np.load(base.with_suffix('.npy'), mmap_mode='r', allow_pickle=False)
        rows = pd.read_parquet(base.with_suffix('.rows.parquet'))
        if rows.appId.tolist() != frames[exp].appId.tolist() or rows.genreId.tolist() != frames[exp].genreId.tolist():
            raise ValueError('Input mapping changed: ' + cell)
        inputs[cell] = x
    expected = {':'.join(c) for c in contract['primary_cells_proposed']}
    if set(inputs) != expected:
        raise ValueError('Four-cell scope mismatch')
    return inputs, frames


def run(out):
    inputs, frames = load_validated()
    labels, alignment = {}, []
    for cell, x in inputs.items():
        print('CLUSTER ' + cell, flush=True)
        pred, inertia, n_iter = cluster.run_kmeans_grid(x, cluster.MAIN_N_INIT)
        with (out / (cell.replace(':', '__') + '__kmeans_main.npz')).open('xb') as handle:
            np.savez_compressed(handle, seeds=cluster.SEEDS, labels=pred, inertia=inertia,
                                n_iter=n_iter, n_init=10, k=49)
        labels[cell] = pred
        exp = cell.split(':')[0]
        y = pd.Categorical(frames[exp].genreId.astype(str)).codes
        for seed, partition in zip(cluster.SEEDS, pred):
            alignment.append(dict(cell=cell, seed=seed, n_apps=len(y), **metrics.external_scores(y, partition)))
    pd.DataFrame(alignment).to_csv(out / 'taxonomy_by_seed.csv', index=False)
    pair_rows = agreement_rows(labels['tn2:en:native'], labels['tn2:vi:native'])
    pd.DataFrame(pair_rows).to_csv(out / 'partition_pairs.csv', index=False)
    write_json(out / 'partition_summary.json', summarize_agreement(pair_rows))
    print('VECTOR CORRESPONDENCE', flush=True)
    en = metrics.cosine_unit(inputs['tn2:en:native'])
    vi = metrics.cosine_unit(inputs['tn2:vi:native'])
    matched, negatives, delta = metrics.negative_pair_per_app(en, vi, chunk_rows=128)
    paired = frames['tn2'][['appId', 'genreId']].copy()
    paired['matched_cosine'], paired['negative_cosine_median'], paired['delta_cosine'] = matched, negatives, delta
    paired.to_parquet(out / 'paired_correspondence.parquet', index=False)
    write_json(out / 'correspondence_summary.json', {name: {'mean': float(np.mean(values)), 'median': float(np.median(values))} for name, values in [('matched_cosine', matched), ('negative_cosine_median', negatives), ('delta_cosine', delta)]})
    print('RETRIEVAL: exact tie-aware, full 13657-app catalogue in both directions', flush=True)
    result = g1c_retrieval.evaluate_bidirectional(inputs['tn2:en:native'], inputs['tn2:vi:native'],
        frames['tn2'].appId.tolist(), frames['tn2'].appId.tolist(), block_size=128,
        expected_count=13657, expected_ordered_ids_sha256=embed.ordered_ids_sha256(frames['tn2'].appId))
    with (out / 'retrieval_per_query.jsonl').open('x', encoding='utf-8') as handle:
        for row in result.pop('per_query'):
            handle.write(json.dumps(row) + '\n')
    write_json(out / 'retrieval_summary.json', result)
    strata = []
    provenance = provenance_labels(frames['tn1'])
    for cell in ('tn1:vi:char_budget', 'tn1:vi:native'):
        for group in ('translation-flagged', 'not-flagged', 'unknown'):
            mask = (provenance == group).to_numpy()
            y = frames['tn1'].genreId.astype(str).to_numpy()[mask]
            for seed, partition in zip(cluster.SEEDS, labels[cell]):
                strata.append(dict(cell=cell, provenance=group, seed=seed, n_apps=int(mask.sum()),
                    n_genres=int(len(np.unique(y))), ami=float(adjusted_mutual_info_score(y, partition[mask])),
                    ari=float(adjusted_rand_score(y, partition[mask])), interpretation='Fixed full-cohort partitions restricted to observed provenance stratum; not refitted'))
    pd.DataFrame(strata).to_csv(out / 'provenance_taxonomy_by_seed.csv', index=False)
    return {'n_cells': len(inputs), 'n_kmeans_fits': 40, 'retrieval_queries': 27314,
            'probe_status': 'PENDING_SPLIT_DECISION', 'scope': 'LaBSE KMeans main n_init10; no agglomerative, n_init1 diagnostic, other models or R02 results implied'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    args = parser.parse_args()
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):
        raise ValueError('Invalid run ID')
    out = ROOT / 'results/stage05' / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    files = ['pipeline/evaluate_stage05.py', 'pipeline/cluster.py', 'pipeline/evaluate_reviewed.py', 'pipeline/g1c_retrieval.py', 'pipeline/embed.py']
    receipt = dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip(),
        code_sha256={p: embed.sha256(ROOT / p) for p in files},
        validation_receipt_sha256=embed.sha256(VALIDATION), packages=embed.package_versions(),
        threads=4, protocol={'k':49,'n_init':10,'seeds':cluster.SEEDS,'input_vectors':'frozen float32 L2, no PCA/scaling/refit'})
    try:
        with threadpool_limits(limits=4):
            receipt.update(run(out))
        receipt['status'] = 'LABSE_CLUSTERING_RETRIEVAL_CORRESPONDENCE_COMPLETE'
    except Exception as exc:
        receipt.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        receipt['elapsed_seconds'] = time.perf_counter() - started
        receipt['artifacts_sha256'] = {p.name:embed.sha256(p) for p in out.iterdir() if p.is_file()}
        write_json(out / 'execution_receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
