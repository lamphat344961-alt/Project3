#!/usr/bin/env python
"""PAAC-VN Gate 4: evaluation of FROZEN clustering outputs (no reclustering).

Official inputs: data_v1/*.parquet, embeddings/tn[12]/*,
results/labels/*, results/audit/clustering_manifest.csv.

Frozen protocol implemented:
TN1: external AMI/ARI/NMI/V-measure/Hungarian macro-F1/ACC;
     internal Silhouette/DB/CH; 49 cluster sizes; diagnostic seed stability.
TN2-A: paired cosine, 1-cosine, all-negative per-app median and delta cosine.
TN2-B: 100 cross-language KMeans pairs, 45+45 within-language pairs,
       10 matched-seed auxiliary pairs; AMI/ARI/NMI.
TN2-C: EN and VI external alignment and EN-minus-VI matched-seed deltas.
TN2-D: geometry with same metrics as TN1 internal.

Outputs:
 results/evaluation/metrics_long.csv (single official metrics source)
 results/evaluation/tn2_a_app_pairs.parquet (paired app-level evidence for Gate 5)
 results/evaluation/evaluation_meta.json

No metric is selected based on observed performance. This script does NOT run
clustering, choose best model, perform bootstrap/Wilcoxon, or perform k-sweep.

Implementation choices clarified (not new experiments):
* Cosine Silhouette uses the full L2-normalized dataset and a mathematically
  equivalent cluster-sums identity (float32 rounding remains possible).
* Davies-Bouldin and Calinski-Harabasz use standard EUCLIDEAN geometry, even
  for the cosine-Agglomerative partitions. These are diagnostics only.
* TN2 negative baseline is a CUSTOM per-anchor median over every j != i;
  it is an empirical catalog-relative contrast, NOT a null distribution.
* Hungarian mapping maximizes MATCHED COUNT (ACC), not macro-F1. The reported
  macro-F1 uses that same ACC-optimal assignment.
* NMI (arithmetic) and V-measure (beta=1) are mathematically equivalent;
  both are reported strictly to honor the frozen six-column protocol.
* A seed pair is not an independent dataset; matched seed is auxiliary.
* Gate 4 checks embedding L2 norms and hashes CURRENT embedding bytes, but
  old clustering metadata did NOT record an embedding-byte hash. Thus the
  historical clustering input/embedding-byte identity cannot be proven by hash.
* Diagnostic n_init=1 is used ONLY for pairwise initialization stability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_mutual_info_score, adjusted_rand_score,
    calinski_harabasz_score, davies_bouldin_score,
    normalized_mutual_info_score, v_measure_score,
)

K = 49
SEEDS = tuple(range(42, 52))
EXPECTED_CONFIGS = 32
EXPECTED_METRIC_ROWS = 30128
TN1_N = 23883
TN2_N = 13657
MODEL_HASH = 'bdc8a5f2f72d699ff76dbedf5ceca015d5f2c2d2ff531c6289d900fd4b3e01d6'
DATA_HASH = {
    'tn1': '0cf5e72c00f1392e9f7e83ef2fc282e8deb9e81bc6e5d332ce5d761a41f7c52b',
    'tn2': '9208bfc42fc66b45c19496af34088cb742d1ea2e7cca25998e139b16c7e77493',
}
TN1_REPS = {
    'tfidf_lsa', 'phobert_base_v2', 'videberta_base',
    'sup_simcse_phobert', 'bkai_vietnamese_biencoder',
    'multilingual_e5_large', 'bge_m3', 'qwen3_embedding_0_6b',
}
TN2_REPS = {'tfidf_lsa', 'multilingual_e5_large', 'bge_m3', 'qwen3_embedding_0_6b'}
EXTERNAL = ('ami', 'ari', 'nmi', 'v_measure', 'macro_f1_hungarian', 'acc_hungarian')
INTERNAL = ('silhouette_cosine', 'davies_bouldin', 'calinski_harabasz')
OUTPUT_METRIC_COLUMNS = (
    'experiment', 'section', 'representation', 'truncation', 'language',
    'algorithm', 'run', 'seed', 'seed_other', 'comparison', 'metric', 'value',
    'n_apps', 'k', 'cluster_id', 'pair_role',
)


def sha256(path: Path, size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(size), b''):
            h.update(b)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def stem(exp: str, rep: str, lang: str, trunc: str) -> str:
    return f'{exp}__{rep}__{lang}__{trunc}'


def row_hash(app_ids: list[str]) -> str:
    return hashlib.sha256('\n'.join(app_ids).encode('utf-8')).hexdigest()


def validate_manifest(repo: Path) -> tuple[pd.DataFrame, Path]:
    mp = repo / 'results/audit/clustering_manifest.csv'
    if not mp.is_file():
        raise FileNotFoundError('Gate 3 manifest absent: ' + str(mp))
    m = pd.read_csv(mp, dtype={'experiment': str, 'representation': str,
                               'language': str, 'truncation': str, 'status': str})
    fields = {'experiment', 'representation', 'language', 'truncation', 'status'}
    if not fields.issubset(m.columns):
        raise ValueError('clustering_manifest missing fields: ' + str(fields - set(m.columns)))
    if len(m) != EXPECTED_CONFIGS or (m.status != 'COMPLETE').any():
        raise ValueError('Require Gate 3 32/32 COMPLETE, found: ' + str(m.status.value_counts().to_dict()))
    combos = {(str(r.experiment), str(r.representation), str(r.language), str(r.truncation))
              for r in m.itertuples(index=False)}
    expected = {('tn1', rep, 'vi', tr) for rep in TN1_REPS for tr in ('char_budget', 'native')}
    expected |= {('tn2', rep, lang, tr) for rep in TN2_REPS
                 for lang in ('en', 'vi') for tr in ('token_budget', 'native')}
    if combos != expected or len(combos) != EXPECTED_CONFIGS:
        raise ValueError('Manifest config grid differs from frozen 32 configs; '
                         f'missing={sorted(expected-combos)}, extra={sorted(combos-expected)}')
    return m, mp


class Evidence:
    def __init__(self, repo: Path, manifest: pd.DataFrame):
        self.repo = repo
        if sha256(repo / 'pipeline/models.yaml') != MODEL_HASH:
            raise ValueError('Frozen models.yaml SHA256 mismatch')
        self.manifest = {stem(r.experiment, r.representation, r.language, r.truncation): r
                         for r in manifest.itertuples(index=False)}
        self.datasets = {}
        self.embedding_provenance = {}
        for exp, filename, n in (('tn1', 'benchmark_v1.parquet', TN1_N),
                                 ('tn2', 'paired_en.parquet', TN2_N)):
            path = repo / 'data_v1' / filename
            if sha256(path) != DATA_HASH[exp]:
                raise ValueError(f'{exp} frozen dataset SHA256 mismatch: {path}')
            df = pd.read_parquet(path, columns=['appId', 'genreId'])
            if len(df) != n or df.genreId.nunique() != K:
                raise ValueError(f'{exp} unexpected dataset n/k: {len(df)} / {df.genreId.nunique()}')
            if df.appId.isna().any() or df.genreId.isna().any() or df.appId.duplicated().any():
                raise ValueError(f'{exp} has nulls or repeated appId')
            # Preserve original row order; clusters are aligned by row, not by sort.
            y = pd.Categorical(df.genreId.astype(str))
            self.datasets[exp] = {
                'ids': df.appId.astype(str).tolist(),
                'y': y.codes.astype(np.int32),
                'genre_names': y.categories.tolist(),
                'n': n,
                'app_hash': row_hash(df.appId.astype(str).tolist()),
                'dataset_hash': DATA_HASH[exp],
                'genres': df.genreId.astype(str).tolist(),
            }

    def get(self, exp: str, rep: str, lang: str, trunc: str, load_X: bool = False):
        key = stem(exp, rep, lang, trunc)
        if key not in self.manifest:
            raise ValueError('Unknown configuration: ' + key)
        base = self.repo / 'embeddings' / exp / key
        labelbase = self.repo / 'results/labels' / key
        meta = read_json(base.with_suffix('.json'))
        d = self.datasets[exp]
        for name, exp_val in (
            ('models_yaml_sha256', MODEL_HASH),
            ('experiment', exp), ('representation', rep), ('language', lang),
            ('truncation_mode', trunc), ('n_rows', d['n']),
            ('dataset_sha256', d['dataset_hash']), ('app_ids_sha256', d['app_hash']),
            ('embedding_dtype', 'float32'), ('l2_normalized', True),
        ):
            if meta.get(name) != exp_val:
                raise ValueError(f'{key}: embedding metadata {name} mismatch')
        if exp == 'tn1' and trunc == 'char_budget' and meta.get('n_chars') != 950:
            raise ValueError(key + ': TN1 N_CHARS must be 950')
        if exp == 'tn2' and trunc == 'token_budget' and meta.get('max_tokens_requested') != 512:
            raise ValueError(key + ': TN2 token budget must be 512')
        if trunc == 'native' and (meta.get('n_chars') is not None
                                  or meta.get('max_tokens_requested') is not None):
            raise ValueError(key + ': native configuration has non-native budget metadata')
        # Confirm sidecar matches original reference labels and ORIGINAL app order.
        rows = pq.read_table(base.with_suffix('.rows.parquet'),
                             columns=['row_idx', 'appId', 'genreId'])
        inds = rows['row_idx'].to_numpy()
        if not np.array_equal(inds, np.arange(d['n'])):
            raise ValueError(key + ': row_idx misaligned')
        if [str(x) for x in rows['appId'].to_pylist()] != d['ids']:
            raise ValueError(key + ': appId order misaligned')
        if [str(x) for x in rows['genreId'].to_pylist()] != d['genres']:
            raise ValueError(key + ': genreId order misaligned')
        cm = read_json(self.repo / 'results/labels' / f'{key}__cluster_meta.json')
        for name in ('dataset_sha256', 'models_yaml_sha256', 'app_ids_sha256',
                     'revision', 'n_rows', 'embedding_dim'):
            if cm.get('source_embedding', {}).get(name) != meta.get(name):
                raise ValueError(f'{key}: clustering provenance {name} mismatch')
        with np.load(self.repo / 'results/labels' / f'{key}__kmeans_main.npz',
                     allow_pickle=False) as z:
            main = np.asarray(z['labels'], dtype=np.int32)
            main_seeds = np.asarray(z['seeds']).tolist()
            if int(z['k']) != K or int(z['n_init']) != 10:
                raise ValueError(key + ': main KMeans protocol mismatch')
        with np.load(self.repo / 'results/labels' / f'{key}__kmeans_diag.npz',
                     allow_pickle=False) as z:
            diag = np.asarray(z['labels'], dtype=np.int32)
            diag_seeds = np.asarray(z['seeds']).tolist()
            if int(z['k']) != K or int(z['n_init']) != 1:
                raise ValueError(key + ': diagnostic KMeans protocol mismatch')
        agg = np.load(self.repo / 'results/labels' / f'{key}__agglomerative.npy', mmap_mode='r')
        if (main.shape != (10, d['n']) or diag.shape != (10, d['n'])
                or agg.shape != (d['n'],)
                or main_seeds != list(SEEDS) or diag_seeds != list(SEEDS)):
            raise ValueError(key + ': label shapes/seeds mismatch')
        # Check file fingerprints included in Gate 3 audit: no silent mutations.
        audit_row = self.manifest[key]
        for artifact, suffix in (('kmeans_main', '__kmeans_main.npz'),
                                 ('kmeans_diag', '__kmeans_diag.npz'),
                                 ('agglomerative', '__agglomerative.npy'),
                                 ('linkage', '__linkage.npy'),
                                 ('meta', '__cluster_meta.json')):
            fp = self.repo / 'results/labels' / f'{key}{suffix}'
            expected_hash = str(getattr(audit_row, f'{artifact}_sha256'))
            if sha256(fp) != expected_hash:
                raise ValueError(f'{key}: Gate 3 audited SHA changed: {fp.name}')
        # No silent geometry change: validate the actual floating-point vectors,
        # including all rows in --preflight. TF-IDF may contain legitimate zero
        # vectors, so exclude exact zero rows from the unit-norm assertion.
        array_path = base.with_suffix('.npy')
        X = np.load(array_path, mmap_mode='r')
        if X.dtype != np.float32 or X.shape != (d['n'], int(meta['embedding_dim'])):
            raise ValueError(key + ': embedding shape/dtype mismatch')
        norm_info = check_embedding_norms(X)
        if not load_X:
            X = None
        self.embedding_provenance[key] = {
            'sha256_npy_current': sha256(array_path),
            'zero_vector_count': norm_info['zero_vector_count'],
            'nonzero_unit_norm_max_error': norm_info['max_unit_norm_error'],
            'sha256_json': sha256(base.with_suffix('.json')),
            'model_compute_dtype': meta.get('model_compute_dtype', 'not_recorded'),
            'dataset_sha256': meta['dataset_sha256'],
            'models_yaml_sha256': meta['models_yaml_sha256'],
        }
        return {'key': key, 'main': main, 'diag': diag, 'agg': agg,
                'X': X, 'd': d, 'meta': meta}


def external_scores(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """49-to-49 Hungarian ACC/F1: agreement with taxonomy, not supervised accuracy."""
    if true.shape != pred.shape:
        raise ValueError('Reference and clustering label shapes differ')
    a, at = np.unique(true, return_inverse=True)
    b, bp = np.unique(pred, return_inverse=True)
    if len(a) != K or len(b) != K:
        raise ValueError(f'External evaluation expects 49 true and predicted groups; {len(a)}, {len(b)}')
    c = np.zeros((K, K), dtype=np.int64)
    np.add.at(c, (at, bp), 1)
    rows, cols = linear_sum_assignment(c, maximize=True)
    matched = c[rows, cols].astype(np.float64)
    # For the assigned label, TP is c[row,col], FN true support - TP,
    # FP predicted cluster size - TP. F1 = 2*TP/(row_support+col_support).
    true_sizes = c.sum(axis=1)[rows]
    pred_sizes = c.sum(axis=0)[cols]
    f1 = 2.0 * matched / (true_sizes + pred_sizes)
    return {
        'ami': float(adjusted_mutual_info_score(true, pred)),
        'ari': float(adjusted_rand_score(true, pred)),
        'nmi': float(normalized_mutual_info_score(true, pred, average_method='arithmetic')),
        'v_measure': float(v_measure_score(true, pred, beta=1.0)),
        'macro_f1_hungarian': float(f1.mean()),
        'acc_hungarian': float(matched.sum() / len(true)),
    }


def partition_similarity(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    return {
        'ami': float(adjusted_mutual_info_score(a, b)),
        'ari': float(adjusted_rand_score(a, b)),
        'nmi': float(normalized_mutual_info_score(a, b, average_method='arithmetic')),
    }


def check_embedding_norms(X: np.ndarray, chunk_rows: int = 2048,
                          tolerance: float = 1e-3) -> dict:
    """Fail-fast audit of frozen float32 L2 embeddings; allow exact zero rows.

    This is validation, not normalization/repair. Does not mutate input arrays.
    """
    if X.dtype != np.float32 or X.ndim != 2:
        raise ValueError(f'Embedding must be 2D float32, got {X.shape}/{X.dtype}')
    zero_count = 0
    max_error = 0.0
    for start in range(0, len(X), chunk_rows):
        z = np.asarray(X[start:start+chunk_rows])
        if not np.isfinite(z).all():
            raise ValueError(f'Non-finite embedding found in rows starting at {start}')
        norm = np.linalg.norm(z, axis=1)
        nonzero = norm > 0
        zero_count += int(np.count_nonzero(~nonzero))
        if nonzero.any():
            err = float(np.max(np.abs(norm[nonzero] - 1.0)))
            max_error = max(max_error, err)
            if err > tolerance:
                raise ValueError(f'Non-unit frozen embeddings, rows {start}:{start+len(z)}, '
                                 f'max abs norm error={err:.6g}, tol={tolerance}')
    return {'zero_vector_count': zero_count, 'max_unit_norm_error': max_error}


def cosine_unit(X: np.ndarray, chunk_rows: int = 2048) -> np.ndarray:
    """Safe roundoff normalization AFTER verifying frozen input unit norms.

    Zero vectors are preserved and reported in the embedding provenance.
    """
    check_embedding_norms(X, chunk_rows)
    n, p = X.shape
    out = np.empty((n, p), dtype=np.float32)
    for start in range(0, n, chunk_rows):
        end = min(n, start + chunk_rows)
        block = np.asarray(X[start:end], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError(f'Nonfinite embeddings at rows {start}:{end}')
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        out[start:end] = block / np.maximum(norms, np.float32(1e-30))
    return out


def exact_cosine_silhouette(Xunit: np.ndarray, labels: np.ndarray,
                            chunk_rows: int = 2048) -> float:
    """Exact per-point cosine silhouette without all-pairs distance matrix.

    For unit vectors, mean cosine distance from x to cluster C is
    1 - x · mean(C). Own-cluster excludes self and must use (count - 1).
    Floating-point roundoff may be <= several 1e-6 vs sklearn's direct path.
    """
    n, p = Xunit.shape
    classes, group = np.unique(labels, return_inverse=True)
    if len(classes) < 2 or len(classes) >= n:
        raise ValueError('Silhouette undefined for k<2 or k>=n')
    sizes = np.bincount(group, minlength=len(classes)).astype(np.int32)
    sums = np.zeros((len(classes), p), dtype=np.float64)
    np.add.at(sums, group, Xunit)
    means = np.asarray(sums / sizes[:, None], dtype=np.float32)
    total = 0.0
    for start in range(0, n, chunk_rows):
        end = min(n, start + chunk_rows)
        xi = Xunit[start:end]
        own = group[start:end]
        c = sizes[own]
        dot_means = np.asarray(xi @ means.T, dtype=np.float64)
        rows = np.arange(end - start)
        dot_own = dot_means[rows, own]
        selfdot = np.einsum('ij,ij->i', xi, xi, dtype=np.float64)
        # Sum_{j != i}(1 - x_i · x_j) / (|C|-1)
        a = np.zeros(end - start, dtype=np.float64)
        non_singletons = c > 1
        a[non_singletons] = (
            c[non_singletons] - 1
            - dot_own[non_singletons] * c[non_singletons]
            + selfdot[non_singletons]
        ) / (c[non_singletons] - 1)
        a = np.clip(a, 0.0, 2.0)
        dot_means[rows, own] = -np.inf
        b = np.clip(1.0 - dot_means.max(axis=1), 0.0, 2.0)
        denom = np.maximum(a, b)
        s = np.divide(b - a, denom, out=np.zeros_like(a), where=denom > 0)
        s[~non_singletons] = 0.0
        total += float(np.sum(s))
    return total / n


def geometry_scores(Xunit: np.ndarray, labels: np.ndarray,
                    chunk_rows: int) -> dict[str, float]:
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
    values = {
        'silhouette_cosine': exact_cosine_silhouette(Xunit, labels, chunk_rows),
        'davies_bouldin': float(davies_bouldin_score(Xunit, labels)),
        'calinski_harabasz': float(calinski_harabasz_score(Xunit, labels)),
    }
    if not all(np.isfinite(x) for x in values.values()):
        raise ValueError('Non-finite internal metric detected')
    return values


def negative_pair_per_app(en: np.ndarray, vi: np.ndarray,
                          chunk_rows: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All j!=i, exact per-app negative cosine median, with O(chunk*n) memory."""
    if en.shape != vi.shape or en.ndim != 2:
        raise ValueError('Paired EN/VI embedding shape mismatch')
    n = len(en)
    if n < 3:
        raise ValueError('Need at least 3 app pairs for negative baseline')
    aligned = np.einsum('ij,ij->i', en, vi, dtype=np.float64)
    med = np.empty(n, dtype=np.float64)
    n_neg = n - 1
    centers = ((n_neg - 1) // 2, n_neg // 2)
    vi_t = np.asarray(vi.T, order='C')
    for start in range(0, n, chunk_rows):
        end = min(n, start + chunk_rows)
        sim = np.asarray(en[start:end] @ vi_t, dtype=np.float32)
        np.clip(sim, -1.0, 1.0, out=sim)
        sim[np.arange(end-start), np.arange(start, end)] = np.inf
        # Inf occupies last column after partition; first n-1 are negatives.
        v = np.partition(sim, centers, axis=1)
        med[start:end] = (v[:, centers[0]].astype(np.float64)
                          + v[:, centers[1]].astype(np.float64)) * .5
    aligned = np.clip(aligned, -1.0, 1.0)
    return aligned, med, aligned - med


class Recorder:
    def __init__(self):
        self.rows = []
        self.app_rows = []

    def add(self, *, exp: str, section: str, rep: str, trunc: str, lang: str,
            method: str, run: str, metric: str, value: float, n: int,
            seed: int | None = None, other: int | None = None,
            comparison: str = '', cluster_id: int | None = None,
            role: str = ''):
        if not np.isfinite(value):
            raise ValueError(f'Invalid metric {section}/{rep}/{metric}: {value}')
        self.rows.append(dict(zip(OUTPUT_METRIC_COLUMNS, (
            exp, section, rep, trunc, lang, method, run, seed, other,
            comparison, metric, float(value), n, K, cluster_id, role,
        ))))

    def scores(self, values: dict, *, exp: str, section: str, rep: str,
               trunc: str, lang: str, method: str, run: str, n: int,
               seed: int | None = None, other: int | None = None,
               comparison: str = '', role: str = ''):
        for key, val in values.items():
            self.add(exp=exp, section=section, rep=rep, trunc=trunc,
                     lang=lang, method=method, run=run, metric=key,
                     value=val, n=n, seed=seed, other=other,
                     comparison=comparison, role=role)


def add_cluster_sizes(rec: Recorder, obj: dict, labels: np.ndarray,
                      section: str, method: str, run: str,
                      seed: int | None = None):
    exp = obj['meta']['experiment']
    rep = obj['meta']['representation']
    trunc = obj['meta']['truncation_mode']
    lang = obj['meta']['language']
    cluster_labels, group = np.unique(labels, return_inverse=True)
    counts = np.bincount(group)
    if len(counts) != K or np.sum(counts) != obj['d']['n']:
        raise ValueError(obj['key'] + ': unexpected cluster sizes')
    for cid, count in enumerate(counts):
        rec.add(exp=exp, section=section, rep=rep, trunc=trunc,
                lang=lang, method=method, run=run, metric='cluster_size',
                value=float(count), n=obj['d']['n'], seed=seed,
                cluster_id=int(cluster_labels[cid]))


def evaluate_alignment(rec: Recorder, obj: dict, section: str,
                       chunk_rows: int):
    exp = obj['meta']['experiment']
    rep = obj['meta']['representation']
    trunc = obj['meta']['truncation_mode']
    lang = obj['meta']['language']
    n = obj['d']['n']
    y = obj['d']['y']
    Xunit = cosine_unit(obj['X'], chunk_rows)
    external_by_run = {}
    for i, seed in enumerate(SEEDS):
        pred = obj['main'][i]
        values = external_scores(y, pred)
        external_by_run[seed] = values
        rec.scores(values, exp=exp, section=section,
                   rep=rep, trunc=trunc, lang=lang,
                   method='kmeans_main', run='main', n=n, seed=seed)
        rec.scores(geometry_scores(Xunit, pred, chunk_rows),
                   exp=exp, section='TN1_INTERNAL' if exp == 'tn1' else 'TN2_D',
                   rep=rep, trunc=trunc, lang=lang,
                   method='kmeans_main', run='main', n=n, seed=seed)
        add_cluster_sizes(rec, obj, pred,
                          section='TN1_SIZES' if exp == 'tn1' else 'TN2_SIZES',
                          method='kmeans_main', run='main', seed=seed)
    pred = np.asarray(obj['agg'])
    values = external_scores(y, pred)
    external_by_run['agglomerative'] = values
    rec.scores(values, exp=exp, section=section,
               rep=rep, trunc=trunc, lang=lang,
               method='agglomerative', run='deterministic', n=n)
    rec.scores(geometry_scores(Xunit, pred, chunk_rows),
               exp=exp, section='TN1_INTERNAL' if exp == 'tn1' else 'TN2_D',
               rep=rep, trunc=trunc, lang=lang,
               method='agglomerative', run='deterministic', n=n)
    add_cluster_sizes(rec, obj, pred,
                      section='TN1_SIZES' if exp == 'tn1' else 'TN2_SIZES',
                      method='agglomerative', run='deterministic')
    del Xunit
    return external_by_run


def add_diag_stability(rec: Recorder, obj: dict):
    exp = obj['meta']['experiment']
    rep = obj['meta']['representation']
    trunc = obj['meta']['truncation_mode']
    lang = obj['meta']['language']
    n = obj['d']['n']
    for i in range(len(SEEDS)):
        for j in range(i + 1, len(SEEDS)):
            sc = partition_similarity(obj['diag'][i], obj['diag'][j])
            rec.scores(sc, exp=exp, section='TN1_DIAG' if exp == 'tn1' else 'TN2_DIAG',
                       rep=rep, trunc=trunc, lang=lang, method='kmeans_diag',
                       run='initialization_stability', n=n,
                       seed=SEEDS[i], other=SEEDS[j],
                       comparison='within_language')


def run_tn1(ev: Evidence, rec: Recorder, chunk_rows: int):
    for rep in sorted(TN1_REPS):
        for tr in ('char_budget', 'native'):
            obj = ev.get('tn1', rep, 'vi', tr, load_X=True)
            print(f'[TN1] {obj["key"]}', flush=True)
            evaluate_alignment(rec, obj, 'TN1_EXTERNAL', chunk_rows)
            add_diag_stability(rec, obj)
            del obj


def run_tn2(ev: Evidence, rec: Recorder, chunk_rows: int, neg_chunk: int):
    for rep in sorted(TN2_REPS):
        for tr in ('token_budget', 'native'):
            en = ev.get('tn2', rep, 'en', tr, load_X=True)
            vi = ev.get('tn2', rep, 'vi', tr, load_X=True)
            print(f'[TN2] {rep} {tr}', flush=True)
            if en['d']['app_hash'] != vi['d']['app_hash']:
                raise ValueError('TN2 app pairing mismatch')
            # TN2-A: exact all-negative median baseline, no random pairs.
            en_x = cosine_unit(en['X'], chunk_rows)
            vi_x = cosine_unit(vi['X'], chunk_rows)
            paired, negative, delta = negative_pair_per_app(en_x, vi_x, neg_chunk)
            pairs = pd.DataFrame({
                'experiment': 'tn2', 'representation': rep, 'truncation': tr,
                'row_idx': np.arange(en['d']['n'], dtype=np.int32),
                'appId': en['d']['ids'],
                'genreId': en['d']['genres'],
                'matched_cosine': paired,
                'cosine_distance': 1 - paired,
                'negative_cosine_median': negative,
                'delta_cosine': delta,
            })
            rec.app_rows.append(pairs)
            # Report population mean and median for distributions, while keeping
            # app-level values intact for later app-paired bootstrap.
            for key, vals in (
                ('matched_cosine', paired),
                ('cosine_distance', 1. - paired),
                ('negative_cosine_median', negative),
                ('delta_cosine', delta),
            ):
                for stat in ('mean', 'median'):
                    rec.add(exp='tn2', section='TN2_A', rep=rep, trunc=tr,
                            lang='en_vs_vi', method='embedding_pair',
                            run='paired_apps', metric=key, n=en['d']['n'],
                            value=float(np.mean(vals) if stat == 'mean' else np.median(vals)),
                            role=stat)
            del en_x, vi_x, pairs
            # TN2-C + D for both languages; compute matched-seed EN-VI delta.
            en_scores = evaluate_alignment(rec, en, 'TN2_C', chunk_rows)
            vi_scores = evaluate_alignment(rec, vi, 'TN2_C', chunk_rows)
            for runkey in (*SEEDS, 'agglomerative'):
                method = 'agglomerative' if runkey == 'agglomerative' else 'kmeans_main'
                for metric in EXTERNAL:
                    rec.add(exp='tn2', section='TN2_C_DELTA', rep=rep, trunc=tr,
                            lang='en_minus_vi', method=method,
                            run='deterministic' if method == 'agglomerative' else 'main',
                            seed=None if method == 'agglomerative' else runkey,
                            comparison='same_seed' if method == 'kmeans_main' else 'agglomerative',
                            metric=metric, n=en['d']['n'],
                            value=en_scores[runkey][metric] - vi_scores[runkey][metric])
            add_diag_stability(rec, en)
            add_diag_stability(rec, vi)
            # TN2-B only KMeans main: ALL cross (10x10), within (C(10,2))
            # for each language, plus auxiliary matched-seed subset of cross.
            for i, sa in enumerate(SEEDS):
                for j, sb in enumerate(SEEDS):
                    s = partition_similarity(en['main'][i], vi['main'][j])
                    rec.scores(s, exp='tn2', section='TN2_B', rep=rep, trunc=tr,
                               lang='en_vs_vi', method='kmeans_main',
                               run='cross_language_all_pairs', n=en['d']['n'],
                               seed=sa, other=sb, comparison='cross_language')
                    if i == j:
                        rec.scores(s, exp='tn2', section='TN2_B', rep=rep, trunc=tr,
                                   lang='en_vs_vi', method='kmeans_main',
                                   run='matched_seed_auxiliary', n=en['d']['n'],
                                   seed=sa, other=sb, comparison='cross_language',
                                   role='auxiliary_subset_of_all_pairs')
            for obj in (en, vi):
                for i in range(len(SEEDS)):
                    for j in range(i + 1, len(SEEDS)):
                        s = partition_similarity(obj['main'][i], obj['main'][j])
                        rec.scores(s, exp='tn2', section='TN2_B', rep=rep, trunc=tr,
                                   lang=obj['meta']['language'], method='kmeans_main',
                                   run='within_language_all_pairs', n=obj['d']['n'],
                                   seed=SEEDS[i], other=SEEDS[j],
                                   comparison='empirical_seed_stability_baseline')
            del en, vi


def atomic_csv(df: pd.DataFrame, target: Path) -> None:
    tmp = target.with_name(target.name + '.tmp')
    df.to_csv(tmp, index=False, encoding='utf-8')
    os.replace(tmp, target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, default=Path('.'))
    ap.add_argument('--out-dir', type=Path, default=Path('results/evaluation'))
    ap.add_argument('--silhouette-chunk-rows', type=int, default=2048,
                    help='Only numerical memory block size; no sampling.')
    ap.add_argument('--negative-chunk-rows', type=int, default=128,
                    help='Only BLAS/median memory block size; all negatives used.')
    ap.add_argument('--preflight', action='store_true',
                    help='Check frozen 32-config Gate 3 manifest + datasets, no metrics.')
    args = ap.parse_args()
    if args.silhouette_chunk_rows < 1 or args.negative_chunk_rows < 1:
        raise ValueError('Chunk sizes must be positive')
    repo = args.repo.resolve()
    out = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir
    manifest, mpath = validate_manifest(repo)
    ev = Evidence(repo, manifest)
    print('Gate 3 manifest: 32/32 COMPLETE; canonical datasets verified.', flush=True)
    if args.preflight:
        # Full preflight checks metadata/labels and audited cluster hashes,
        # row pairing AND finite/unit vector contents via memory mapping.
        for row in manifest.itertuples(index=False):
            ev.get(row.experiment, row.representation, row.language,
                   row.truncation, load_X=False)
        print('PREFLIGHT PASS: 32/32 metadata, labels/hashes, row order, finite/L2 embeddings verified.')
        print('NOTICE: Gate 3 lacks historical NPY-byte hashes; current NPY fingerprints recorded only.')
        return
    if any((out / fn).exists() for fn in ('metrics_long.csv',
                                         'tn2_a_app_pairs.parquet',
                                         'evaluation_meta.json')):
        raise FileExistsError('Refusing overwrite of official evaluation outputs in: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    rec = Recorder()
    run_tn1(ev, rec, args.silhouette_chunk_rows)
    run_tn2(ev, rec, args.silhouette_chunk_rows, args.negative_chunk_rows)
    metrics = pd.DataFrame(rec.rows, columns=OUTPUT_METRIC_COLUMNS)
    pairs = pd.concat(rec.app_rows, ignore_index=True)
    if len(pairs) != 8 * TN2_N:
        raise ValueError(f'TN2-A app rows wrong count: {len(pairs)} != {8 * TN2_N}')
    expected_config_count = metrics[['experiment', 'representation',
                                     'truncation', 'language']].drop_duplicates()
    if len(expected_config_count[expected_config_count.experiment.isin(['tn1','tn2'])]) < 32:
        raise ValueError('Missing metric configurations')
    if len(metrics) != EXPECTED_METRIC_ROWS:
        raise ValueError(f'Expected {EXPECTED_METRIC_ROWS} metric rows; found {len(metrics)}')
    unique_key = [c for c in OUTPUT_METRIC_COLUMNS if c not in ('value', 'n_apps', 'k')]
    if metrics.duplicated(subset=unique_key).any():
        raise ValueError('Duplicate metric keys detected in official long table')
    # Always write app-level evidence first, then long table, then done marker.
    tmp_pairs = out / 'tn2_a_app_pairs.parquet.tmp'
    pairs.to_parquet(tmp_pairs, index=False)
    os.replace(tmp_pairs, out / 'tn2_a_app_pairs.parquet')
    atomic_csv(metrics, out / 'metrics_long.csv')
    elapsed = time.perf_counter() - start
    provenance = {
        'project': 'PAAC-VN', 'gate': 4,
        'status': 'COMPLETE',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'clustering_manifest_sha256': sha256(mpath),
        'dataset_sha256': DATA_HASH,
        'sources': ev.embedding_provenance,
        'n_metrics_rows': len(metrics),
        'n_tn2_a_pairs_rows': len(pairs),
        'silhouette_definition': 'cosine on all apps, algebraically equivalent cluster sums, float32 tolerance',
        'davies_bouldin_calinski_harabasz_geometry': 'euclidean sklearn standard',
        'negative_pair_baseline': 'per app i exact median of all cos(EN_i, VI_j), j!=i',
        'interpretation_restriction': 'internal metrics must NOT rank representations across different embedding spaces',
        'script_sha256': sha256(Path(__file__).resolve()),
        'metric_sections': {str(k): int(v) for k, v in metrics['section'].value_counts().items()},
        'delta_convention': 'EN_minus_VI for TN2-C; matched_minus_negative for TN2-A',
        'hungarian_mapping': 'ACC-maximizing assignment; macro-F1 on SAME mapping, not F1-optimal',
        'nmi_v_measure_redundancy': 'NMI arithmetic is exactly V-measure beta=1',
        'embedding_byte_hash_limit': 'SHA256 of current NPY stored; Gate 3 did not freeze NPY byte hashes',
        'tn2_negative_scope': 'all non-matched apps, including same genre and possible near duplicates',
        'internal_metric_geometry': 'cosine silhouette; Euclidean DB and CH; no cross-space ranking',
        'seeds': list(SEEDS), 'k': K,
        'diagnostic_only_stability': True,
        'runtime_seconds': elapsed,
        'metrics_long_sha256': sha256(out / 'metrics_long.csv'),
        'tn2_a_app_pairs_sha256': sha256(out / 'tn2_a_app_pairs.parquet'),
    }
    mt = out / 'evaluation_meta.json.tmp'
    mt.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(mt, out / 'evaluation_meta.json')
    print('=' * 72)
    print(f'GATE 4 EVALUATION COMPLETE: metrics_rows={len(metrics):,}; TN2_app_rows={len(pairs):,}')
    print(f'Runtime: {elapsed / 60:.1f} minutes | Output: {out}')
    print('METRICS LONG: results/evaluation/metrics_long.csv')
    print('=' * 72)


if __name__ == '__main__':
    main()
