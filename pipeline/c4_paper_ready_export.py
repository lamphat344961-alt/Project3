#!/usr/bin/env python3
"""PAAC-VN C4 paper-ready exporter.

Read-only with respect to canonical experiment artifacts. This script does NOT
run embeddings, clustering, evaluation, bootstrap, or inferential tests. It only
filters/reshapes frozen outputs, adds pre-approved reporting labels, and writes
new paper-ready artifacts under results/final_closure/C4_paper_ready/.

Refuses overwrite.
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

SESOI = 0.01


def sha256(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(block), b''):
            h.update(b)
    return h.hexdigest()


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + '.tmp')
    df.to_csv(tmp, index=False, encoding='utf-8')
    os.replace(tmp, path)


def atomic_text(text: str, path: Path) -> None:
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path)


def classify_equivalence(row: pd.Series) -> tuple[str, str, str]:
    """Return classification, inference_scope, equivalence_rule.

    Frozen reporting rule:
    - primary TN1-A/TN2-C: Holm-adjusted TOST determines EQUIVALENT.
      If not Holm-equivalent, 90% CI wholly outside ±.01 => NOMINALLY_BEYOND_SESOI;
      otherwise INCONCLUSIVE_SESOI, including nominal CI-inside but Holm failure.
    - exploratory TOP3: nominal unadjusted TOST may label EQUIVALENT, but scope is
      explicitly exploratory. Otherwise use same CI-based descriptive taxonomy.
    """
    role = str(row['role'])
    fam = str(row['family'])
    lo = float(row['ci90_lo'])
    hi = float(row['ci90_hi'])
    outside = (hi < -SESOI) or (lo > SESOI)

    if role == 'primary' and fam in {'TN1-A', 'TN2-C'}:
        equivalent = bool(row['holm_equivalence_reject_005'])
        if equivalent:
            return 'EQUIVALENT', 'primary_familywise', 'holm_adjusted_tost_alpha_0.05'
        if outside:
            return 'NOMINALLY_BEYOND_SESOI', 'descriptive_only', 'no_prespecified_minimum_effect_test'
        return 'INCONCLUSIVE_SESOI', 'primary_familywise', 'holm_tost_not_rejected_or_boundary_crossed'

    if role == 'exploratory' and fam == 'TOP3':
        p = row['tost_p_001']
        equivalent = pd.notna(p) and float(p) < 0.05
        if equivalent:
            return 'EQUIVALENT', 'exploratory', 'nominal_unadjusted_tost_alpha_0.05'
        if outside:
            return 'NOMINALLY_BEYOND_SESOI', 'exploratory_descriptive', 'no_confirmatory_minimum_effect_test'
        return 'INCONCLUSIVE_SESOI', 'exploratory', 'nominal_tost_not_rejected_or_boundary_crossed'

    return 'NOT_APPLICABLE', role, 'no_equivalence_classification_for_family'


def summarize_metric_rows(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Descriptive aggregation only; no new inferential metric."""
    g = df.groupby(keys, dropna=False)['value']
    out = g.agg(n_runs='count', mean='mean', std='std', min='min', max='max').reset_index()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, default=Path('.'))
    args = ap.parse_args()
    root = args.repo.resolve()

    src = {
        'metrics_long': root/'results/evaluation/metrics_long.csv',
        'bootstrap_summary': root/'results/statistics/gate5/bootstrap_summary.csv',
        'gate5_manifest': root/'results/statistics/gate5/gate5_manifest.json',
        'statistical_audit': root/'results/statistics/gate5/statistical_audit.json',
        'k_sweep_csv': root/'results/figures/k_sweep_tfidf_char_budget.csv',
        'k_sweep_manifest': root/'results/figures/k_sweep_tfidf_char_budget_manifest.json',
        'c3_co_raw': root/'results/gate6/error_analysis/gate6_coassignment_6pairs.csv',
        'c3_co_summary': root/'results/gate6/error_analysis/gate6_coassignment_6pairs_summary.csv',
        'c3_map_raw': root/'results/gate6/error_analysis/gate6_mapped_confusion_6pairs.csv',
        'c3_map_summary': root/'results/gate6/error_analysis/gate6_mapped_confusion_6pairs_summary.csv',
        'c3_recon': root/'results/gate6/error_analysis/gate6_hungarian_reconciliation.csv',
        'c3_manifest': root/'results/gate6/error_analysis/gate6_error_analysis_manifest.json',
        'a1c': root/'results/audit/final_forensic_closure/a1c.json',
        'a2': root/'results/audit/final_forensic_closure/a2.json',
        'a3': root/'results/audit/final_forensic_closure/a3.json',
        'c0_manifest': root/'results/final_closure/C0_archive/c0_archive_manifest.json',
    }
    missing = [str(p) for p in src.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing C4 input(s):\n' + '\n'.join(missing))

    out = root/'results/final_closure/C4_paper_ready'
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('C4 output directory is not empty; refusing overwrite: ' + str(out))

    metrics = read_csv_required(src['metrics_long'])
    boot = read_csv_required(src['bootstrap_summary'])

    # 1) TN1 official external metrics: exact raw rows + descriptive summary.
    tn1 = metrics[metrics['section'].eq('TN1_EXTERNAL')].copy()
    atomic_csv(tn1, out/'tn1_external_raw.csv')
    tn1_summary = summarize_metric_rows(
        tn1,
        ['representation','truncation','language','algorithm','metric']
    )
    atomic_csv(tn1_summary, out/'tn1_main_results.csv')

    # 2) TN2-C official external point metrics: exact raw rows + descriptive summary.
    tn2c_point = metrics[metrics['section'].eq('TN2_C')].copy()
    atomic_csv(tn2c_point, out/'tn2_genre_alignment_point_raw.csv')
    tn2c_point_summary = summarize_metric_rows(
        tn2c_point,
        ['representation','truncation','language','algorithm','metric']
    )
    atomic_csv(tn2c_point_summary, out/'tn2_genre_alignment_point_summary.csv')

    # 3) Gate5 family extracts.
    family_files = {
        'TN1-A': 'tn1_context_effects.csv',
        'TN1-B': 'tn1_neural_vs_tfidf_exploratory.csv',
        'TOP3': 'tn1_top3_exploratory.csv',
        'TN2-A': 'tn2_vector_alignment.csv',
        'TN2-B': 'tn2_partition_shift.csv',
        'TN2-C': 'tn2_genre_alignment_inference.csv',
    }
    for fam, fn in family_files.items():
        part = boot[boot['family'].eq(fam)].copy()
        if part.empty:
            raise ValueError('Missing Gate5 family: ' + fam)
        atomic_csv(part, out/fn)

    # 4) Pre-approved SESOI taxonomy only for TN1-A, TN2-C, TOP3.
    eq = boot[boot['family'].isin(['TN1-A','TN2-C','TOP3'])].copy()
    labels = eq.apply(classify_equivalence, axis=1, result_type='expand')
    labels.columns = ['equivalence_classification','inference_scope','equivalence_rule']
    eq = pd.concat([eq.reset_index(drop=True), labels.reset_index(drop=True)], axis=1)
    eq['sesoi_abs_ami'] = SESOI
    eq['confirmatory_minimum_effect_test'] = False
    atomic_csv(eq, out/'gate5_equivalence_classification.csv')

    # 5) C2 and C3: exact copies into paper-ready naming; no recalculation.
    copies = {
        src['k_sweep_csv']: out/'k_sweep.csv',
        src['c3_co_raw']: out/'error_analysis_6pairs_coassignment_raw.csv',
        src['c3_co_summary']: out/'error_analysis_6pairs_coassignment_summary.csv',
        src['c3_map_raw']: out/'error_analysis_6pairs_mapped_confusion_raw.csv',
        src['c3_map_summary']: out/'error_analysis_6pairs_mapped_confusion_summary.csv',
        src['c3_recon']: out/'error_analysis_hungarian_reconciliation.csv',
    }
    for source, target in copies.items():
        target.write_bytes(source.read_bytes())

    # 6) Claim-evidence matrix: evidence mapping, not new statistical analysis.
    claims = pd.DataFrame([
        {
            'claim_id':'RQ1_CONTEXT',
            'claim':'Context-regime effects on AMI are representation-dependent; no blanket native-context advantage is supported.',
            'evidence':'tn1_context_effects.csv',
            'status':'supported_with_SESOI_taxonomy',
            'scope':'TN1 fixed corpus; KMeans AMI; paired app bootstrap',
            'mandatory_caveat':'Use EQUIVALENT / INCONCLUSIVE_SESOI / NOMINALLY_BEYOND_SESOI; do not equate failed equivalence with practical importance.'
        },
        {
            'claim_id':'RQ1_TOP3',
            'claim':'Among the three leading representations, ordering changes with context regime; no representation dominates both regimes.',
            'evidence':'tn1_top3_exploratory.csv',
            'status':'exploratory',
            'scope':'Post-point-estimate top-three comparisons',
            'mandatory_caveat':'Nominal unadjusted TOST only; label all conclusions exploratory.'
        },
        {
            'claim_id':'RQ2_VECTOR',
            'claim':'Within each representation, matched EN/VI app vectors show positive matched-minus-negative cosine contrast.',
            'evidence':'tn2_vector_alignment.csv; A3 forensic report',
            'status':'supported',
            'scope':'TN2 paired apps',
            'mandatory_caveat':'Do not rank raw cosine contrasts across embedding spaces; geometry/anisotropy differs.'
        },
        {
            'claim_id':'RQ2_PARTITION',
            'claim':'Cross-language partition agreement is lower than the within-language seed-stability baseline in the frozen TN2 comparison.',
            'evidence':'tn2_partition_shift.csv',
            'status':'supported_with_structural_caveat',
            'scope':'TN2 KMeans frozen partitions',
            'mandatory_caveat':'Within-language compares partitions from the same input while cross-language compares different inputs; emphasize magnitude, not merely negative sign.'
        },
        {
            'claim_id':'RQ2_GENRE',
            'claim':'EN-versus-MT-VI genre alignment differences vary by representation and regime.',
            'evidence':'tn2_genre_alignment_inference.csv; gate5_equivalence_classification.csv',
            'status':'supported',
            'scope':'TN2 fixed paired corpus',
            'mandatory_caveat':'Observational source-English vs MT-Vietnamese comparison; no causal translation-effect claim.'
        },
        {
            'claim_id':'GATE6_K',
            'claim':'The supplementary saved-linkage k-sweep does not provide a single alternative k preferred by all internal diagnostics.',
            'evidence':'k_sweep.csv; k_sweep manifest',
            'status':'descriptive_only',
            'scope':'TN1 TF-IDF-LSA char_budget Agglomerative saved Z only',
            'mandatory_caveat':'k=49 remains fixed by label-space design; this is not model selection or evidence of a true natural k.'
        },
        {
            'claim_id':'GATE6_ERRORS',
            'claim':'The six pre-specified related genre pairs are analyzed descriptively using mapping-free co-assignment, with Hungarian mapped confusion auxiliary.',
            'evidence':'error_analysis_6pairs_coassignment_summary.csv; error_analysis_6pairs_mapped_confusion_summary.csv',
            'status':'descriptive_only',
            'scope':'TN1 16 KMeans configurations × 10 seeds × 6 pre-specified genre pairs',
            'mandatory_caveat':'No bootstrap, p-value, Holm, TOST, or post-hoc genre-pair selection.'
        },
    ])
    atomic_csv(claims, out/'claim_evidence_matrix.csv')

    limitations = "# PAAC-VN — C4 limitations register\n\n"
    limitations += "This register carries forward already documented limitations; it does not introduce new experimental claims.\n\n"
    items = [
        "TN1 is a mixed-source Vietnamese corpus; translation provenance is not isolated as a causal factor.",
        "TN2 compares English-source listings with Google MT-Vietnamese listings observationally; English source is not guaranteed human-original.",
        "TN1 common-content uses characters while TN2 common-token uses tokenizer units; budgets are not directly comparable.",
        "TN2 token-budget can have asymmetric EN/VI truncation.",
        "genreId is an external taxonomy with label noise, not semantic ground truth or proof of 49 natural clusters.",
        "Internal geometry metrics must not rank different embedding spaces with different geometry/dimension.",
        "KMeans outer seeds are initialization variability, not independent datasets.",
        "Gate5 bootstrap resamples apps on fixed partitions; it does not refit/recluster and is not independent-dataset replication.",
        "Gate5 percentile-tail p-values are approximate/not null-calibrated; B=1000 limits tail resolution.",
        "TOP3 comparisons are exploratory and were selected after Gate4 point estimates.",
        "TN2-B within-language and cross-language baselines are structurally asymmetric (same input versus different inputs).",
        "Raw TN2-A cosine contrasts are not directly rankable across embedding spaces due to differing geometry/anisotropy.",
        "A1-C: ViDeBERTa runtime loader maps to checkpoint `weight`, but effective GDES/export semantics remain NOT_VERIFIED; do not attribute low AMI to mean pooling alone.",
        "A2: multilingual-E5 uses max_len=512 in both TN2 regimes; EN artifacts differ only at very small numerical scale, so downstream differences must not be interpreted as context-capacity effects.",
        "A0/C0: historical artifact provenance is incomplete for some embedding runtimes; exact end-to-end historical byte identity cannot be claimed.",
        "Main `.venv` experienced a later NumPy/matplotlib drift event after C0/C1/C3 outputs; supplementary C2 was executed in a separate `.venv_c2` and must be documented separately in C5.",
        "Supplementary k-sweep is a saved-linkage internal sensitivity analysis only; it cannot change the main k=49 design or main RQ conclusions.",
    ]
    for x in items:
        limitations += f"- {x}\n"
    atomic_text(limitations, out/'limitations_register.md')

    # 7) Manifest hashes all C4 outputs and all source inputs.
    outputs = []
    for p in sorted(out.iterdir()):
        if p.name == 'c4_manifest.json':
            continue
        if p.is_file():
            outputs.append({'path': str(p.relative_to(root)), 'sha256': sha256(p), 'size_bytes': p.stat().st_size})
    source_hashes = {k: sha256(v) for k, v in src.items()}
    manifest = {
        'project':'PAAC-VN',
        'phase':'C4_PAPER_READY_EXPORT',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'COMPLETE',
        'method':'read-only filter/reshape/copy of frozen outputs; no new experiment or inferential test',
        'sesoi_abs_ami':SESOI,
        'equivalence_reporting_rule':{
            'primary':'Holm-adjusted TOST; nominal 90% CI in bounds but Holm failure => INCONCLUSIVE_SESOI',
            'exploratory_TOP3':'nominal unadjusted TOST with inference_scope=exploratory',
            'nominally_beyond':'90% CI wholly outside ±0.01; descriptive only because no confirmatory minimum-effect test was prespecified'
        },
        'source_sha256':source_hashes,
        'outputs':outputs,
        'canonical_inputs_modified':False,
        'gate6_cannot_change_main_conclusions':True,
        'script_sha256':sha256(Path(__file__).resolve()),
    }
    tmp = out/'c4_manifest.json.tmp'
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, out/'c4_manifest.json')
    print(f'C4 COMPLETE: {len(outputs)} paper-ready artifacts -> {out}')
    print('manifest:', out/'c4_manifest.json')


if __name__ == '__main__':
    main()
