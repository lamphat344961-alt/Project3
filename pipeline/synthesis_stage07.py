"""Stage 07: assemble verified saved results; no fitting or inference.

Run from the repository: python -B -m pipeline.synthesis_stage07 RUN_ID
Output IDs are append-only. The selected notebook remains untouched.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
S5 = ROOT / 'results/stage05/combined_20261025_001'
S6 = ROOT / 'results/stage06/n05_20261025_001'
NOTEBOOK = ROOT / 'results/stage07/gallery_20261025_005/PAAC_VN_Stage07_Review.ipynb'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(run_id):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if Path(run_id).name != run_id or run_id in ('.', '..'):
        raise ValueError('Use a single output folder name')
    out = ROOT / 'results/stage07' / run_id
    out.mkdir(exist_ok=False)
    inputs = {}

    def record(p):
        inputs[p.relative_to(ROOT).as_posix()] = sha(p)
        return p

    def read(folder, name):
        p = record(folder / name)
        receipt = json.loads((folder / 'execution_receipt.json').read_text(encoding='utf-8'))
        assert sha(p) == receipt['artifacts_sha256'][name], name
        return pd.read_csv(p)

    for folder in (S5, S6):
        record(folder / 'execution_receipt.json')
    b = pd.read_parquet(record(ROOT / 'data_v1/benchmark_v1.parquet'))
    paired = pd.read_parquet(record(ROOT / 'data_v1/paired_en.parquet'))
    assert len(b) == 23883 and len(paired) == 13657
    assert b.appId.is_unique and paired.appId.is_unique
    assert set(paired.appId) <= set(b.appId)
    tables = {}

    def save(name, frame):
        tables[name] = frame
        frame.to_csv(out / (name + '.csv'), index=False)

    flags = b.is_translated.map({True: 'translation-flagged', False: 'not-flagged'}).fillna('unknown')
    save('01_dataset_profile', pd.DataFrame([
        {'cohort': 'TN1', 'stratum': k, 'n_apps': int((flags == k).sum()),
         'n_genres': b.loc[flags == k, 'genreId'].nunique()}
        for k in ['translation-flagged', 'not-flagged', 'unknown']
    ] + [{'cohort': 'TN1', 'stratum': 'all', 'n_apps': len(b), 'n_genres': b.genreId.nunique()},
         {'cohort': 'TN2', 'stratum': 'all paired', 'n_apps': len(paired), 'n_genres': paired.genreId.nunique()}]))
    index = read(S5, 'artifact_index.csv')
    save('02_recipe_scope', index.drop(columns=['embedding_sha256', 'rows_sha256', 'metadata_sha256', 'labels_sha256']))
    taxonomy = read(S5, 'taxonomy_summary.csv')
    probe = read(S5, 'probe_all.csv').rename(columns={'representation': 'probe_representation'})
    probe['representation'] = probe.probe_representation.replace({'tfidf_lsa_train_only': 'tfidf_lsa'})
    combined = taxonomy.merge(probe, on=['representation', 'cell'], how='left', validate='one_to_one')
    combined['probe_fit_scope'] = np.where(combined.probe_representation.eq('tfidf_lsa_train_only'),
        'train-only vocabulary/IDF/SVD; different matrices from clustering',
        np.where(combined.probe_representation.notna(), 'frozen encoder; fixed held-out test', 'not evaluated'))
    assert len(combined) == 34 and combined.test_macro_f1.notna().sum() == 26
    save('05_rq2_taxonomy_probe', combined)
    retrieval = read(S5, 'retrieval_all.csv')
    correspondence = read(S6, 'correspondence_app_dispersion.csv')
    wide = correspondence.pivot(index=['representation', 'condition'], columns='metric', values='point').reset_index()
    save('03_rq1_correspondence_retrieval', retrieval.merge(wide, on=['representation', 'condition'], validate='many_to_one'))
    partition = read(S5, 'partition_all.csv')
    dispersion = read(S6, 'partition_app_dispersion.csv')
    s = dispersion[dispersion.component.eq('S')][['representation', 'condition', 'q025', 'q975', 'minimum', 'maximum']]
    pairtable = partition.merge(s, on=['representation', 'condition'], validate='one_to_one')
    tn2 = taxonomy[taxonomy.cell.str.startswith('tn2:')].copy()
    tn2[['experiment', 'language', 'condition']] = tn2.cell.str.split(':', expand=True)
    for language in ('en', 'vi'):
        t = tn2[tn2.language.eq(language)][['representation', 'condition', 'ami_mean', 'ari_mean']]
        pairtable = pairtable.merge(t.rename(columns={'ami_mean': language + '_ami_mean', 'ari_mean': language + '_ari_mean'}), on=['representation', 'condition'], validate='one_to_one')
    pairtable['en_minus_vi_ami'] = pairtable.en_ami_mean - pairtable.vi_ami_mean
    assert len(pairtable) == 9 and (pairtable.maximum < 0).all()
    save('04_rq1_partitions_taxonomy', pairtable)
    save('06_input_language_contrasts', read(S6, 'paired_descriptive_contrasts.csv'))
    prov = read(S5, 'provenance_all_by_seed.csv')
    prov = prov.groupby(['representation', 'cell', 'provenance'], as_index=False).agg(
        n_apps=('n_apps', 'first'), n_seeds=('seed', 'nunique'), ami_mean=('ami', 'mean'),
        ami_seed_sd=('ami', 'std'), ari_mean=('ari', 'mean'))
    support = dict(zip(tables['01_dataset_profile'].stratum, tables['01_dataset_profile'].n_genres))
    prov['n_genres'] = prov.provenance.map(support)
    prov['estimand'] = 'full-cohort partitions restricted to stratum; no refit'
    assert len(prov) == 48 and prov.n_seeds.eq(10).all()
    save('07_rq2_provenance', prov)
    for kind in ('taxonomy', 'probe', 'retrieval'):
        save('08_' + kind + '_rank_support', read(S6, kind + '_rank_stability.csv'))

    # Illustrations are deterministic saved-query examples, not selected by title.
    # Per recipe/direction choose lexicographically first exact hit and exact miss.
    examples = []
    for rep in sorted(partition.representation.unique()):
        if rep == 'labse':
            folder = ROOT / 'results/stage05/labse_20261025_001'
        elif rep == 'multilingual_e5_large':
            folder = ROOT / 'results/stage05/e5_tn2_20261025_001/native'
        else:
            folder = ROOT / 'results/stage05/remaining_eval_20261025_001' / (rep + '__native')
        queries = pd.read_json(record(folder / 'retrieval_per_query.jsonl'), lines=True)
        for direction, group in queries.groupby('direction'):
            for result, val in [('hit', 1), ('miss', 0)]:
                candidates = group[group.expected_recall_at_1.eq(val)].sort_values('query_app_id')
                if len(candidates):
                    row = candidates.iloc[0].to_dict()
                    row.update(representation=rep, condition='native', illustration=result,
                               selection='lexicographically first appId among exact R@1 hits/misses, per recipe/direction')
                    examples.append(row)
    app_examples = pd.DataFrame(examples).merge(b[['appId', 'title_clean', 'genreId']], left_on='query_app_id', right_on='appId', validate='many_to_one')
    save('09_app_examples', app_examples)

    # All 49 genres, all five native paired recipes, fixed seed 42: avoid cherry-picking.
    genres = []
    for rep in sorted(partition.representation.unique()):
        lang_labels = {}
        for language in ('en', 'vi'):
            row = index[(index.experiment == 'tn2') & (index.representation == rep) &
                        (index.language == language) & (index.condition == 'native')].iloc[0]
            lp = record(ROOT / row.labels_path)
            assert sha(lp) == row.labels_sha256
            rp = record((ROOT / row.embedding_path).with_suffix('.rows.parquet'))
            assert sha(rp) == row.rows_sha256
            rows = pd.read_parquet(rp)
            assert rows.appId.tolist() == paired.appId.tolist()
            with np.load(lp, allow_pickle=False) as z:
                position = list(z['seeds']).index(42)
                labels = z['labels'][position]
            assert len(labels) == len(paired)
            lang_labels[language] = labels
        for genre, subset in paired.groupby('genreId'):
            mask = paired.genreId.eq(genre).to_numpy()
            rec = {'representation': rep, 'condition': 'native', 'seed': 42, 'genreId': genre, 'n_apps': len(subset)}
            for language in ('en', 'vi'):
                counts = pd.Series(lang_labels[language][mask]).value_counts()
                rec[language + '_occupied_clusters'] = len(counts)
                rec[language + '_dominant_cluster_fraction'] = counts.iloc[0] / len(subset)
            genres.append(rec)
    save('10_genre_examples_all49', pd.DataFrame(genres))

    # Execute only plotting cells, replacing their display hook with vector export.
    notebook = json.loads(record(NOTEBOOK).read_text(encoding='utf-8'))
    cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    assert len(cells) == 9
    namespace = {'__name__': '__stage07_export__'}
    exec(compile(''.join(cells[0]['source']), str(NOTEBOOK) + ':setup', 'exec'), namespace)
    figdir = out / 'figures'
    figdir.mkdir()
    figure_records = []

    def export(fig, slug, title, origin, caption, group):
        if slug == '02a_benchmark_all49':
            caption = caption.replace(' Description lengths are shown separately in 02c.', '')
        fig.suptitle(title, fontsize=15, fontweight='bold', y=.995)
        fig.text(.015, .008, caption, fontsize=8, ha='left', va='bottom', wrap=True)
        fig.tight_layout(rect=(0, .075, 1, .95), pad=1.5)
        for extension in ('png', 'pdf', 'svg'):
            fig.savefig(figdir / (slug + '.' + extension), dpi=160)
        figure_records.append({'slug': slug, 'title': title, 'existing_caption': caption, 'source': origin})
        namespace['plt'].close(fig)

    namespace['show_figure'] = export
    for cell in cells[1:]:
        exec(compile(''.join(cell['source']), str(NOTEBOOK) + ':' + cell['id'], 'exec'), namespace)
    assert len(figure_records) == 8
    (out / 'selected_figures.json').write_text(json.dumps(figure_records, indent=2, ensure_ascii=False), encoding='utf-8')
    # Portable Markdown numeric tables; no optional tabulate dependency.
    md = ['# Stage 07: saved-result tables', '',
          'CSV files retain full precision. Display values below use six decimals.',
          'Seed SD is across 10 KMeans seeds; probe scores are single held-out evaluations.',
          'N05 quantiles are descriptive fixed-label/model app-composition dispersion, not population CIs.',
          'Provenance partitions are restricted, never refitted. Missing probe values mean not evaluated.', '']
    for name, frame in tables.items():
        md += ['## ' + name, '', '| ' + ' | '.join(frame.columns) + ' |',
               '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
        for values in frame.itertuples(index=False, name=None):
            def fmt(x):
                if pd.isna(x):
                    return 'NA'
                return (f'{x:.6f}' if isinstance(x, (float, np.floating)) else str(x)).replace('|', '/')
            md.append('| ' + ' | '.join(map(fmt, values)) + ' |')
        md.append('')
    (out / 'tables.md').write_text('\n'.join(md), encoding='utf-8')
    manifest = {'status': 'STAGE07_SAVED_RESULT_SYNTHESIS_COMPLETE', 'no_inference_or_refitting': True,
                'source_code_sha256': sha(Path(__file__)), 'inputs_sha256': inputs,
                'table_rows': {k: len(v) for k, v in tables.items()},
                'selected_figures': [r['slug'] for r in figure_records],
                'outputs_sha256': {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(out), 'tables': manifest['table_rows'], 'figures': len(figure_records)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    main(parser.parse_args().run_id)
