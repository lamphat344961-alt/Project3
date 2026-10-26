"""Generate manuscript numbers/tables and copy the eight accepted figure PDFs.

Reads saved Stage 07 artifacts only; never fits or changes an experiment.
Run: python -B -m pipeline.manuscript_stage08
"""
from pathlib import Path
import hashlib
import json
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/stage07/synthesis_20261026_003'
PAPER = ROOT / 'paper'
NAMES = {
    'tfidf_lsa': 'TF-IDF--LSA', 'phobert_base_v2': 'PhoBERT-v2',
    'sup_simcse_phobert': 'Sup-SimCSE', 'bkai_vietnamese_biencoder': 'BKAI',
    'labse': 'LaBSE', 'multilingual_e5_large': 'E5-large', 'bge_m3': 'BGE-M3',
    'qwen3_embedding_0_6b': 'Qwen3-0.6B',
}
ORDER = list(NAMES)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((SOURCE / 'manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest['outputs_sha256'].items():
        assert sha(SOURCE / name) == digest, name
    generated = PAPER / 'generated'
    figures = PAPER / 'figures'
    generated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)
    data = {}

    def read(name):
        data[name] = pd.read_csv(SOURCE / (name + '.csv'))
        return data[name]

    tax = read('05_rq2_taxonomy_probe')
    ret = read('03_rq1_correspondence_retrieval')
    part = read('04_rq1_partitions_taxonomy')
    provenance = read('07_rq2_provenance')
    scope = read('02_recipe_scope')
    scope[['experiment', 'representation', 'language', 'condition', 'revision', 'dimension', 'k', 'n_init', 'seeds']].to_csv(
        generated / 'recipe_revisions.csv', index=False)
    contrasts = read('06_input_language_contrasts')
    ranks = read('08_taxonomy_rank_support')
    assert len(tax) == 34 and tax.test_macro_f1.notna().sum() == 26

    def write(name, value):
        (generated / (name + '.tex')).write_text(value + '\n', encoding='utf-8')

    def number(x):
        return '--' if pd.isna(x) else f'{x:.3f}'

    def mean_sd(row):
        return r'$' + number(row.ami_mean) + r'\pm' + number(row.ami_std) + '$'

    def table(name, headers, rows, alignment, multipage=False):
        environment = 'longtable' if multipage else 'tabular'
        lines = [r'\begin{' + environment + '}{' + alignment + '}', r'\toprule',
                 ' & '.join(headers) + r' \\', r'\midrule']
        if multipage:
            lines += [r'\endfirsthead', r'\toprule', ' & '.join(headers) + r' \\',
                      r'\midrule', r'\endhead']
        lines += [' & '.join(map(str, row)) + r' \\' for row in rows]
        lines += [r'\bottomrule', r'\end{' + environment + '}']
        write(name, '\n'.join(lines))

    rows = []
    for rep in ORDER:
        d = tax[tax.representation.eq(rep)].set_index('cell')
        char, native = d.loc['tn1:vi:char_budget'], d.loc['tn1:vi:native']
        rows.append([NAMES[rep], mean_sd(char), mean_sd(native),
                     number(char.test_macro_f1), number(native.test_macro_f1)])
    table('tn1', ['Recipe', 'AMI char950', 'AMI native', 'F1 char950', 'F1 native'], rows, 'lrrrr')
    pair_order = [(r, c) for r in ORDER for c in ('native', 'token_budget')
                  if ((part.representation == r) & (part.condition == c)).any()]
    retrieval_rows, vector_rows, partition_rows, taxonomy_rows = [], [], [], []
    macros = {}
    for rep, condition in pair_order:
        label = NAMES[rep] + (' (T512)' if condition == 'token_budget' else ' (N)')
        p = part[(part.representation == rep) & (part.condition == condition)].iloc[0]
        r = ret[(ret.representation == rep) & (ret.condition == condition)].set_index('direction')
        en, vi = r.loc['EN_TO_VI'], r.loc['VI_TO_EN']
        retrieval_rows.append([label] + [number(row[key]) for row in (en, vi) for key in
            ['recall_at_1_primary', 'recall_at_10_supplementary', 'mrr_supplementary']])
        vector_rows.append([label] + [number(en[key]) for key in ['matched_cosine', 'negative_cosine_median', 'delta_cosine']])
        partition_rows.append([label] + [number(p[key]) for key in ['C', 'W_EN', 'W_VI', 'S', 'q025', 'q975']])
        taxonomy_rows.append([label] + [number(p[key]) for key in ['en_ami_mean', 'vi_ami_mean', 'en_minus_vi_ami']])
        if rep == 'labse':
            macros.update(LabRoneEN=number(en.recall_at_1_primary), LabRoneVI=number(vi.recall_at_1_primary),
                          LabMRREN=number(en.mrr_supplementary), LabMRRVI=number(vi.mrr_supplementary),
                          LabC=number(p.C), LabS=number(p.S))
    table('retrieval', ['Recipe/input', r'R1$^{\to}$', r'R10$^{\to}$', r'MRR$^{\to}$',
          r'R1$^{\leftarrow}$', r'R10$^{\leftarrow}$', r'MRR$^{\leftarrow}$'], retrieval_rows, 'lrrrrrr')
    table('correspondence', ['Recipe/input', 'Cos cùng app', 'Nền khác app', r'$\overline{\Delta\cos}$'], vector_rows, 'lrrr')
    table('partition', ['Recipe/input', '$C$', '$W_{EN}$', '$W_{VI}$', '$S$', '$Q_{.025}$', '$Q_{.975}$'], partition_rows, 'lrrrrrr')
    table('taxonomy_tn2', ['Recipe/input', 'AMI EN', 'AMI VI', 'EN trừ VI'], taxonomy_rows, 'lrrr')
    rows = []
    for rep in ORDER:
        for cond in ('char_budget', 'native'):
            d = provenance[(provenance.representation == rep) &
                           (provenance.cell == 'tn1:vi:' + cond)].set_index('provenance')
            rows.append([NAMES[rep], '950' if cond == 'char_budget' else 'Native',
                number(d.loc['translation-flagged', 'ami_mean']), number(d.loc['not-flagged', 'ami_mean'])])
    table('provenance', ['Recipe', 'Input', 'Có cờ dịch', 'Không gắn cờ'], rows, 'llrr')
    rows = []
    for rep in ORDER:
        for c in ['tn1:vi:char_budget', 'tn1:vi:native', 'tn2:en:native', 'tn2:vi:native', 'tn2:en:token_budget', 'tn2:vi:token_budget']:
            d = tax[(tax.representation == rep) & (tax.cell == c)]
            if d.empty:
                continue
            r = d.iloc[0]
            cell = c.replace('char_budget', '950').replace('token_budget', 'T512').replace('native', 'N')
            rows.append([NAMES[rep], cell, mean_sd(r), number(r.ari_mean), number(r.test_macro_f1)])
    table('all_results', ['Recipe', 'Cell', 'AMI', 'ARI', 'Macro-F1'], rows, 'llrrr', multipage=True)
    macros.update(Smin=number(part.S.min()), Smax=number(part.S.max()))
    for rep, prefix in [('multilingual_e5_large', 'Efive'), ('bge_m3', 'Bge'), ('qwen3_embedding_0_6b', 'Qwen')]:
        for c, suffix in [('char_budget', 'Char'), ('native', 'Native')]:
            row = tax[(tax.representation == rep) & (tax.cell == 'tn1:vi:' + c)].iloc[0]
            macros[prefix + suffix] = number(row.ami_mean)
            macros[prefix + suffix + 'F'] = number(row.test_macro_f1)
    write('numbers', '\n'.join('\\newcommand{\\' + key + '}{' + value + '}' for key, value in macros.items()))
    for slug in manifest['selected_figures']:
        source = SOURCE / 'figures' / (slug + '.pdf')
        target = figures / source.name
        if target.exists() and sha(target) != sha(source):
            raise RuntimeError('Refuse to overwrite edited figure: ' + str(target))
        shutil.copyfile(source, target)
    result = {'source_bundle': SOURCE.relative_to(ROOT).as_posix(), 'source_manifest_sha256': sha(SOURCE / 'manifest.json'),
              'code_sha256': sha(Path(__file__)), 'no_fit_or_inference': True,
              'generated_sha256': {p.relative_to(PAPER).as_posix(): sha(p) for folder in (generated, figures) for p in folder.iterdir()},
              'counts': {'clustering_cells': len(tax), 'probe_cells': int(tax.test_macro_f1.notna().sum()), 'paired_conditions': len(part), 'figures': 8}}
    (PAPER / 'result_sources.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['counts']))


if __name__ == '__main__':
    main()
