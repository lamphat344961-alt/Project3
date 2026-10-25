"""Build an executed, Vietnamese, figure-by-figure Stage07 review notebook.

Only reads the existing gallery and tabular results. Never infers or refits.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import nbformat as nbf
from nbclient import NotebookClient
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / 'results/stage07/gallery_20261025_004'
NAMES = {'tfidf_lsa':'TF-IDF–LSA', 'labse':'LaBSE', 'multilingual_e5_large':'E5-large',
         'bge_m3':'BGE-M3','qwen3_embedding_0_6b':'Qwen3-Embedding-0.6B',
         'phobert_base_v2':'PhoBERT-base-v2', 'sup_simcse_phobert':'Sup-SimCSE-PhoBERT',
         'bkai_vietnamese_biencoder':'BKAI biencoder'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def describe(item, tables):
    """Each returned observation is calculated from the figure's actual tables."""
    slug = item['id']
    tax,probe,part,ret,length,contr,ranks = [tables[n] for n in
        ['taxonomy_summary','probe_taxonomy_comparison','partition_all','retrieval_all',
         'character_length_summary','paired_descriptive_contrasts','taxonomy_rank_stability']]
    if slug.startswith('01'):
        rq='Bối cảnh phương pháp cho RQ1 và RQ2; không phải bằng chứng hiệu năng.'
        question='Dữ liệu nào được dùng cho từng phép đánh giá và hai cohort liên hệ thế nào?'
        reading='Đọc các khối và mũi tên từ trên xuống: thu thập → làm sạch → cohort → recipes → các nhiệm vụ đánh giá.'
        observed=['TN1 có 23.883 app tiếng Việt; TN2 có 13.657 app có cặp EN–VI và nằm trong TN1.',
                  'Bộ kết quả có 34 clustering cells, 26 probe cells và 9 paired conditions. Các con số chỉ phạm vi thực hiện, không phải 34 mô hình khác nhau.']
        meaning='Giúp người đọc phân biệt phép đánh giá trên catalogue VI (RQ2) với so sánh hai view của cùng app (RQ1), tránh so các cohort khác nhau như cùng một thí nghiệm.'
        limit='Sơ đồ không chứng minh recipe tốt hơn hoặc ngôn ngữ gây thay đổi. Không có số lượng raw/funnel lịch sử chưa được xác minh; cờ dịch không chứng minh human authorship.'
        suggestion=('Bản đầy đủ phù hợp giải thích Methods hoặc phụ lục; đoạn chú giải provenance chi tiết cần cân nhắc khi thu nhỏ.' if slug.startswith('01a') else
                    'Bản gọn là ứng viên cho Methods khi chỗ hạn chế; so với 01a để xem phần provenance có cần giữ trên hình không.')
        alternatives='01a_pipeline_full ↔ 01b_pipeline_compact'
        sources=['data_v1/benchmark_v1.parquet','data_v1/paired_en.parquet','results/stage05/combined_20261025_001/execution_receipt.json','research_stages/stage_01/codex.md §10']
    elif slug.startswith('02'):
        counts=pd.read_csv(GALLERY/'data/genre_TN1_all.csv'); counts=counts.sort_values('n_apps',ascending=False)
        total=int(counts.n_apps.sum()); share=counts.n_apps.iloc[:10].sum()/total
        l=length.set_index('view').loc['TN1 VI']
        rq='Bối cảnh dữ liệu cho RQ1/RQ2; genre/provenance và input length đặc biệt liên quan RQ2.'
        question='Catalogue có cân bằng genre không, provenance phân bố thế nào và char950 tác động tới bao nhiêu văn bản?'
        reading='(a) Số app theo genre: xanh TN1, cam TN2; TN2 là subset. (b) Tỷ lệ theo cờ dịch. (c) ECDF: trục x là số ký tự log scale, y là tỷ lệ văn bản có độ dài ≤ x.'
        observed=[f'Genre đông nhất TN1 là {counts.iloc[0].genre}, {int(counts.iloc[0].n_apps):,} app; top10 chiếm {share:.1%} TN1.',
                  'TN1 có 16.394 translation-flagged, 7.466 not-flagged và 23 unknown; TN2 hoàn toàn translation-flagged.',
                  f'Median độ dài mô tả sạch TN1 VI là {l.q50:,.0f} ký tự; {l.fraction_gt_950:.1%} dài hơn 950 ký tự. Đây là tỷ lệ bị cắt theo số ký tự nếu áp dụng char950, không phải token-truncation rate.']
        meaning='Đặt kết quả taxonomy trong bối cảnh class imbalance và mức thay đổi input. TN2 chỉ đại diện nhánh paired translation-flagged đã lọc, không đại diện mọi mô tả VI.'
        limit='Counts không đo hiệu năng theo provenance và không xác định hiệu ứng dịch. “Other” là gộp để hiển thị, không đổi49 nhãn gốc; token cap còn phụ thuộc tokenizer.'
        suggestion=('Bản49 genres thích hợp phụ lục hoặc xem lớn; cần zoom để đọc đủ tên.' if slug.startswith('02a') else
                    'Bản top10+other gọn hơn cho bài chính, nhưng Other gồm39 genres nên không xem như một genre lớn nhất mới.')
        alternatives='02a_benchmark_all49 ↔ 02b_benchmark_top10'
        sources=['gallery_20261025_004/data/genre_TN1_all.csv','gallery_20261025_004/data/character_length_summary.csv','data_v1/*.parquet']
    elif slug.startswith('03'):
        rq='RQ2 trực tiếp; các panel TN2 EN/VI của03a bổ sung taxonomy alignment theo ngôn ngữ cho RQ1.'
        question='Recipe nào phù hợp taxonomy hơn trong từng condition và việc đổi input có giữ nguyên thứ hạng không?'
        reading=('03a: mỗi chấm xanh là một KMeans seed42–51; hình thoi cam là mean. Các panel dùng cùng trục AMI; “not evaluated” không phải0.' if slug.startswith('03a') else
                 '03b: cột là AMI trung bình10 seeds của8 recipes; hai panel TN1 char950/native. Trục bắt đầu0, không có dispersion được thể hiện bằng cột.')
        observed=[]
        for cell,label in [('tn1:vi:char_budget','TN1 char950'),('tn1:vi:native','TN1 native')]:
            d=tax[tax.cell==cell].sort_values('ami_mean',ascending=False)
            a,b=d.iloc[:2].itertuples(index=False)
            observed.append(f'{label}: {NAMES[a.representation]} đứng đầu theo mean AMI ({a.ami_mean:.3f}); {NAMES[b.representation]} tiếp theo ({b.ami_mean:.3f}); chênh mean chưa làm tròn là {a.ami_mean-b.ami_mean:.5f}.')
        meaning='Hình hỗ trợ finding rằng thứ hạng gắn với recipe và input condition trong benchmark này. Dots cho biết biến thiên do khởi tạo; bars cho tổng quan dễ đọc hơn.'
        limit='Không suy statistical superiority từ mean/range10 seeds; không khẳng định khác biệt nhỏ là đáng kể hay các mô hình tương đương. Historical precision và các input recipes không hoàn toàn giống nhau.'
        suggestion=('Ứng viên hình kết quả chính vì giữ được observations và các cells không có dữ liệu.' if slug.startswith('03a') else
                    'Ứng viên tổng quan TN1 nếu ưu tiên đơn giản; nên có03a hoặc bảng seed summary bổ trợ để không che biến thiên.')
        alternatives='03a_taxonomy_seed_dots ↔ 03b_taxonomy_bars'
        sources=['gallery_20261025_004/data/taxonomy_summary.csv','gallery_20261025_004/data/taxonomy_all_by_seed.csv']
    elif slug.startswith('04'):
        rq='RQ1 trực tiếp: partition agreement và pair retrieval là hai cấp độ đánh giá khác nhau.'
        question='Tìm đúng app đối ứng EN–VI có đồng nghĩa hai phân hoạch giống nhau tới mức biến thiên nội ngôn ngữ không?'
        reading='C là mean AMI100 cặp seed EN–VI; W_EN/W_VI là mean45 cặp seed nội ngôn ngữ; S=C−(W_EN+W_VI)/2. Retrieval đo Recall@1/@10/MRR hai chiều, với13.657 queries và13.657 candidates mỗi chiều.'
        p=part[(part.representation=='labse')&(part.condition=='native')].iloc[0]
        r=ret[(ret.representation=='labse')&(ret.condition=='native')].set_index('direction')
        observed=[f'Trong9 conditions, S đều âm: từ {part.S.min():.3f} đến {part.S.max():.3f}.',
                  f'LaBSE native: C={p.C:.3f}, W_EN={p.W_EN:.3f}, W_VI={p.W_VI:.3f}, S={p.S:.3f}.',
                  f'LaBSE native Recall@1: EN→VI={r.loc["EN_TO_VI","recall_at_1_primary"]:.3f}, VI→EN={r.loc["VI_TO_EN","recall_at_1_primary"]:.3f}.']
        if slug=='04b_partition_only':observed=observed[:2]
        if slug in ('04c_retrieval_dots','04d_retrieval_bars'):observed=observed[2:]
        meaning='Kết quả cho phép mô tả retrieval cao cùng tồn tại với seed-relative partition gap. Đây là lý do đọc cả hai loại phép đo; không đủ để khẳng định quan hệ proxy tổng quát.'
        limit='Không gọi S là translation loss. Các seed pairs dùng chung partitions nên không độc lập; không tự gắn error bars từ100 pairs. Retrieval gần1 có thể che các khác biệt nhỏ trên trục0–1.'
        if slug=='04a_partition_retrieval_combined':suggestion='Bản tổng quan đặt hai cấp độ cạnh nhau; hợp xem lớn nhưng dễ quá rộng cho paper.'
        elif slug=='04b_partition_only':suggestion='Ứng viên hình riêng cho partition agreement trong bài chính; kết hợp04c hoặc04d ở hình khác/bảng.'
        elif slug=='04c_retrieval_dots':suggestion='Bản dots ít chi tiết trang trí; dùng khi cần xem cả hai chiều và9 điều kiện.'
        else:suggestion='Bản grouped bars giúp so chiều truy hồi quen thuộc; xem dots nếu các cột gần1 khó phân biệt.'
        alternatives='04a gộp ↔ 04b +04c/04d tách; 04c dots ↔ 04d bars'
        sources=['gallery_20261025_004/data/partition_all.csv','gallery_20261025_004/data/retrieval_all.csv']
    elif slug.startswith('05'):
        rq='RQ2 trực tiếp; các view EN/VI trên TN2 bổ sung taxonomy predictability cho RQ1.'
        question='Thứ hạng phù hợp taxonomy từ clustering có giống khả năng dự đoán genre từ frozen embeddings không?'
        reading='Macro-F1 là một kết quả held-out trên fixed split/model mỗi cell; AMI là mean10 seeds.05a so score/rank;05b đặt AMI và Macro-F1 trên hai trục;05c tô màu điểm thực với hai color scales riêng.'
        d=probe[probe.cell=='tn1:vi:native'].sort_values('test_macro_f1',ascending=False)
        a,b=d.iloc[:2].itertuples(index=False)
        top=tax[tax.cell=='tn1:vi:native'].sort_values('ami_mean',ascending=False).iloc[0]
        observed=[f'TN1 native, probe: {NAMES[a.representation]}={a.test_macro_f1:.3f}, {NAMES[b.representation]}={b.test_macro_f1:.3f}; chênh chưa làm tròn={a.test_macro_f1-b.test_macro_f1:.5f}.',
                  f'Cùng condition đó, recipe đứng đầu mean AMI là {NAMES[top.representation]} ({top.ami_mean:.3f}), khác recipe đứng đầu probe.']
        meaning='Hỗ trợ phân biệt cấu trúc cụm không giám sát và khả năng tuyến tính dự đoán nhãn. Đánh giá recipe cần nói rõ metric/task, không chỉ một bảng hạng chung.'
        limit='Không có10 probe seeds hoặc CI cho Macro-F1. TF-IDF clustering là transductive fit, probe là train-only fit; các điểm ghép theo family không phải cùng ma trận. Không suy tương quan dân số từ vài recipes.'
        suggestion={'05a_probe_rank_heatmap':'Bản score+rank cho tra cứu thứ hạng nhưng rank che độ lớn chênh lệch; giữ panel điểm thực.',
                    '05b_probe_ami_scatter':'Ứng viên bài chính nếu muốn thấy magnitude của cả hai metric; các leader lines chỉ nối nhãn với điểm, không biểu thị xu hướng.',
                    '05c_probe_ami_score_heatmaps':'Bản bảng màu phù hợp tra cứu nhiều view; không so màu AMI và Macro-F1 như cùng thang.'}[slug]
        alternatives='05a rank ↔ 05b scatter ↔ 05c score heatmaps'
        sources=['gallery_20261025_004/data/probe_taxonomy_comparison.csv','gallery_20261025_004/data/probe_all.csv','gallery_20261025_004/data/taxonomy_summary.csv']
    elif slug.startswith('06'):
        rq='Bổ trợ độ ổn định mô tả của các kết quả RQ1/RQ2 trong catalogue cố định.'
        question='Những contrast/thứ hạng nào giữ được khi thay đổi thành phần app của catalogue?'
        reading='N05:200 subsets/cohort, lấy80% không hoàn lại, giữ labels/models.06a raincloud cho shape+draws;06b dot-range;06c violin; cam rỗng là full-catalogue point, xanh là median/range.06d là tie-adjusted top share.'
        e=contr[(contr.representation=='multilingual_e5_large')&(contr.contrast=='char950_minus_native')&(contr.metric=='ami')].iloc[0]
        d=ranks[(ranks.experiment=='tn1')&(ranks.condition=='native')].set_index('representation')
        observed=[f'E5 TN1 char950−native: point={e.point:+.5f}; Q2.5–Q97.5=[{e.q025:+.5f}, {e.q975:+.5f}].',
                  f'TN1 native AMI: top share BGE-M3={d.loc["bge_m3","share_draws_top_tie_adjusted"]:.1%}, Qwen3={d.loc["qwen3_embedding_0_6b","share_draws_top_tie_adjusted"]:.1%}.']
        if slug=='06d_top_rank_share':observed=observed[1:]
        else:observed=observed[:1]+['Panel S ở06a–06c chỉ dùng native conditions; toàn bộ200 draws được lấy từ outputs đã lưu, không refit clustering.']
        meaning='Cho biết kết luận mô tả/ranking nhạy với app composition tới đâu, thay vì nhìn duy nhất full-catalogue mean. Không đồng nhất seed variation ở03a với subset variation ở đây.'
        limit='Khoảng Q2.5–Q97.5 không population CI; không p-values/significance/equivalence. Top share không probability of superiority. Probe conditional on fitted models/folds; retrieval nếu được bàn tới vẫn giữ full candidate catalogue.'
        suggestion={'06a_raincloud':'Giữ khi muốn quan sát shape/draws; ứng viên phụ lục vì các phân phối khá hẹp.',
                    '06b_dot_range':'Ứng viên bài chính cho contrast/range gọn; caption phải giải thích range mô tả.',
                    '06c_violin':'Phương án thay raincloud khi ưu tiên shape và bớt chấm; density smoothing chỉ là cách vẽ.',
                    '06d_top_rank_share':'Hình bổ trợ ranking, không thay hình điểm số/chênh lệch;0% top share không có nghĩa model có score0.'}[slug]
        alternatives='06a raincloud ↔ 06b dot-range ↔ 06c violin;06d bổ trợ ranking'
        sources=['gallery_20261025_004/data/paired_descriptive_contrasts.csv','gallery_20261025_004/data/taxonomy_rank_stability.csv','gallery_20261025_004/data/n05_S_draws.csv','gallery_20261025_004/data/n05_input_contrast_draws.csv']
    else:
        require_prefix='07_heatmap_';need(slug.startswith(require_prefix),'Unknown figure')
        tail=slug[len(require_prefix):]
        cond='token_budget' if tail.endswith('_token_budget') else 'native'
        rep=tail[:-(len(cond)+1)]
        a=pd.read_csv(GALLERY/'data'/f'{slug}_genre_en.csv',index_col=0)
        b=pd.read_csv(GALLERY/'data'/f'{slug}_genre_vi.csv',index_col=0)
        c=pd.read_csv(GALLERY/'data'/f'{slug}_en_vi.csv',index_col=0)
        need(a.index.tolist()==b.index.tolist(),'Genre row order')
        need(int(a.to_numpy().sum())==int(b.to_numpy().sum())==int(c.to_numpy().sum())==13657,'Heatmap supports')
        support=a.sum(axis=1); pa=a.max(axis=1)/support;pb=b.max(axis=1)/support
        change=(pb-pa).abs().sort_values(ascending=False,kind='stable')
        genre=change.index[0]; n=int(support.loc[genre])
        count=int(c.to_numpy().max());i,j=np.unravel_index(c.to_numpy().argmax(),c.shape)
        p=part[(part.representation==rep)&(part.condition==cond)].iloc[0]
        rq=f'RQ1: minh họa EN–VI partition/genre alignment cho {NAMES[rep]} · {cond}; bổ trợ taxonomy context của RQ2.'
        question='App được chia/gộp giữa các cụm EN/VI ra sao, và từng genre phân tán vào các cụm thế nào?'
        reading='(a) EN cluster × VI cluster: màu là app count theo log scale, ô trắng là0. Hungarian chỉ sắp lại cột để hiển thị. (b,c) Hàng=genre, cột=cluster; mỗi hàng chuẩn hóa tổng1, chung scale0–1. Cột không phải nhãn genre.'
        observed=[f'Cả ba bảng có13.657 app; ô chung nhiều app nhất trong (a) là EN cluster{i} × VI cluster{j}: {count:,} app (IDs gốc, không phải vị trí cột sau sắp xếp).',
                  f'Trong49 genres, {genre} (n={n}) có |Δ độ tập trung vào cụm lớn nhất| cao nhất: EN={pa.loc[genre]:.1%}, VI={pb.loc[genre]:.1%}. Tiêu chí minh họa hậu nghiệm là max |max-cluster fraction VI − EN|, ties theo thứ tự genre; phải đọc cùng n.',
                  f'Summary trên10 seeds cho đúng recipe/condition: C={p.C:.3f}, S={p.S:.3f}; heatmap riêng này chỉ minh họa seed42.']
        meaning='Cho một ví dụ cụ thể về cấu trúc bị che bởi scalar summary: một genre có thể tập trung vào một cụm hoặc trải trên nhiều cụm. So EN/VI ở cùng genre làm rõ câu hỏi partition/genre alignment của RQ1.'
        limit='Cluster ID không có ý nghĩa ngữ nghĩa chung giữa các view. Tỷ lệ tập trung không phải AMI hoặc accuracy, không chứng minh genre bảo toàn hay dịch gây biến đổi. Genre có n nhỏ cần thận trọng; ví dụ chọn hậu nghiệm không đại diện toàn catalogue.'
        suggestion=f'Giữ trong bộ so sánh đủ9 conditions; nếu chọn cho bài chính cần giải thích tính đại diện. {NAMES[rep]} · {cond} chưa được ưu tiên chỉ vì hình đẹp hoặc score cao. Bảng49×49 cần xem lớn/PDF vector.'
        alternatives='So native/T512 trong cùng recipe khi có; so các recipes trên cùng condition. LaBSE chỉ có native trong bộ paired này.'
        sources=[f'gallery_20261025_004/data/{slug}_{suffix}.csv' for suffix in ('en_vi','genre_en','genre_vi')]+['gallery_20261025_004/data/partition_all.csv']
    return dict(rq=rq,question=question,reading=reading,observed=observed,meaning=meaning,
                limit=limit,suggestion=suggestion,alternatives=alternatives,sources=sources)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default='gallery_20261025_005')
    args=parser.parse_args()
    need(all(c.isalnum() or c in '_-' for c in args.run_id),'Invalid run ID')
    out=ROOT/'results/stage07'/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((GALLERY/'manifest.json').read_text(encoding='utf-8'))
    need(manifest['n_figures']==26,'Expected 26 figures')
    # Check the immutable previous gallery, without editing any source artifacts.
    for name,digest in manifest['outputs_sha256'].items():
        need(sha(GALLERY/name)==digest,'Gallery source changed: '+name)
    used=['taxonomy_summary','probe_taxonomy_comparison','partition_all','retrieval_all',
          'character_length_summary','paired_descriptive_contrasts','taxonomy_rank_stability']
    tables={name:pd.read_csv(GALLERY/'data'/f'{name}.csv') for name in used}
    cells=[nbf.v4.new_markdown_cell('''# PAAC-VN · Chặng07 · Duyệt hình theo RQ

**26 hình, từng cell riêng, giữ cả ý tưởng Codex và Claude.** Đây là notebook xem/duyệt, chưa chốt hình cho paper.

- Mở trong VS Code hoặc Jupyter và xem output đã lưu: **không cần Run All**. Mỗi ảnh đã được nhúng trong notebook.
- Nếu muốn chạy lại cell hiển thị ảnh: giữ notebook trong repository và dùng Python có IPython (kernel `.venv` đã chuẩn bị). Mỗi cell tự tìm folder ảnh, không cần chạy cell setup trước.
- Sửa Markdown **Ghi chú duyệt của tác giả** dưới từng hình: giữ/chỉnh/phụ lục/chưa chọn; lưu notebook để giữ ghi chú. Code không tự xuất file ghi chú hoặc sửa kết quả.
- So các phương án cùng nhóm theo cách thể hiện; không phải26 thí nghiệm mới. Mọi đề xuất main/phụ lục ở dưới đều chờ tác giả quyết định.

**RQ1:** Tương ứng giữa các mô tả EN–VI cùng app: vector correspondence, retrieval, partition agreement và taxonomy alignment theo ngôn ngữ.

**RQ2:** Đánh giá representation/input recipes với taxonomy Google Play trên benchmark VI, xét input conditions/provenance.

**Phạm vi:** Bộ hình hiện có chưa có panel riêng cho vector cosine correspondence hoặc hiệu năng theo provenance stratum. Thành phần cờ dịch không thay thế phân tích hiệu năng provenance. R02/back-translation đã bỏ; không diễn giải causal translation loss. Chặng08 chưa mở.
''')]
    toc=['## Danh mục hình\n','| Mã | Nguồn ý tưởng | Tên |','|---|---|---|']
    for item in manifest['figures']:
        toc.append(f'| [{item["id"]}](#{item["id"]}) | {item["origin"]} | {item["title"]} |')
    cells.append(nbf.v4.new_markdown_cell('\n'.join(toc)))
    annotations={}
    for item in manifest['figures']:
        slug=item['id'];a=describe(item,tables);annotations[slug]=a
        observations='\n'.join('- '+x for x in a['observed'])
        text=f'''<a id="{slug}"></a>
## {slug} — {item['title']}

**Nguồn ý tưởng:** {item['origin']} · **Vai trò với RQ:** {a['rq']}

**1. Câu hỏi hình giúp trả lời:** {a['question']}

**2. Cách đọc:** {a['reading']}

**3. Quan sát từ dữ liệu đã có:**

{observations}

**4. Gợi ý ý nghĩa/giá trị với RQ:** {a['meaning']}

**5. Giới hạn diễn giải:** {a['limit']}

**6. Gợi ý sử dụng và đối chiếu:** {a['suggestion']} **Biến thể:** {a['alternatives']}

**Nguồn số liệu:** {', '.join('`'+s+'`' for s in a['sources'])}.

[PDF vector](../gallery_20261025_004/{slug}.pdf) · [SVG](../gallery_20261025_004/{slug}.svg). Cell kế tiếp hiển thị đúng ảnh đã dựng, không tính lại metric.
'''
        cells.append(nbf.v4.new_markdown_cell(text,metadata={'tags':['figure-guide',slug]}))
        digest=sha(GALLERY/f'{slug}.png')
        code=f'''# Chỉ đọc/hiển thị một ảnh; cell này có thể chạy độc lập.
from pathlib import Path
import hashlib
from IPython.display import Image, display

_figure_id = {slug!r}
_candidates = [candidate for parent in [Path.cwd(), *Path.cwd().parents]
               for candidate in (parent / 'results/stage07/gallery_20261025_004',
                                 parent / 'gallery_20261025_004')]
_folder = next((p for p in _candidates if (p / 'manifest.json').is_file()), None)
if _folder is None:
    raise FileNotFoundError('Không tìm thấy gallery_20261025_004. Mở notebook trong repository; output ảnh đã lưu vẫn có thể xem mà không chạy cell.')
_image = _folder / (_figure_id + '.png')
assert hashlib.sha256(_image.read_bytes()).hexdigest() == {digest!r}, 'Ảnh nguồn đã thay đổi.'
display(Image(filename=str(_image), width=1400))
'''
        cells.append(nbf.v4.new_code_cell(code,metadata={'tags':['figure-display',slug]}))
        cells.append(nbf.v4.new_markdown_cell(f'''### Ghi chú duyệt của tác giả — `{slug}`

- **Quyết định:** Chưa chọn / Giữ / Chỉnh sửa / Phụ lục.
- **Điểm muốn giữ:** …
- **Điểm cần chỉnh:** …
- **Thông điệp muốn nhấn mạnh với RQ:** …
''',metadata={'tags':['author-notes',slug]}))
    cells.append(nbf.v4.new_markdown_cell('''## Sau khi duyệt

Gửi mã hình và nội dung chỉnh sửa hoặc lưu các ghi chú trên notebook. Chưa có lựa chọn cuối cùng được tự động áp dụng.
Việc xem hoặc chạy các cell này chỉ hiển thị ảnh đã có, không khởi chạy inference, KMeans, probe hoặc N05.
'''))
    source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in [GALLERY/'manifest.json',Path(__file__)]}
    for p in (GALLERY/'data').glob('*.csv'):
        source_hashes[p.relative_to(ROOT).as_posix()]=sha(p)
    nb=nbf.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3 (PAAC-VN review)','language':'python','name':'python3'},
        'language_info':{'name':'python','version':sys.version.split()[0]},
        'paac_vn':{'purpose':'Author review; existing figures only','source_gallery':GALLERY.relative_to(ROOT).as_posix(),
                   'inputs_sha256':source_hashes,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                   'n_figures':26,'annotations':annotations,'selection_status':'PENDING_AUTHOR',
                   'packages':{p:importlib.metadata.version(p) for p in ('nbformat','nbclient','ipykernel','numpy','pandas')}}})
    nbf.validate(nb)
    client=NotebookClient(nb,timeout=60,resources={'metadata':{'path':str(ROOT)}},kernel_name='python3')
    client.create_kernel_manager()
    client.km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    client.execute()
    need(len([c for c in nb.cells if c.cell_type=='code'])==26,'26 display cells')
    for cell in nb.cells:
        if cell.cell_type=='code':
            need(cell.execution_count is not None,'Cell not executed')
            need(len(cell.outputs)==1 and cell.outputs[0].output_type=='display_data','Expected one display per cell')
            need('image/png' in cell.outputs[0].data,'PNG not embedded')
    nb.metadata.paac_vn['execution_status']='ALL_26_DISPLAY_CELLS_EXECUTED'
    nbf.validate(nb)
    path=out/'PAAC_VN_Stage07_Review.ipynb'
    nbf.write(nb,path)
    print(json.dumps({'path':str(path),'cells':len(nb.cells),'executed_figure_cells':26,
                      'mb':round(path.stat().st_size/1e6,2)},ensure_ascii=True))


if __name__=='__main__':
    main()
