"""Apply the author's 01a–05c selection and specific notebook review notes.

Keeps author code/notes outside requested edits, drops groups06/07 from the
active notebook, separates length histograms and executes the retained figures.
No model inference, refitting or modifications to historical gallery artifacts.
"""
from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import nbformat as nbf
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'results/stage07/gallery_20261025_005/PAAC_VN_Stage07_Review.ipynb'
HISTORY=TARGET.with_name('PAAC_VN_Stage07_Review_before_selection.ipynb')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def cell_code(slug,body):
    return (f'# {slug}: code tạo hình; chạy setup trước, rồi chỉnh/chạy riêng cell này.\n'
            'def draw_this_figure():\n'+textwrap.indent(textwrap.dedent(body).strip(),'    ')
            +'\n\ndraw_this_figure()')


def main():
    original=TARGET.read_bytes();before=digest(original)
    nb=nbf.reads(original.decode('utf-8'),as_version=4)
    if nb.metadata.paac_vn.get('author_selection_applied'):
        raise RuntimeError('Selection already applied; preserve subsequent author edits.')
    groups={}
    for cell in nb.cells:
        tags=cell.metadata.get('tags',[])
        if any(t in tags for t in ('figure-guide','figure-drawing','author-notes')):
            groups.setdefault(tags[1],[]).append(cell)
    selected=[slug for slug in groups if slug[:2] in ('01','02','03','04','05')]
    if len(selected)!=13 or len(groups)!=26:
        raise RuntimeError('Unexpected source selection')
    author_notes={c.id:c.source for slug in selected for c in groups[slug] if 'author-notes' in c.metadata.tags}
    original_code={slug:next(c.source for c in groups[slug] if c.cell_type=='code') for slug in selected}
    compact=next(c for c in groups['01b_pipeline_compact'] if c.cell_type=='code')
    compact.source=cell_code('01b_pipeline_compact','''
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.set(xlim=(0, 14), ylim=(0, 3)); ax.axis('off')
        stages = [
            ('Google Play', '#e9f2fa'),
            ('Dataset preparation', '#e9f2fa'),
            ('TN1 / TN2', '#e9f2fa'),
            ('Text representations', '#e5f1eb'),
            ('Evaluation', '#e9f2fa'),
        ]
        for i, (label, color) in enumerate(stages):
            x = .2 + 2.8 * i
            ax.add_patch(FancyBboxPatch((x, 1), 2.3, .9,
                boxstyle='round,pad=0.08', facecolor=color, edgecolor='#738296'))
            ax.text(x+1.15, 1.45, label, ha='center', va='center', fontsize=11)
            if i < len(stages)-1:
                ax.add_patch(FancyArrowPatch((x+2.4, 1.45), (x+2.7, 1.45),
                    arrowstyle='-|>', mutation_scale=14, color='#596779'))
        show_figure(fig, '01b_pipeline_compact', 'Research workflow', 'Chung',
            'Overview of the study workflow. Cohort definitions, input recipes and evaluation details are described in Methods.',
            '1 · Quy trình')
    ''')
    guide=next(c for c in groups['01b_pipeline_compact'] if 'figure-guide' in c.metadata.tags)
    guide.source+='\n\n**Đã chỉnh theo ghi chú:** Rút từ các khối chi tiết xuống5 bước tổng quát; giữ màu xanh nhạt/xanh lục. Hình thể hiện luồng nghiên cứu, phần thao tác cụ thể chuyển sang Methods.'
    for slug,top10 in [('02a_benchmark_all49',False),('02b_benchmark_top10',True)]:
        cell=next(c for c in groups[slug] if c.cell_type=='code')
        # Preserve the already editable source of panels a/b; remove only panel c.
        source=cell.source
        source=source.replace("plt.subplots(1, 3, figsize=(17, 8 if top10 else 15), gridspec_kw={'width_ratios': [1.25, 1, 1]})",
                              "plt.subplots(1, 2, figsize=(14, 7 if top10 else 14), gridspec_kw={'width_ratios': [1.4, 1]})")
        start=source.index('    ax = axes[2]')
        stop=source.index('    show_figure(',start)
        source=source[:start]+source[stop:]
        source=source.replace('Lengths are before input clipping; characters are not model tokens. Counts include TN2 within TN1. Provenance indicates observed flags, not authorship.',
                              'TN2 is a subset of TN1; provenance indicates observed flags, not authorship. Description lengths are shown separately in 02c.')
        if "plt.subplots(1, 3" in source or 'axes[2]' in source:
            raise RuntimeError('Benchmark edit did not match the current author code')
        cell.source=source
        guide=next(c for c in groups[slug] if 'figure-guide' in c.metadata.tags)
        guide.source=guide.source.replace('(c) ECDF: trục x là số ký tự log scale, y là tỷ lệ văn bản có độ dài ≤ x.',
                                          'Độ dài văn bản được tách thành histogram riêng ở02c, không còn panel(c) trong hình này.')
        guide.source+='\n\n**Đã chỉnh theo ghi chú02a:** Hình hiện chỉ có(a) genre support và(b) provenance; giữ dữ liệu/colors. Xem02c cho độ dài văn bản dạng histogram trên trục ký tự tuyến tính.02b cũng dùng bố cục hai panel để không lặp lại ECDF khó đọc.'
    length_id='02c_text_length_histograms'
    length_code=cell_code(length_id,'''
        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, sharey=True)
        views = [
            ('tn1', 'description_clean', 'TN1 VI', BLUE),
            ('tn2', 'description_clean', 'TN2 VI', ORANGE),
            ('tn2', 'description_original_clean', 'TN2 EN', GREEN),
        ]
        max_length = max(int(frames[exp][field].str.len().max()) for exp, field, _, _ in views)
        bins = np.arange(0, (max_length // 250 + 2) * 250, 250)
        for ax, (exp, field, label, color) in zip(axes, views):
            lengths = frames[exp][field].str.len().to_numpy()
            weights = np.full(len(lengths), 100 / len(lengths))
            counts, _, _ = ax.hist(lengths, bins=bins, weights=weights,
                color=color, edgecolor='white', linewidth=.3, label=label)
            require(np.isclose(counts.sum(), 100), 'Histogram must include all cohort texts')
            ax.axvline(950, color='#555', linestyle='--', linewidth=1, label='950 characters')
            ax.set_ylabel('Apps (%) per bin')
            ax.legend(loc='upper right', fontsize=8)
            ax.text(.98, .58, f'n={len(lengths):,} · median={np.median(lengths):,.0f} chars',
                transform=ax.transAxes, ha='right', fontsize=9)
            panel(ax, label)
        axes[-1].set_xlabel('Clean description length (characters; 250-character bins)')
        axes[-1].set_xlim(0, bins[-1])
        show_figure(fig, '02c_text_length_histograms', 'Description lengths', 'Author revision',
            'All clean descriptions before input clipping; each panel totals 100%. Linear character axis, common bins/scales, including the full long tail.\\n'
            'The dashed line marks 950 characters, not a tokenizer limit; character clipping and token truncation are different.',
            '2 · Dữ liệu')
    ''')
    length_guide=nbf.v4.new_markdown_cell('''<a id="02c_text_length_histograms"></a>
## 02c_text_length_histograms — Độ dài mô tả, tách riêng theo ghi chú tác giả

**Vai trò RQ:** Bối cảnh input conditions cho RQ2 và paired views của RQ1; không phải phép đo hiệu năng.

**Cách đọc:** Ba hàng là TN1 VI, TN2 VI và TN2 EN. Trục ngang là số ký tự, cùng trục tuyến tính; mỗi cột ứng với khoảng250 ký tự, chiều cao là phần trăm app của từng cohort. Tổng mỗi hàng=100%; biểu diễn cả đuôi dài, không bỏ outliers. Đường đứt tại950 là ngưỡng char950.

**Quan sát:** Median lần lượt1.874,1.888 và1.670 ký tự. Khoảng78,8% TN1 VI dài hơn950 ký tự, cho thấy clipping theo ký tự tác động đến nhiều mô tả. Điều này không chứng minh clipping giúp/hại hiệu năng.

**Giá trị:** Cho thấy độ dài đầu vào theo cách trực tiếp hơn ECDF và giúp đọc các so sánh char950/native. Không suy ra số tokens hoặc hiệu ứng dịch từ độ dài ký tự.

**Nguồn:** `data_v1/benchmark_v1.parquet`, `data_v1/paired_en.parquet`. Hình được tách từ panel(c) của02a; không thêm thí nghiệm. Có thể chỉnh bins/figsize trong cell dưới, không thay dataset.
''',metadata={'tags':['figure-guide',length_id]})
    length_notes=nbf.v4.new_markdown_cell('''### Ghi chú duyệt của tác giả — `02c_text_length_histograms`

- **Quyết định:** Chờ duyệt bản histogram được tách riêng.
- **Điểm muốn giữ:** …
- **Điểm cần chỉnh:** …
- **Thông điệp muốn nhấn mạnh với RQ:** …
''',metadata={'tags':['author-notes',length_id]})
    groups[length_id]=[length_guide,nbf.v4.new_code_cell(length_code,metadata={'tags':['figure-drawing',length_id]}),length_notes]
    order=selected.copy();order.insert(order.index('02b_benchmark_top10')+1,length_id)
    setup=next(c for c in nb.cells if 'plot-setup' in c.metadata.get('tags',[]))
    setup.source='\n'.join(line for line in setup.source.splitlines()
                            if not line.startswith(('index=g.read_csv','contrast=g.read_csv','receipt6=json.loads','S6=ROOT/',
                                                    'from matplotlib.colors import LogNorm','from scipy.optimize import linear_sum_assignment')))
    header=nb.cells[0]
    header.source=header.source.replace('**26 cell tạo hình Matplotlib trực tiếp**','**14 cell tạo hình Matplotlib trực tiếp**')
    header.source+='\n\n**Lựa chọn tác giả26/10:** Giữ13 hình01a–05c theo tin nhắn mới nhất (ghi chú “chưa chọn” cũ được giữ làm lịch sử), thêm02c để tách độ dài văn bản. Đã xóa toàn bộ nhóm06/07 khỏi bản duyệt này.01a chỉ ghi “Chỉnh sửa” nhưng chưa có hướng dẫn cụ thể nên giữ nguyên hình; không tự suy ra yêu cầu. Bản26 hình kèm ghi chú trước lựa chọn được lưu riêng để bảo toàn.'
    toc=nbf.v4.new_markdown_cell('## Danh mục sau lựa chọn\n\n'+ '\n'.join(
        f'- [{slug}](#{slug})' for slug in order))
    footer=nbf.v4.new_markdown_cell('''## Tiếp tục duyệt

Đã áp dụng lựa chọn01a–05c. Các ghi chú gốc được giữ nguyên; chỉ những yêu cầu có nội dung cụ thể được triển khai.02c là bản histogram mới tách ra, cần tác giả xem lại.
Sửa code/ghi chú rồi lưu notebook hoặc gửi mã hình muốn chỉnh. Chưa tự chốt hình main/phụ lục hoặc mở Chặng08.
''')
    nb.cells=[header,setup,toc,*[c for slug in order for c in groups[slug]],footer]
    for c in nb.cells:
        if c.cell_type=='code':
            c.outputs=[];c.execution_count=None
    meta=nb.metadata.paac_vn
    meta['n_figures']=len(order)
    meta['annotations']={k:v for k,v in meta.get('annotations',{}).items() if k in selected}
    for slug in ('01b_pipeline_compact','02a_benchmark_all49','02b_benchmark_top10'):
        meta['annotations'].pop(slug,None)  # Current guide Markdown supersedes older generated annotations.
    meta.update({'author_selection_applied':True,'selected_original_figures':selected,
                 'removed_from_active_notebook':[s for s in groups if s not in order],
                 'added_split_figure':length_id,'before_selection_sha256':before,
                 'before_selection_notebook':HISTORY.name,'selection_status':'AUTHOR_SELECTED_AWAITING_REVISION_REVIEW',
                 'revision_code_sha256':digest(Path(__file__).read_bytes()),
                 'revision_source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    nbf.validate(nb)
    client=NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(TARGET.parent)}})
    client.create_kernel_manager();client.km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    client.execute()
    checks=[]
    for c in nb.cells:
        if 'figure-drawing' in c.metadata.get('tags',[]):
            slug=c.metadata.tags[1]
            imgs=[o.data['image/png'] for o in c.outputs if o.output_type in ('display_data','execute_result') and 'image/png' in o.get('data',{})]
            if len(imgs)!=1 or any(o.output_type=='error' for o in c.outputs):raise RuntimeError('Figure output invalid: '+slug)
            checks.append({'id':slug,'png_sha256':digest(base64.b64decode(imgs[0]))})
    if {c.id:c.source for c in nb.cells if c.id in author_notes}!=author_notes:raise RuntimeError('Author notes changed')
    if any(slug.startswith(('06','07')) for slug in order):raise RuntimeError('Excluded figures retained')
    if digest(TARGET.read_bytes())!=before:raise RuntimeError('Notebook changed during edit; preserve concurrent edits')
    if HISTORY.exists():raise RuntimeError('Historical selection notebook already exists; preserve it')
    with HISTORY.open('xb') as handle:handle.write(original)
    meta['render_checks']=checks;meta['execution_status']='ALL_14_SELECTED_DRAWING_CELLS_EXECUTED'
    meta['render_mode']='INLINE_MATPLOTLIB_SCOPED_AUTHOR_SELECTION'
    meta['unchanged_author_code_ids']=[s for s in selected if s not in ('01b_pipeline_compact','02a_benchmark_all49','02b_benchmark_top10')]
    for slug in meta['unchanged_author_code_ids']:
        if next(c.source for c in groups[slug] if c.cell_type=='code')!=original_code[slug]:raise RuntimeError('Unexpected code edit')
    nbf.validate(nb);nbf.write(nb,TARGET)
    print(json.dumps({'notebook':str(TARGET),'cells':len(nb.cells),'figures':len(checks),
                      'removed_figure_groups':13,'author_notes_preserved':len(author_notes)},ensure_ascii=True))


def finalize_selection():
    """Apply the next author decision: split03a, use04c and rank-only05a."""
    import copy
    original=TARGET.read_bytes();before=digest(original)
    nb=nbf.reads(original.decode('utf-8'),as_version=4)
    if nb.metadata.paac_vn.get('final_selection_applied'):
        raise RuntimeError('Final selection already applied; preserve author edits.')
    history=TARGET.with_name('PAAC_VN_Stage07_Review_before_final_selection.ipynb')
    groups={}
    for cell in nb.cells:
        tags=cell.metadata.get('tags',[])
        if any(t in tags for t in ('figure-guide','figure-drawing','author-notes')):
            groups.setdefault(tags[1],[]).append(cell)
    original03=next(c for c in groups['03a_taxonomy_seed_dots'] if c.cell_type=='code')
    for exp in ('tn1','tn2'):
        slug='03a_taxonomy_seed_dots_'+exp
        copied=copy.deepcopy(groups['03a_taxonomy_seed_dots'])
        for cell in copied:
            cell.id=nbf.v4.new_code_cell().id
            cell.metadata.tags=[cell.metadata.tags[0],slug]
            cell.source=cell.source.replace('03a_taxonomy_seed_dots',slug)
        plot=next(c for c in copied if c.cell_type=='code')
        old="fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharex=True, sharey=True)"
        dims="(1, 2, figsize=(13, 6)" if exp=='tn1' else "(2, 2, figsize=(14, 9)"
        chosen="CELLS[:2]" if exp=='tn1' else "CELLS[2:]"
        assert old in plot.source,'03a author code changed; preserve it before splitting'
        plot.source=plot.source.replace(old,
            f"selected_cells = {chosen}\n    plot_reps = [r for r in REPS if ((tax.representation == r) & tax.cell.isin(selected_cells)).any()]\n    fig, axes = plt.subplots{dims}, sharex=True, sharey=True)")
        plot.source=plot.source.replace('zip(axes.flat, CELLS)','zip(axes.flat, selected_cells)')
        plot.source=plot.source.replace('enumerate(REPS)','enumerate(plot_reps)')
        plot.source=plot.source.replace('ax.set_yticks(range(8), NAMES)',
                                        'ax.set_yticks(range(len(plot_reps)), [NAME[r] for r in plot_reps])')
        plot.source=plot.source.replace('ax.set_ylim(7.6, -0.6)','ax.set_ylim(len(plot_reps)-0.4, -0.6)')
        plot.source=plot.source.replace('Taxonomy alignment · all available cells','Taxonomy alignment · '+exp.upper())
        if exp=='tn2':
            plot.source=plot.source.replace('Unavailable cells are not zero scores.',
                'LaBSE: native cap256 only; missing is not zero.\\nE5 native/T512 use the same prepared inputs/cap512; observed equal outputs are shown as separate protocol cells.')
        guide=next(c for c in copied if 'figure-guide' in c.metadata.tags)
        if exp=='tn1':
            guide.source='''<a id="03a_taxonomy_seed_dots_tn1"></a>
## 03a TN1 — Taxonomy alignment trên benchmark tiếng Việt

**RQ2:** Hai panel char950/native,8 recipes. Mỗi chấm xanh là một seed42–51, hình thoi cam là mean AMI; cùng trục AMI với bản TN2 để đối chiếu khi cần.

**Quan sát:** E5 đầu char950 theo mean AMI; BGE-M3 đầu native. Chênh nhỏ giữa các recipes không tự chứng minh superiority/equivalence. Tách TN1 khỏi TN2 chỉ đổi bố cục, không đổi seed observations hoặc metric.

**Nguồn:** `results/stage05/combined_20261025_001/taxonomy_all_by_seed.csv`. Được tác giả chọn, thay phần TN1 trong03a gộp trước đó.
'''
        else:
            guide.source='''<a id="03a_taxonomy_seed_dots_tn2"></a>
## 03a TN2 — Taxonomy alignment theo hai ngôn ngữ

**RQ1:** Bốn panel EN/VI × native/T512, chỉ hiển thị5 recipes có paired results. Chấm xanh=10 seed observations, cam=mean. LaBSE chỉ có native cap256; T512 ghi not evaluated.

**Ý nghĩa:** Đặt EN và VI cạnh nhau theo cùng input condition để xem taxonomy alignment phụ thuộc recipe/ngôn ngữ thế nào. Không suy causal translation loss.

**E5:** Native/T512 cùng prepared inputs và cap512, encode riêng và quan sát outputs trùng; không phải do TN2 bị cắt char950. Giữ cells riêng theo protocol, không coi là bằng chứng độc lập của hiệu ứng input.

**Nguồn:** `results/stage05/combined_20261025_001/taxonomy_all_by_seed.csv`. Tách khỏi TN1 theo yêu cầu tác giả; không có thí nghiệm mới.
'''
        groups[slug]=copied
    # Retain exactly panel(b) of05a, alongside the selected05b scatter.
    rank=next(c for c in groups['05a_probe_rank_heatmap'] if c.cell_type=='code')
    start=rank.source.index('    fig, axes = plt.subplots(')
    matrix=rank.source.index('    matrix = np.full(',start)
    end=rank.source.index('    show_figure(',matrix)
    rank.source=(rank.source[:start]+"    fig, ax = plt.subplots(figsize=(12, 8))\n"
        +rank.source[matrix:end].replace('axes[1]','ax').replace('(b) Ranks within available models of each view','Ranks within available models of each view')
        +"    show_figure(fig, '05a_probe_rank_heatmap', 'Clustering and probe ranks', 'Author selection', "
        +repr('Selected panel(b) from05a, companion to05b. Ranks use common recipe families within each view; ties receive average rank.\n'
              'AMI: ten-seed mean; probe: one fixed-split evaluation. TF-IDF clustering is transductive, probe fit is train-only.')
        +", '5 - Probe and AMI')\n\ndraw_this_figure()")
    rank_guide=next(c for c in groups['05a_probe_rank_heatmap'] if 'figure-guide' in c.metadata.tags)
    rank_guide.source='''<a id="05a_probe_rank_heatmap"></a>
## 05a(b) — Heatmap thứ hạng, bổ trợ scatter05b

Tác giả chọn **chỉ panel(b)** của05a để đặt cùng05b; panel(a) score-dot đã bỏ. Mỗi hàng là recipe, mỗi cặp cột là hạng AMI và Macro-F1 trong cùng view. Hạng1 cao nhất; ties dùng average rank; ô trống là không có phép đánh giá, không phải score0.

**RQ2:** Dùng cùng05b để phân biệt thứ hạng với độ lớn điểm thực. AMI dùng mean10seeds; probe có một test evaluation. TF-IDF clustering/probe khác fit scope, chỉ ghép theo recipe family. Số model trong TN1/TN2 khác nhau nên không coi một bậc hạng là cùng khoảng cách điểm giữa các view.

**Nguồn:** Stage05 taxonomy_summary/probe_all; không thay số liệu. Đây là hình bổ trợ05b, chưa khóa cách ghép panel ở layout paper.
'''
    # Latest author steering: no special color for the highest F1 in05b.
    scatter=next(c for c in groups['05b_probe_ami_scatter'] if c.cell_type=='code')
    scatter.source='\n'.join(line for line in scatter.source.splitlines()
                             if 'best_f1 =' not in line and 'point_color =' not in line)
    scatter.source=scatter.source.replace('color=point_color','color=BLUE').replace(
        '\\nOrange points mark the highest Macro-F1 within each panel; exact ties are all highlighted.','')
    nb.metadata.paac_vn.pop('05b_highlight',None)
    order=['01a_pipeline_full','02a_benchmark_all49',
           '03a_taxonomy_seed_dots_tn1','03a_taxonomy_seed_dots_tn2',
           '04b_partition_only','04c_retrieval_dots','05b_probe_ami_scatter','05a_probe_rank_heatmap']
    original_notes={c.id:c.source for slug in order if slug in groups and not slug.startswith('03a_')
                    for c in groups[slug] if 'author-notes' in c.metadata.tags}
    header=nb.cells[0]
    header.source='''# PAAC-VN · Chặng07 · Bộ hình đã chốt để viết bài

Tác giả đã chọn02a thay02b;03a tách TN1/TN2; retrieval dùng04c; scatter05b kèm heatmap(b)05a. Giữ01a cho thiết kế và04b cho partition C/W/S theo dàn ý đang dùng.

Notebook có8 cell hình (heatmap05a là bản bổ trợ05b), giữ code Matplotlib trực tiếp. Chạy setup một lần rồi chạy cell bất kỳ; outputs đã lưu sẵn. Chỉnh code/ghi chú trong notebook không tự đổi module Python, dữ liệu hoặc kết quả.

Các phương án thay thế không còn trong bản duyệt chính; bản trước lần chốt giữ ở `PAAC_VN_Stage07_Review_before_final_selection.ipynb`. Không tự phân bổ hình đã bỏ sang phụ lục.

RQ1 đánh giá tương ứng EN–VI; RQ2 đánh giá recipes/input conditions với taxonomy VI. Chốt lựa chọn hình không đồng nghĩa toàn bộ findings/outline/manuscript đã được phê duyệt. Chặng08 chưa mở.
'''
    setup=next(c for c in nb.cells if 'plot-setup' in c.metadata.get('tags',[]))
    toc=nbf.v4.new_markdown_cell('## Danh mục hình đã chốt\n\n'+'\n'.join(f'- [{slug}](#{slug})' for slug in order))
    end=nbf.v4.new_markdown_cell('''## Bước tiếp theo

Đã chốt nội dung hình. Tiếp tục chốt mạch viết/findings, caption, bảng kết quả và cách bố trí paper theo RQ. Notebook giữ riêng scatter05b và heatmap05a(b) để quan sát rõ; có thể ghép panel khi typeset sau này, không cần tính lại metrics.
''')
    nb.cells=[header,setup,toc,*[c for slug in order for c in groups[slug]],end]
    for c in nb.cells:
        if c.cell_type=='code':c.outputs=[];c.execution_count=None
    meta=nb.metadata.paac_vn
    meta.update({'final_selection_applied':True,'n_figures':8,'main_figure_ids':order,
                 'selection_status':'AUTHOR_FINAL_FIGURE_SELECTION',
                 'companion_figures':{'05b_probe_ami_scatter':['05a_probe_rank_heatmap (former panel b)']},
                 'before_final_selection_sha256':before,'before_final_selection_notebook':history.name,
                 'revision_code_sha256':digest(Path(__file__).read_bytes())})
    meta['selected_original_figures']=['01a_pipeline_full','02a_benchmark_all49',
        '03a_taxonomy_seed_dots','04b_partition_only','04c_retrieval_dots',
        '05b_probe_ami_scatter','05a_probe_rank_heatmap (panel b)']
    meta['split_figures']={'03a_taxonomy_seed_dots':order[2:4]}
    meta['unchanged_author_code_ids']=['01a_pipeline_full','02a_benchmark_all49','04b_partition_only','04c_retrieval_dots']
    meta['removed_from_active_notebook']=list(dict.fromkeys(
        meta.get('removed_from_active_notebook',[])+[s for s in groups if s not in order and s!='03a_taxonomy_seed_dots']))
    meta.pop('added_split_figure',None)
    meta['annotations']={k:v for k,v in meta.get('annotations',{}).items() if k in order}
    meta['annotations'].pop('05a_probe_rank_heatmap',None)
    nbf.validate(nb)
    client=NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(TARGET.parent)}})
    client.create_kernel_manager();client.km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    client.execute()
    checks=[]
    for c in nb.cells:
        if 'figure-drawing' in c.metadata.get('tags',[]):
            images=[o.data['image/png'] for o in c.outputs if 'image/png' in o.get('data',{})]
            if len(images)!=1:raise RuntimeError('Expected one image per selected cell')
            checks.append({'id':c.metadata.tags[1],'png_sha256':digest(base64.b64decode(images[0]))})
    if len(checks)!=8:raise RuntimeError('Expected8 figures')
    if {c.id:c.source for c in nb.cells if c.id in original_notes}!=original_notes:raise RuntimeError('Author notes changed')
    if digest(TARGET.read_bytes())!=before:raise RuntimeError('Concurrent notebook edit')
    if history.exists():raise RuntimeError('Final-selection history already exists')
    with history.open('xb') as handle:handle.write(original)
    meta['render_checks']=checks;meta['execution_status']='ALL_8_FINAL_FIGURE_CELLS_EXECUTED'
    nbf.validate(nb);nbf.write(nb,TARGET)
    print(json.dumps({'figures':len(checks),'cells':len(nb.cells),'selected':order},ensure_ascii=True))


if __name__=='__main__':
    if '--finalize-selection' in sys.argv:finalize_selection()
    else:main()
