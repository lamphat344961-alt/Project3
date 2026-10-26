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


if __name__=='__main__':main()
