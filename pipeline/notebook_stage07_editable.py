"""Put the existing Matplotlib drawing code into the Stage07 review notebook.

Extracts/specializes plotting statements from figures_stage07.py, suppresses
gallery CSV/file writes, and executes one figure per notebook cell. No refits.
The old display-only notebook and every author Markdown cell are preserved.
"""
from __future__ import annotations
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import nbformat as nbf
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
PLOT=ROOT/'pipeline/figures_stage07.py'
TARGET=ROOT/'results/stage07/gallery_20261025_005/PAAC_VN_Stage07_Review.ipynb'
ARCHIVE=TARGET.with_name('PAAC_VN_Stage07_Review_display_only.ipynb')


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def is_save(node):
    return (isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
            and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='save')


class InlineDrawing(ast.NodeTransformer):
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute):
            if node.value.func.attr=='to_csv':
                return None
            if is_save(node):
                node.value.func=ast.Name(id='show_figure',ctx=ast.Load())
        return self.generic_visit(node)

    def visit_If(self,node):
        node=self.generic_visit(node)
        return node if node.body else None

    def visit_For(self,node):
        node=self.generic_visit(node)
        return node if node.body else None


def code(nodes):
    module=ast.Module(body=copy.deepcopy(nodes),type_ignores=[])
    module=InlineDrawing().visit(module)
    return ast.unparse(ast.fix_missing_locations(module))


def blocks(nodes):
    result=[];current=[]
    for node in nodes:
        current.append(node)
        if is_save(node):
            result.append(current);current=[]
    if current:
        raise RuntimeError('Unsplit drawing statements')
    return result


def assignments(text):
    return ast.parse(text).body


def drawing_cells(tree):
    fs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    result={}
    for slug,compact in [('01a_pipeline_full',False),('01b_pipeline_compact',True)]:
        result[slug]=assignments(f'compact={compact}')+fs['flow'].body
    for slug,top in [('02a_benchmark_all49',False),('02b_benchmark_top10',True)]:
        result[slug]=assignments(f'top10={top}')+fs['benchmark'].body
    for slug,nodes in zip(['03a_taxonomy_seed_dots','03b_taxonomy_bars'],blocks(fs['taxonomy'].body)):
        result[slug]=nodes
    paired=fs['paired_figures'].body
    flat=blocks(paired[1:-1]);common=[paired[0]]
    result['04a_partition_retrieval_combined']=[fs['partition_axes'],fs['retrieval_axes']]+common+flat[0]
    result['04b_partition_only']=[fs['partition_axes']]+common+flat[1]
    for slug,bars in [('04c_retrieval_dots',False),('04d_retrieval_bars',True)]:
        result[slug]=[fs['retrieval_axes']]+common+assignments(f'bars={bars}')+paired[-1].body
    probe=fs['probes'].body
    first_fig=next(i for i,n in enumerate(probe) if any(
        isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute) and c.func.attr=='subplots' for c in ast.walk(n)))
    for slug,nodes in zip(['05a_probe_rank_heatmap','05b_probe_ami_scatter','05c_probe_ami_score_heatmaps'],blocks(probe[first_fig:])):
        result[slug]=assignments("probe=g.read_csv(S5,'probe_all.csv')")+probe[:first_fig]+nodes
    disp=fs['dispersion'].body
    style_index=next(i for i,n in enumerate(disp) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple)
                     and any(isinstance(x,ast.Name) and x.id=='style' for x in n.target.elts))
    for style,slug,origin in [('raincloud','06a_raincloud','Codex'),('dot-range','06b_dot_range','Claude'),('violin','06c_violin','Claude')]:
        result[slug]=assignments(f'style={style!r}\nslug={slug!r}\norigin={origin!r}\nreceipt=receipt6')+disp[:style_index]+disp[style_index].body
    result['06d_top_rank_share']=disp[style_index+1:]
    heat=fs['heatmaps'].body
    for rep,cond in [('tfidf_lsa','native'),('tfidf_lsa','token_budget'),('labse','native'),
                     ('multilingual_e5_large','native'),('multilingual_e5_large','token_budget'),
                     ('bge_m3','native'),('bge_m3','token_budget'),
                     ('qwen3_embedding_0_6b','native'),('qwen3_embedding_0_6b','token_budget')]:
        slug=f'07_heatmap_{rep}_{cond}'
        result[slug]=assignments(f'rep={rep!r}\ncond={cond!r}')+heat[:-1]+heat[-1].body
    return {slug:code(nodes) for slug,nodes in result.items()}


def setup_code(tree):
    fs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    constants=[]
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in
            {'REPS','NAMES','NAME','CELLS','CELL_NAMES','N05_NOTE'} or isinstance(t,ast.Tuple) for t in n.targets):
            constants.append(n)
    helpers=[fs[k] for k in ['sha','require','panel','cell_title','pair_order','pair_name']]
    prefix='''# Chạy cell thiết lập này một lần trước các cell vẽ hình.
from pathlib import Path
from io import BytesIO
import hashlib, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from IPython.display import Image, display

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p/'pipeline/figures_stage07.py').is_file()
             and (p/'data_v1/benchmark_v1.parquet').is_file()), None)
if ROOT is None:
    raise FileNotFoundError('Mở notebook trong repository PAAC-VN.')
S5=ROOT/'results/stage05/combined_20261025_001'
S6=ROOT/'results/stage06/n05_20261025_001'
SOURCE_GALLERY=ROOT/'results/stage07/gallery_20261025_004'
source_manifest=json.loads((SOURCE_GALLERY/'manifest.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
    'axes.titlesize':11,'axes.labelsize':9,'savefig.facecolor':'white',
    'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':True})
'''
    suffix='''
class NotebookSources:
    """Read frozen sources; never write gallery files or result tables."""
    def __init__(self):
        self.rng=np.random.default_rng(707)
    def source(self,path):
        path=Path(path)
        relative=path.relative_to(ROOT).as_posix()
        require(sha(path)==source_manifest['inputs_sha256'][relative], 'Source changed: '+relative)
        return path
    def read_csv(self,folder,name):
        return pd.read_csv(self.source(folder/name))

def show_figure(fig,slug,title,origin,caption,group):
    """Render the newly created Matplotlib figure in memory; save no file."""
    fig.suptitle(title,fontsize=15,fontweight='bold',y=.995)
    fig.text(.015,.008,caption,fontsize=8,ha='left',va='bottom',wrap=True)
    fig.tight_layout(rect=(0,.075,1,.95),pad=1.5)
    with BytesIO() as buffer:
        fig.savefig(buffer,format='png',dpi=160)
        display(Image(data=buffer.getvalue(),width=1400))
    plt.close(fig)

g=NotebookSources()
index=g.read_csv(S5,'artifact_index.csv')
tax=g.read_csv(S5,'taxonomy_all_by_seed.csv')
summary=g.read_csv(S5,'taxonomy_summary.csv')
probe=g.read_csv(S5,'probe_all.csv')
part=g.read_csv(S5,'partition_all.csv')
ret=g.read_csv(S5,'retrieval_all.csv')
contrast=g.read_csv(S6,'paired_descriptive_contrasts.csv')
receipt6=json.loads(g.source(S6/'execution_receipt.json').read_text(encoding='utf-8'))
frames={exp:pd.read_parquet(g.source(ROOT/'data_v1'/name)) for exp,name in
        [('tn1','benchmark_v1.parquet'),('tn2','paired_en.parquet')]}
print('Dữ liệu sẵn sàng. Chạy riêng cell hình muốn xem hoặc chỉnh code Matplotlib trong cell đó.')
'''
    return prefix+'\n'+code(constants+helpers)+'\n'+textwrap.dedent(suffix)


def main():
    original=TARGET.read_bytes();old_sha=sha_bytes(original)
    nb=nbf.reads(original.decode('utf-8'),as_version=4)
    mode=nb.metadata.get('paac_vn',{}).get('render_mode')
    if mode=='INLINE_MATPLOTLIB_SCOPED':
        raise RuntimeError('Notebook already converted; do not overwrite author edits.')
    repairing=mode=='INLINE_MATPLOTLIB'
    need_notes={c.id:c.source for c in nb.cells if 'author-notes' in c.metadata.get('tags',[])}
    tree=ast.parse(PLOT.read_text(encoding='utf-8'))
    drawings=drawing_cells(tree)
    if len(drawings)!=26:
        raise RuntimeError('Expected exactly 26 independent drawing blocks')
    def unscoped_cell(slug):
        return (f'# {slug}: code tạo đúng một hình từ figures_stage07.py.\n'
                '# Chạy cell thiết lập chung trước; chỉnh lệnh vẽ bên dưới rồi chạy lại.\n'
                'g = NotebookSources()  # reset drawing jitter; no file writes\n\n'+drawings[slug])
    if repairing:
        for cell in nb.cells:
            if 'figure-drawing' in cell.metadata.get('tags',[]):
                if cell.source!=unscoped_cell(cell.metadata.tags[1]):
                    raise RuntimeError('Author drawing edits detected; preserve them before conversion.')
    display_sha=nb.metadata.paac_vn['parent_notebook_sha256'] if repairing else old_sha
    nb.cells[0].source='''# PAAC-VN · Chặng07 · Code tạo từng hình và duyệt theo RQ

**26 cell tạo hình Matplotlib trực tiếp**, giữ cả ý tưởng Codex/Claude. Mỗi cell có code vẽ được lấy từ `pipeline/figures_stage07.py`, không đọc ảnh PNG để thay cho việc vẽ.

1. Mở bằng VS Code/Jupyter, chọn kernel `.venv/Scripts/python.exe` của dự án.
2. Chạy **cell thiết lập chung** ngay dưới đây một lần; sau đó chạy cell hình bất kỳ. Các cell hình không phụ thuộc thứ tự của nhau.
3. Sửa code trong cell hình: figsize, màu, markers, trục, labels… rồi chạy lại để xem. Output đã lưu sẵn để quan sát khi chưa chạy code.
4. Giữ quyết định duyệt ở Markdown **Ghi chú duyệt của tác giả** dưới từng hình rồi lưu notebook. Chỉnh notebook không tự sửa module Python hay ghi đè PNG/PDF/SVG lịch sử.

**RQ1:** Tương ứng EN–VI cùng app: vector correspondence, retrieval, partition agreement và taxonomy alignment theo ngôn ngữ.

**RQ2:** Representation/input recipes với taxonomy Google Play trên benchmark VI, xét input conditions/provenance.

Đây là notebook duyệt, chưa chọn hình cuối cùng. Chỉ đọc dữ liệu, nhãn và metrics đã có để vẽ lại; không inference, refit hoặc chạy lại N05. Bộ hình chưa có panel riêng cho cosine correspondence hay hiệu năng theo provenance. R02 đã bỏ; không suy causal translation loss.
'''
    if repairing:
        next(c for c in nb.cells if 'plot-setup' in c.metadata.get('tags',[])).source=setup_code(tree)
    else:
        nb.cells.insert(1,nbf.v4.new_code_cell(setup_code(tree),metadata={'tags':['plot-setup']}))
    converted=[]
    for c in nb.cells:
        tags=c.metadata.get('tags',[])
        if c.cell_type=='code' and ('figure-display' in tags or 'figure-drawing' in tags):
            slug=tags[1]
            c.source=(f'# {slug}: code tạo đúng một hình từ figures_stage07.py.\n'
                      '# Chạy cell thiết lập chung trước; chỉnh lệnh vẽ bên dưới rồi chạy lại.\n'
                      '# Hàm trong cell giữ biến của hình này riêng, tránh ảnh hưởng hình khác.\n'
                      'def draw_this_figure():\n'+textwrap.indent(
                          'g = NotebookSources()  # reset drawing jitter; no file writes\n\n'+drawings[slug],'    ')
                      +'\n\ndraw_this_figure()')
            c.metadata.tags=['figure-drawing',slug]
            c.metadata['source_module']='pipeline/figures_stage07.py'
            c.outputs=[];c.execution_count=None;converted.append(slug)
        if c.cell_type=='markdown' and 'figure-guide' in tags:
            c.source=c.source.replace('Cell kế tiếp hiển thị đúng ảnh đã dựng, không tính lại metric.',
                'Cell kế tiếp chứa code Matplotlib tạo hình từ dữ liệu/nhãn đã có, không refit hay tính lại các metric nghiên cứu.')
    nb.cells[-1].source='''## Sau khi duyệt

Lưu notebook với ghi chú/mã hình muốn giữ hoặc chỉnh. Cell thiết lập chỉ đọc nguồn; các cell hình tạo ảnh trong bộ nhớ và hiển thị, không ghi đè artifacts.
Không có lựa chọn cuối cùng được tự áp dụng, không khởi chạy model, KMeans, probe hoặc tính lại N05.
'''
    for c in nb.cells:
        if c.cell_type=='code':
            c.outputs=[];c.execution_count=None
    nb.metadata.paac_vn.update({'render_mode':'INLINE_MATPLOTLIB_SCOPED',
        'purpose':'Editable drawing cells plus author review; no scientific reruns',
        'parent_notebook_sha256':display_sha,'previous_display_notebook':ARCHIVE.name,
        'figures_source_sha256':sha_bytes(PLOT.read_bytes()),
        'editable_generator_sha256':sha_bytes(Path(__file__).read_bytes()),
        'conversion_source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    nbf.validate(nb)
    client=NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(TARGET.parent)}})
    client.create_kernel_manager()
    client.km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    client.execute()
    rendered=[c for c in nb.cells if 'figure-drawing' in c.metadata.get('tags',[])]
    if len(rendered)!=26:
        raise RuntimeError('Unexpected figure count')
    checks=[]
    gallery=ROOT/'results/stage07/gallery_20261025_004'
    for c in rendered:
        images=[o.data['image/png'] for o in c.outputs if o.output_type in ('display_data','execute_result') and 'image/png' in o.get('data',{})]
        if len(images)!=1 or any(o.output_type=='error' for o in c.outputs):
            raise RuntimeError('Expected exactly one generated PNG per drawing cell')
        raw=base64.b64decode(images[0]);slug=c.metadata.tags[1]
        checks.append({'id':slug,'png_sha256':sha_bytes(raw),'matches_previous_png':sha_bytes(raw)==sha_bytes((gallery/f'{slug}.png').read_bytes())})
    if {c.id:c.source for c in nb.cells if 'author-notes' in c.metadata.get('tags',[])}!=need_notes:
        raise RuntimeError('Author notes changed')
    if sha_bytes(TARGET.read_bytes())!=old_sha:
        raise RuntimeError('Notebook changed during conversion; original preserved')
    if ARCHIVE.exists():
        if sha_bytes(ARCHIVE.read_bytes())!=display_sha:
            raise RuntimeError('Historical display notebook already exists with different contents')
    else:
        with ARCHIVE.open('xb') as handle:
            handle.write(original)
    nb.metadata.paac_vn.update({'execution_status':'ALL_26_MATPLOTLIB_DRAWING_CELLS_EXECUTED',
                               'render_checks':checks,'author_notes_preserved':True})
    nbf.validate(nb)
    nbf.write(nb,TARGET)
    print(json.dumps({'path':str(TARGET),'cells':len(nb.cells),'drawing_cells':26,
                      'matches_previous_png':sum(c['matches_previous_png'] for c in checks),
                      'author_notes_preserved':True},ensure_ascii=True))


if __name__=='__main__':
    main()
