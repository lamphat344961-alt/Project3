"""Render both Stage07 proposals from frozen results; no model inference/refitting.

Run: .venv/Scripts/python.exe -B -m pipeline.figures_stage07 --run-id gallery_20261025_001
Plot dependency: matplotlib==3.9.4, contourpy==1.3.2 (NumPy remains 1.26.4).
"""
from __future__ import annotations
import argparse
import hashlib
import html
import importlib.metadata
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
S5 = ROOT / 'results/stage05/combined_20261025_001'
S6 = ROOT / 'results/stage06/n05_20261025_001'
REPS = ['tfidf_lsa', 'phobert_base_v2', 'sup_simcse_phobert', 'bkai_vietnamese_biencoder',
        'labse', 'multilingual_e5_large', 'bge_m3', 'qwen3_embedding_0_6b']
NAMES = ['TF-IDF–LSA', 'PhoBERT-base-v2', 'Sup-SimCSE-PhoBERT', 'BKAI biencoder',
         'LaBSE', 'Multilingual E5-large', 'BGE-M3', 'Qwen3-Embedding-0.6B']
NAME = dict(zip(REPS, NAMES))
BLUE, ORANGE, GREEN = '#0072B2', '#D55E00', '#009E73'
CELLS = ['tn1:vi:char_budget', 'tn1:vi:native', 'tn2:en:native', 'tn2:vi:native',
         'tn2:en:token_budget', 'tn2:vi:token_budget']
CELL_NAMES = ['TN1 VI · char950', 'TN1 VI · native', 'TN2 EN · native', 'TN2 VI · native',
              'TN2 EN · T512', 'TN2 VI · T512']
N05_NOTE = ('N05: 200 subsets, 80% without replacement; fixed labels/models. '
            'Q2.5–Q97.5 describes app composition, NOT a population confidence interval.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def panel(ax, title, xlabel=None):
    ax.set_title(title, loc='left', fontweight='bold', pad=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#e5e7eb', linewidth=.6)
    if xlabel:
        ax.set_xlabel(xlabel)


def cell_title(cell):
    return CELL_NAMES[CELLS.index(cell)]


class Gallery:
    def __init__(self, out):
        self.out = out
        out.mkdir(parents=True, exist_ok=False)
        (out / 'data').mkdir()
        self.inputs, self.items, self.checks = {}, [], []
        self.book = PdfPages(out / 'all_figures.pdf')
        self.rng = np.random.default_rng(707)
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                             'axes.titlesize': 11, 'axes.labelsize': 9,
                             'savefig.facecolor': 'white', 'pdf.fonttype': 42,
                             'svg.fonttype': 'none', 'axes.unicode_minus': True})

    def source(self, path):
        path = Path(path)
        self.inputs[path.relative_to(ROOT).as_posix()] = sha(path)
        return path

    def read_csv(self, folder, name):
        path = self.source(folder / name)
        frame = pd.read_csv(path)
        frame.to_csv(self.out / 'data' / name, index=False)
        return frame

    def save(self, fig, slug, title, origin, caption, group):
        fig.suptitle(title, fontsize=15, fontweight='bold', y=.995)
        fig.text(.015, .008, caption, fontsize=8, ha='left', va='bottom', wrap=True)
        fig.tight_layout(rect=(0, .075, 1, .95), pad=1.5)
        # Detect text escaping the figure canvas before writing final artifacts.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        outer = fig.bbox
        overflow = []
        # Matplotlib retains off-view tick Text objects (e.g. the next log decade),
        # but does not actually render them. Exclude those from canvas checks.
        off_view_ticks=set()
        for ax in fig.axes:
            for axis,limits in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
                low,high=sorted(limits)
                for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                    if not low-1e-10 <= tick.get_loc() <= high+1e-10:
                        off_view_ticks.update([id(tick.label1),id(tick.label2)])
        for text in fig.findobj(matplotlib.text.Text):
            if text.get_visible() and text.get_text() and id(text) not in off_view_ticks:
                box = text.get_window_extent(renderer)
                if box.x0 < -3 or box.y0 < -3 or box.x1 > outer.width+3 or box.y1 > outer.height+3:
                    overflow.append(text.get_text()[:100])
        self.checks.append({'figure': slug, 'text_outside_canvas': overflow})
        for suffix in ('png', 'pdf', 'svg'):
            fig.savefig(self.out / f'{slug}.{suffix}', dpi=160)
        self.book.savefig(fig)
        self.items.append(dict(id=slug, title=title, origin=origin, caption=caption, group=group))
        plt.close(fig)
        print('FIGURE ' + slug, flush=True)

    def finish(self):
        self.book.close()
        cards = []
        for x in self.items:
            slug = x['id']
            cards.append(f'''<article data-origin="{html.escape(x['origin'])}" data-group="{x['group']}" id="{slug}">
<div class="meta">{slug} · {html.escape(x['origin'])}</div><h2>{html.escape(x['title'])}</h2>
<a href="{slug}.png" target="_blank"><img loading="lazy" src="{slug}.png" alt="{html.escape(x['title'])}"></a>
<p>{html.escape(x['caption'])}</p><div class="links"><button onclick="pick('{slug}',this)">☆ Chọn để so sánh</button>
<a href="{slug}.png">PNG</a><a href="{slug}.pdf">PDF</a><a href="{slug}.svg">SVG</a></div></article>''')
        page = '''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PAAC-VN · Chặng 07 · Toàn bộ phương án hình</title><style>
body{margin:0;background:#f3f5f7;color:#17212e;font:15px/1.6 system-ui,sans-serif}header,main{max-width:1500px;margin:auto;padding:24px}
header{padding-bottom:12px}h1{font-size:28px;margin:8px 0}h2{font-size:19px;margin:4px 0 16px}.intro{max-width:1000px}
.toolbar{position:sticky;top:0;background:#fff;z-index:2;padding:14px 24px;border-bottom:1px solid #ddd;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}.grid.single{grid-template-columns:1fr}
article{background:white;padding:20px;border:1px solid #ddd;border-radius:10px;scroll-margin-top:90px}article img{width:100%;height:auto;border:1px solid #eee}
.meta{font-size:12px;color:#596779}.links{display:flex;gap:16px;align-items:center}a{color:#00679e}button,select,input{padding:9px;border:1px solid #bbb;border-radius:5px;background:#fff}
button.active{background:#e1f3ff;border-color:#0072b2}p{font-size:13px}article.hidden{display:none}@media(max-width:900px){.grid{grid-template-columns:1fr}}
</style><header><div>PAAC-VN · STAGE 07 · REVIEW GALLERY</div><h1>Toàn bộ phương án hình: Codex + Claude</h1>
<div class="intro">Giữ các ý tưởng của cả hai Agent để tác giả lựa chọn. Hình dùng dữ liệu/kết quả đã có; không refit, không inference lại.
Nhãn Codex/Claude chỉ nguồn ý tưởng, không có nghĩa Agent kia đã phê duyệt ảnh thực tế. Các biến thể chưa được chọn cho bài chính/phụ lục.</div>
<p>Nhấn ảnh để xem lớn. Chọn nhiều hình rồi bật “Chỉ hình đã chọn” để đối chiếu; lựa chọn lưu trong trình duyệt nếu được hỗ trợ.
Tải <a href="all_figures.pdf">PDF toàn bộ hình</a> · <a href="manifest.json">Nguồn dữ liệu và kiểm chứng</a> · <a href="data/">Bảng số liệu</a>.</p>
<p><strong>Lưu ý:</strong> Probe là một kết quả trên fixed split, không có 10 seed observations. N05 là dispersion mô tả, không population CI.
Heatmap cụm không chứng minh tác động nhân quả của dịch; không có đối chứng back-translation.</p></header>
<div class="toolbar"><select id="origin" onchange="filter()"><option value="">Mọi nguồn ý tưởng</option><option>Codex</option><option>Claude</option><option>Chung</option></select>
<select id="group" onchange="filter()"><option value="">Mọi nhóm hình</option>GROUPS</select>
<input id="search" placeholder="Tìm mã hoặc tên hình" oninput="filter()"><label><input id="only" type="checkbox" onchange="filter()">Chỉ hình đã chọn</label>
<button onclick="document.querySelector('.grid').classList.toggle('single')">Đổi 1 / 2 cột</button><span id="count"></span></div>
<main class="grid">CARDS</main><script>
let selected=new Set();try{selected=new Set(JSON.parse(localStorage.getItem('paac-stage07-selection')||'[]'))}catch(e){}
function sync(){document.querySelectorAll('article').forEach(a=>{let b=a.querySelector('button');b.classList.toggle('active',selected.has(a.id));b.textContent=selected.has(a.id)?'★ Đã chọn':'☆ Chọn để so sánh'})}
function pick(id){selected.has(id)?selected.delete(id):selected.add(id);try{localStorage.setItem('paac-stage07-selection',JSON.stringify([...selected]))}catch(e){}sync();filter()}
function filter(){let n=0;document.querySelectorAll('article').forEach(a=>{let show=(!origin.value||a.dataset.origin.includes(origin.value))&&(!group.value||a.dataset.group===group.value)&&(!only.checked||selected.has(a.id))&&a.textContent.toLowerCase().includes(search.value.toLowerCase());a.classList.toggle('hidden',!show);n+=show});document.querySelector('#count').textContent=n+' hình · '+selected.size+' đã chọn'}
const origin=document.querySelector('#origin'),group=document.querySelector('#group'),only=document.querySelector('#only'),search=document.querySelector('#search');sync();filter();
</script></html>'''
        page = page.replace('CARDS', '\n'.join(cards)).replace('GROUPS', ''.join(
            f'<option>{g}</option>' for g in dict.fromkeys(x['group'] for x in self.items)))
        (self.out / 'index.html').write_text(page, encoding='utf-8')
        outputs = {p.relative_to(self.out).as_posix(): sha(p) for p in self.out.rglob('*') if p.is_file()}
        receipt = dict(status='FIGURE_GALLERY_RENDERED_FOR_AUTHOR_SELECTION', n_figures=len(self.items),
                       source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                       code_sha256=sha(Path(__file__)), inputs_sha256=self.inputs, outputs_sha256=outputs,
                       packages={n:importlib.metadata.version(n) for n in ('numpy','pandas','matplotlib','scipy','pyarrow','contourpy')},
                       figures=self.items, layout_checks=self.checks,
                       scope='Read-only frozen inputs. No inference/refit; author selection pending. No population CI or causal claims.')
        (self.out / 'manifest.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
        require(not any(c['text_outside_canvas'] for c in self.checks), 'Some text escapes figure canvas; inspect manifest')


def flow(g, compact=False):
    fig, ax = plt.subplots(figsize=(13, 6 if compact else 8))
    ax.set(xlim=(0, 12), ylim=(0, 7)); ax.axis('off')
    def box(x,y,w,h,text,color='#e9f2fa'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.1',facecolor=color,edgecolor='#738296'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
    def arrow(x,y,xx,yy):
        ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=13,color='#596779'))
    box(.2,5.45,2.4,.9,'Google Play listings\nvi / vn · list/search/similar')
    box(3,5.45,2.4,.9,'Custom Node.js scraper\nMongoDB · appId dedup')
    box(6,5.45,2.4,.9,'Frozen preprocessing\nEligibility + near-duplicates')
    arrow(2.7,5.9,2.9,5.9);arrow(5.5,5.9,5.9,5.9)
    box(1,3.2,4,.95,'TN1 · 23,883 apps · VI\n49 genres · char950 / native')
    box(7,3.2,4,.95,'TN2 · 13,657 paired apps · EN + VI\n49 genres · native / T512 where available')
    arrow(7.2,5.3,3,4.25);arrow(7.2,5.3,9,4.25)
    box(2.1,1.6,7.8,.9,'Frozen representations / input recipes\n34 clustering cells · 26 probe cells · 9 paired conditions','#e5f1eb')
    arrow(3,3.1,4.5,2.6);arrow(9,3.1,7.5,2.6)
    for x,text in [(0.2,'Taxonomy alignment\nKMeans K=49 · seeds 42–51'),(4.25,'Cross-language correspondence\nPartition C/W/S + pair retrieval'),(8.3,'Frozen-embedding probe\nLogistic regression · Macro-F1')]:
        box(x,.1,3.4,.9,text);arrow(6,1.5,x+1.7,1.1)
    note = 'Observed catalogue, not a random sample of Google Play. TN2 is a paired subset of TN1; counts are frozen cohort sizes.'
    if not compact:
        ax.text(.2,4.7,'Translation flag: presence of translated-description slot in Google Play response.\nCollector does not call a separate translation API; not-flagged does not establish human authorship.',fontsize=9,color='#526171')
    g.save(fig,'01b_pipeline_compact' if compact else '01a_pipeline_full','Research workflow · '+('compact' if compact else 'expanded'), 'Chung', note,'1 · Quy trình')


def benchmark(g, frames, top10):
    tn1,tn2=frames['tn1'],frames['tn2']
    counts=tn1.genreId.value_counts().sort_values(ascending=False)
    fig,axes=plt.subplots(1,3,figsize=(17,8 if top10 else 15),gridspec_kw={'width_ratios':[1.25,1,1]})
    ax=axes[0]
    if top10:
        order=counts.index[:10].tolist()+['OTHER (39 genres)']
    else:
        order=counts.index.tolist()
    for offset,(d,label,color) in enumerate([(tn1,'TN1',BLUE),(tn2,'TN2',ORANGE)]):
        c=d.genreId.value_counts()
        values=[int(c.get(k,0)) if not k.startswith('OTHER') else int(c[~c.index.isin(order[:10])].sum()) for k in order]
        ax.barh(np.arange(len(order))+(offset-.5)*.36,values,height=.35,label=label,color=color)
        pd.DataFrame({'genre':order,'n_apps':values}).to_csv(g.out/'data'/f'genre_{label}_{"top10" if top10 else "all"}.csv',index=False)
    ax.set_yticks(range(len(order)),[x.replace('_',' ').title() for x in order],fontsize=8);ax.invert_yaxis();ax.legend()
    panel(ax,'(a) Genre support','Number of apps')
    ax=axes[1]
    colors=[BLUE,ORANGE,'#888888'];labels=['translation-flagged','not-flagged','unknown']
    for j,d in enumerate((tn1,tn2)):
        counts2=[int((d.is_translated==True).sum()),int((d.is_translated==False).sum()),int(d.is_translated.isna().sum())]
        left=0
        for n,color,label in zip(counts2,colors,labels):
            ax.barh(j,n/len(d),left=left,color=color,label=label if j==0 else None,height=.4)
            if n/len(d)>.08:ax.text(left+n/len(d)/2,j,f'{n:,}',ha='center',va='center',color='white',fontsize=9)
            left+=n/len(d)
    ax.set_yticks([0,1],['TN1','TN2']);ax.set_xlim(0,1);ax.set_ylim(1.7,-.7);ax.legend(loc='upper left',fontsize=7)
    ax.text(.01,.05,'TN1 unknown: 23 apps\nTN2: all translation-flagged',transform=ax.transAxes,fontsize=9)
    panel(ax,'(b) Observed provenance','Fraction of cohort')
    ax=axes[2]
    lengths=[]
    for exp,field,label,color,style in [('tn1','description_clean','TN1 VI',BLUE,'-'),('tn2','description_clean','TN2 VI',ORANGE,'-'),('tn2','description_original_clean','TN2 EN',GREEN,'--')]:
        a=np.sort(frames[exp][field].str.len().to_numpy());ax.plot(a,np.arange(1,len(a)+1)/len(a),label=label,color=color,linestyle=style)
        lengths.append(dict(view=label,n=len(a),q50=float(np.median(a)),q95=float(np.quantile(a,.95)),fraction_gt_950=float((a>950).mean())))
    pd.DataFrame(lengths).to_csv(g.out/'data'/'character_length_summary.csv',index=False)
    ax.axvline(950,color='#555',ls=':',label='950 chars');ax.set_xscale('log');ax.set_ylim(0,1);ax.legend(fontsize=8)
    panel(ax,'(c) Clean description length','Characters (log scale)')
    g.save(fig,'02b_benchmark_top10' if top10 else '02a_benchmark_all49','Benchmark composition · '+('top 10 + other' if top10 else 'all 49 genres'),'Claude' if top10 else 'Codex',
           'Lengths are before input clipping; characters are not model tokens. Counts include TN2 within TN1. Provenance indicates observed flags, not authorship.','2 · Dữ liệu')


def taxonomy(g,tax,summary):
    fig,axes=plt.subplots(2,3,figsize=(17,10),sharex=True,sharey=True)
    for ax,cell in zip(axes.flat,CELLS):
        rows=tax[tax.cell==cell]
        for y,rep in enumerate(REPS):
            a=rows[rows.representation==rep].sort_values('seed').ami.to_numpy()
            if len(a):
                require(len(a)==10,'Each clustering cell must have ten seeds')
                ax.scatter(a,y+np.linspace(-.13,.13,10),s=17,color=BLUE,alpha=.45)
                ax.scatter(a.mean(),y,s=48,color=ORANGE,marker='D',zorder=3)
            else:ax.text(.24,y,'not evaluated',fontsize=7,color='#999',va='center')
        ax.set_yticks(range(8),NAMES);ax.set_ylim(7.6,-.6);ax.set_xlim(.20,.57)
        panel(ax,cell_title(cell),'AMI')
    g.save(fig,'03a_taxonomy_seed_dots','Taxonomy alignment · all available cells','Chung',
           'Blue dots: ten KMeans seeds (42–51). Orange diamond: mean. K=49, n_init=10. Unavailable cells are not zero scores.','3 · Clustering')
    fig,axes=plt.subplots(1,2,figsize=(14,6),sharey=True)
    for ax,cell in zip(axes,CELLS[:2]):
        d=summary[summary.cell==cell].set_index('representation').loc[REPS]
        ax.barh(range(8),d.ami_mean,color=BLUE,alpha=.8)
        for y,v in enumerate(d.ami_mean):ax.text(v+.003,y,f'{v:.3f}',va='center',fontsize=8)
        ax.set_yticks(range(8),NAMES);ax.set_ylim(7.6,-.6);ax.set_xlim(0,.59)
        panel(ax,cell_title(cell),'Mean AMI')
    g.save(fig,'03b_taxonomy_bars','Taxonomy alignment · eight recipes as bars','Claude',
           'Ten-seed means; bars begin at zero. Compare with 03a to inspect the actual seed observations.','3 · Clustering')


def pair_order(part):
    return [(r,c) for r in REPS for c in ('native','token_budget') if ((part.representation==r)&(part.condition==c)).any()]


def pair_name(rep,cond):
    return NAME[rep]+' · '+('T512' if cond=='token_budget' else 'native')


def partition_axes(axes,part,order):
    p=part.set_index(['representation','condition']).loc[order]
    for key,color,marker in [('C',BLUE,'o'),('W_EN',ORANGE,'s'),('W_VI',GREEN,'^')]:
        axes[0].scatter(p[key],range(len(p)),label=key,color=color,marker=marker,s=35)
    axes[0].legend(fontsize=8);axes[0].set_xlim(.6,.86)
    axes[1].scatter(p.S,range(len(p)),color=BLUE,s=40);axes[1].axvline(0,color='#777',ls=':');axes[1].set_xlim(-.18,.015)
    for ax,title,xlabel in zip(axes,['Partition agreement','Seed-relative gap'],['AMI','S = C − (W_EN + W_VI) / 2']):
        ax.set_yticks(range(len(p)),[pair_name(*x) for x in order]);ax.set_ylim(len(p)-.5,-.7);panel(ax,title,xlabel)


def retrieval_axes(axes,ret,order,bars=False):
    metrics=['recall_at_1_primary','recall_at_10_supplementary','mrr_supplementary']
    for ax,metric,title in zip(axes,metrics,['Recall@1','Recall@10','MRR']):
        for offset,(direction,color,marker,label) in enumerate([('EN_TO_VI',BLUE,'o','EN→VI'),('VI_TO_EN',ORANGE,'s','VI→EN')]):
            d=ret[ret.direction==direction].set_index(['representation','condition']).loc[order]
            if bars:ax.barh(np.arange(len(order))+(offset-.5)*.35,d[metric],height=.34,color=color,label=label)
            else:ax.scatter(d[metric],np.arange(len(order))+(offset-.5)*.18,color=color,marker=marker,label=label,s=30)
        ax.set_yticks(range(len(order)),[pair_name(*x) for x in order]);ax.set_ylim(len(order)-.5,-.7);ax.set_xlim(0,1.035)
        ax.legend(fontsize=8,loc='lower left');panel(ax,title,'Fraction / score')


def paired_figures(g,part,ret):
    order=pair_order(part)
    fig,axes=plt.subplots(1,5,figsize=(24,7),sharey=True,gridspec_kw={'width_ratios':[1.1,1.2,1,1,1]})
    partition_axes(axes[:2],part,order);retrieval_axes(axes[2:],ret,order)
    g.save(fig,'04a_partition_retrieval_combined','Paired evaluation · partition and retrieval together','Codex',
           'Separate axes/metrics; no combined score. C/W/S are partition-level summaries, retrieval is query-level against the full 13,657-app catalogue.','4 · EN–VI')
    fig,axes=plt.subplots(1,2,figsize=(13,7),sharey=True)
    partition_axes(axes,part,order)
    g.save(fig,'04b_partition_only','Paired partition agreement · C, W_EN, W_VI, S','Claude',
           'Scalar summaries: C averages 100 cross-language seed pairs; each W averages 45 within-language seed pairs. No independent-pair error bars.','4 · EN–VI')
    for bars in (False,True):
        fig,axes=plt.subplots(1,3,figsize=(15,7),sharey=True)
        retrieval_axes(axes,ret,order,bars)
        g.save(fig,'04d_retrieval_bars' if bars else '04c_retrieval_dots','Bidirectional pair retrieval · '+('grouped bars' if bars else 'dots'),
               'Claude' if bars else 'Codex + Claude','All 13,657 queries per direction against the fixed full catalogue. Recall@1 primary; Recall@10 and MRR supplementary.','4 · EN–VI')


def probes(g,probe,summary):
    # This is a display-family join, not a claim that lexical vectors are shared.
    probe=probe.rename(columns={'representation':'probe_representation'}).copy()
    probe['representation']=probe.probe_representation.replace({'tfidf_lsa_train_only':'tfidf_lsa'})
    merged=summary.merge(probe,on=['representation','cell'],how='inner',validate='one_to_one')
    require(len(merged)==26,'Expected 26 comparable cells')
    merged.to_csv(g.out/'data'/'probe_taxonomy_comparison.csv',index=False)
    view=CELLS[:4]
    fig,axes=plt.subplots(1,2,figsize=(17,8),gridspec_kw={'width_ratios':[1,1.3]})
    for j,cell in enumerate(view):
        d=merged[merged.cell==cell].set_index('representation')
        for y,rep in enumerate(REPS):
            if rep in d.index:axes[0].scatter(d.loc[rep,'test_macro_f1'],y+(j-1.5)*.14,marker=['o','s','^','D'][j],color=[BLUE,ORANGE,GREEN,'#CC79A7'][j],s=35,label=cell_title(cell) if y==0 or (y==4 and j>=2) else None)
    axes[0].set_yticks(range(8),NAMES);axes[0].set_ylim(7.6,-.6);axes[0].set_xlim(.4,.7)
    handles,labels=axes[0].get_legend_handles_labels();unique=dict(zip(labels,handles))
    axes[0].legend(unique.values(),unique.keys(),fontsize=7,loc='lower left')
    panel(axes[0],'(a) Held-out linear probe','Macro-F1')
    matrix=np.full((8,8),np.nan)
    for j,cell in enumerate(view):
        d=merged[merged.cell==cell]
        for k,col in enumerate(['ami_mean','test_macro_f1']):
            ranks=rankdata(-d[col].to_numpy(),method='average')
            for rep,rank in zip(d.representation,ranks):matrix[REPS.index(rep),j*2+k]=rank
    image=axes[1].imshow(np.ma.masked_invalid(matrix),cmap='viridis_r',vmin=1,vmax=8,aspect='auto')
    axes[1].set_yticks(range(8),NAMES);axes[1].set_xticks(range(8),[cell_title(c)+'\n'+m for c in view for m in ['AMI rank','F1 rank']],rotation=45,ha='right',fontsize=8)
    for i,j in zip(*np.where(np.isfinite(matrix))):axes[1].text(j,i,f'{matrix[i,j]:g}',ha='center',va='center',color='white' if matrix[i,j]>4 else '#111',fontsize=9)
    axes[1].set_title('(b) Ranks within available models of each view',loc='left',fontweight='bold');fig.colorbar(image,ax=axes[1],shrink=.6,label='Rank (lower is better)')
    g.save(fig,'05a_probe_rank_heatmap','Linear probe · scores and clustering ranks','Codex',
           'Probe: one fixed split/model per cell; AMI: ten-seed mean. TF-IDF–LSA uses different fits: clustering transductive, probe train-only.\nRanks use common recipe families within each view; ties receive average rank.','5 · Probe và AMI')
    fig,axes=plt.subplots(2,2,figsize=(16,11),sharex=True,sharey=True)
    for ax,cell in zip(axes.flat,view):
        d=merged[merged.cell==cell].sort_values('test_macro_f1')
        for label_y,row in zip(np.linspace(.43,.68,len(d)),d.itertuples()):
            ax.scatter(row.ami_mean,row.test_macro_f1,color=BLUE,s=42)
            ax.annotate(NAME[row.representation],(row.ami_mean,row.test_macro_f1),xytext=(.60,label_y),
                        textcoords='data',fontsize=8,ha='left',va='center',
                        arrowprops=dict(arrowstyle='-',color='#999',linewidth=.6))
        ax.set(xlim=(.22,.83),ylim=(.40,.70),ylabel='Held-out Macro-F1');panel(ax,cell_title(cell),'Taxonomy AMI (10-seed mean)')
    g.save(fig,'05b_probe_ami_scatter','Clustering AMI versus linear-probe Macro-F1','Claude',
           'One point per recipe/view; comparisons are descriptive, not an independent-model significance test. Lexical clustering/probe fit scopes differ (transductive vs train-only).','5 · Probe và AMI')
    fig,axes=plt.subplots(1,2,figsize=(17,8))
    for ax,col,title in zip(axes,['ami_mean','test_macro_f1'],['Taxonomy AMI (mean)','Probe Macro-F1']):
        d=merged.pivot(index='representation',columns='cell',values=col).reindex(index=REPS,columns=view)
        im=ax.imshow(np.ma.masked_invalid(d.to_numpy()),aspect='auto',cmap='viridis')
        ax.set_yticks(range(8),NAMES);ax.set_xticks(range(4),[cell_title(c) for c in view],rotation=35,ha='right',fontsize=8)
        for i,j in zip(*np.where(np.isfinite(d))):ax.text(j,i,f'{d.iloc[i,j]:.3f}',ha='center',va='center',color='white',fontsize=9)
        ax.set_title(title,loc='left',fontweight='bold');fig.colorbar(im,ax=ax,shrink=.65)
    g.save(fig,'05c_probe_ami_score_heatmaps','Clustering and probe · heatmaps of actual scores','Claude',
           'AMI and Macro-F1 have independent color scales; missing is not zero. TF-IDF–LSA: clustering transductive, probe train-only.\nValues rounded to three decimals only for display.','5 · Probe và AMI')


def dispersion(g,index,receipt,contrast):
    z1=np.load(g.source(S6/'tn1_label_dispersion.npz'),allow_pickle=False)
    z2=np.load(g.source(S6/'tn2_label_dispersion.npz'),allow_pickle=False)
    groups=receipt['pair_groups']
    pair=z2['pair_metrics'][...,0]
    ss=pair[:,:,:100].mean(axis=2)-(pair[:,:,100:145].mean(axis=2)+pair[:,:,145:].mean(axis=2))/2
    native=[(j,x) for j,x in enumerate(groups) if x['condition']=='native']
    first=[]
    for j,x in native:
        row=pd.read_csv(S6/'partition_app_dispersion.csv')
        row=row[(row.representation==x['representation'])&(row.condition=='native')&(row.component=='S')].iloc[0]
        vals=ss[:,j];require(np.isclose(np.quantile(vals,.025),row.q025,atol=1e-10),'S draw mapping')
        first.append((NAME[x['representation']],vals,float(row.point)))
    tax=z1['taxonomy'][...,0].mean(axis=2)
    slots={int(i):j for j,i in enumerate(z1['cell_indices'])}
    second=[]
    for rep in REPS:
        a=index[(index.experiment=='tn1')&(index.representation==rep)]
        c=int(a[a.condition=='char_budget'].index[0]);n=int(a[a.condition=='native'].index[0])
        vals=tax[:,slots[c]]-tax[:,slots[n]]
        row=contrast[(contrast.representation==rep)&(contrast.contrast=='char950_minus_native')&(contrast.metric=='ami')].iloc[0]
        require(np.isclose(np.quantile(vals,.975),row.q975,atol=1e-10),'TN1 contrast mapping')
        second.append((NAME[rep],vals,float(row.point)))
    for style,slug,origin in [('raincloud','06a_raincloud','Codex'),('dot-range','06b_dot_range','Claude'),('violin','06c_violin','Claude')]:
        fig,axes=plt.subplots(1,2,figsize=(17,8))
        for ax,items,title in zip(axes,[first,second],['(a) TN2 native · S','(b) TN1 · AMI char950 − native']):
            data=[]
            for y,(label,values,point) in enumerate(items):
                low,mid,high=np.quantile(values,[.025,.5,.975])
                if style!='dot-range':
                    violin=ax.violinplot([values],positions=[y],vert=False,widths=.65,showextrema=False)
                    for body in violin['bodies']:
                        body.set_facecolor(BLUE);body.set_alpha(.25)
                        if style=='raincloud':body.get_paths()[0].vertices[:,1]=np.minimum(body.get_paths()[0].vertices[:,1],y)
                    if style=='raincloud':ax.scatter(values,y+.12+g.rng.uniform(0,.20,len(values)),s=4,color=BLUE,alpha=.20,rasterized=True)
                ax.plot([low,high],[y,y],color=BLUE,lw=2);ax.scatter(mid,y,color=BLUE,s=22,zorder=3)
                ax.scatter(point,y,marker='D',facecolor='white',edgecolor=ORANGE,s=45,zorder=4)
                if style=='dot-range':ax.annotate(f'[{low:.4f}, {high:.4f}]',(high,y),xytext=(5,-3),textcoords='offset points',fontsize=7)
                data.extend(dict(label=label,draw=b,value=float(v),full_catalogue_point=point) for b,v in enumerate(values))
            ax.axvline(0,color='#666',ls=':');ax.set_yticks(range(len(items)),[i[0] for i in items]);ax.set_ylim(len(items)-.4,-.6)
            ax.margins(x=.35 if style=='dot-range' else .08);panel(ax,title,'AMI difference (descriptive)')
            if style=='raincloud':pd.DataFrame(data).to_csv(g.out/'data'/('n05_S_draws.csv' if ax is axes[0] else 'n05_input_contrast_draws.csv'),index=False)
        g.save(fig,slug,'App-composition dispersion · '+style,origin,N05_NOTE+' Blue: median/range; hollow orange diamond: full-catalogue point.','6 · Độ ổn định')
    rows=[]
    for filename,metric in [('taxonomy_rank_stability.csv','AMI'),('probe_rank_stability.csv','Macro-F1')]:
        d=g.read_csv(S6,filename)
        d['representation']=d.representation.replace({'tfidf_lsa_train_only':'tfidf_lsa'})
        d=d[(d.experiment=='tn1')&(d.condition.isin(['char_budget','native']))]
        for cond in ('char_budget','native'):
            a=d[d.condition==cond].set_index('representation').reindex(REPS)
            rows.append((metric+' · '+('char950' if cond=='char_budget' else 'native'),a.share_draws_top_tie_adjusted.to_numpy()))
    fig,ax=plt.subplots(figsize=(12,7))
    matrix=np.stack([a for _,a in rows],axis=1);im=ax.imshow(matrix,aspect='auto',cmap='Blues',vmin=0,vmax=1)
    ax.set_yticks(range(8),NAMES);ax.set_xticks(range(4),[s for s,_ in rows])
    for i in range(8):
        for j in range(4):ax.text(j,i,f'{100*matrix[i,j]:.1f}%',ha='center',va='center',color='white' if matrix[i,j]>.6 else '#111')
    fig.colorbar(im,ax=ax,label='Tie-adjusted fraction of draws ranked first')
    g.save(fig,'06d_top_rank_share','TN1 ranking stability · clustering and probe','Chung',N05_NOTE+'\nShares are not probabilities of superiority. TF-IDF–LSA: clustering transductive fit; probe train-only fit.','6 · Độ ổn định')


def heatmaps(g,index,frames):
    df=frames['tn2'];genres=sorted(df.genreId.unique());yy=pd.Categorical(df.genreId,categories=genres).codes.astype(np.int64)
    for rep,cond in pair_order(pd.read_csv(S5/'partition_all.csv')):
        labels=[]
        for lang in ('en','vi'):
            row=index[(index.experiment=='tn2')&(index.representation==rep)&(index.condition==cond)&(index.language==lang)].iloc[0]
            path=g.source(ROOT/row.labels_path);require(sha(path)==row.labels_sha256,'Frozen label hash')
            z=np.load(path,allow_pickle=False);require(z['seeds'].tolist()==list(range(42,52)),'Frozen seeds')
            side=g.source((ROOT/row.embedding_path).with_suffix('.rows.parquet'))
            require(sha(side)==row.rows_sha256,'Rows hash')
            ids=pd.read_parquet(side);require(ids.appId.tolist()==df.appId.tolist(),'TN2 app alignment')
            labels.append(z['labels'][0])
        en,vi=labels
        contingency=np.bincount(en*49+vi,minlength=49*49).reshape(49,49)
        genre_en=np.bincount(yy*49+en,minlength=49*49).reshape(49,49)
        genre_vi=np.bincount(yy*49+vi,minlength=49*49).reshape(49,49)
        require(contingency.sum()==len(df) and genre_en.sum()==len(df) and genre_vi.sum()==len(df),'Heatmap count conservation')
        # Pure display permutations: same source partitions, no relabeling of stored data.
        en_order=np.lexsort((np.arange(49),-genre_en.max(axis=0),genre_en.argmax(axis=0)))
        rr,cc=linear_sum_assignment(-contingency)
        mapping=dict(zip(rr,cc));vi_order=np.array([mapping[k] for k in en_order])
        title=pair_name(rep,cond)
        fig,axes=plt.subplots(1,3,figsize=(23,11),gridspec_kw={'width_ratios':[1,1.2,1.2]})
        m=contingency[np.ix_(en_order,vi_order)]
        im=axes[0].imshow(m,cmap='viridis',norm=LogNorm(vmin=1,vmax=max(2,m.max())),aspect='auto')
        axes[0].set_xticks(range(49),vi_order,rotation=90,fontsize=5);axes[0].set_yticks(range(49),en_order,fontsize=5)
        axes[0].set(xlabel='VI cluster ID (display-matched)',ylabel='EN cluster ID',title='(a) Shared apps between EN/VI clusters')
        fig.colorbar(im,ax=axes[0],shrink=.5,label='App count (log; zero masked)')
        for ax,counts,order,lang,letter in zip(axes[1:],[genre_en,genre_vi],[en_order,vi_order],['EN','VI'],['b','c']):
            fraction=counts[:,order]/counts.sum(axis=1,keepdims=True)
            im=ax.imshow(fraction,cmap='viridis',vmin=0,vmax=1,aspect='auto')
            ax.set_xticks(range(49),order,rotation=90,fontsize=5)
            ax.set_yticks(range(49),[x.replace('_',' ').title() for x in genres],fontsize=6)
            ax.set(xlabel=f'{lang} cluster ID',title=f'({letter}) Genre → {lang} cluster distribution')
            fig.colorbar(im,ax=ax,shrink=.5,label='Within-genre app fraction')
        slug=f'07_heatmap_{rep}_{cond}'
        for suffix,table,cols in [('en_vi',contingency,[f'vi_{i}' for i in range(49)]),('genre_en',genre_en,[f'en_{i}' for i in range(49)]),('genre_vi',genre_vi,[f'vi_{i}' for i in range(49)])]:
            pd.DataFrame(table,index=range(49) if suffix=='en_vi' else genres,columns=cols).to_csv(g.out/'data'/f'{slug}_{suffix}.csv')
        g.save(fig,slug,'EN–VI cluster and genre structure · '+title,'Claude',
               'Illustrative seed 42; all nine paired conditions shown, not selected by performance. Hungarian permutation only for display. Genre panels row-normalized.\n'
               'Cluster IDs have no shared semantics across views; these heatmaps do not identify a causal translation effect. Full counts exported as CSV.','7 · Genre / cluster')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    require(args.run_id and all(c.isalnum() or c in '_-' for c in args.run_id),'Invalid run ID')
    g=Gallery(ROOT/'results/stage07'/args.run_id)
    for folder,status in [(S5,'STAGE05_COMPLETE_VALIDATED'),(S6,'STAGE06_DESCRIPTIVE_ANALYSIS_COMPLETE')]:
        receipt=json.loads(g.source(folder/'execution_receipt.json').read_text(encoding='utf-8'))
        require(receipt['status']==status,'Source computation incomplete')
        for name,digest in receipt['artifacts_sha256'].items():
            require(sha(folder/name)==digest,'Source artifact changed: '+name)
    receipt6=json.loads((S6/'execution_receipt.json').read_text(encoding='utf-8'))
    index=g.read_csv(S5,'artifact_index.csv')
    tax=g.read_csv(S5,'taxonomy_all_by_seed.csv');summary=g.read_csv(S5,'taxonomy_summary.csv')
    probe=g.read_csv(S5,'probe_all.csv');part=g.read_csv(S5,'partition_all.csv');ret=g.read_csv(S5,'retrieval_all.csv')
    contrast=g.read_csv(S6,'paired_descriptive_contrasts.csv')
    g.source(S6/'partition_app_dispersion.csv')
    frozen={'tn1':('benchmark_v1.parquet','0cf5e72c00f1392e9f7e83ef2fc282e8deb9e81bc6e5d332ce5d761a41f7c52b'),
            'tn2':('paired_en.parquet','9208bfc42fc66b45c19496af34088cb742d1ea2e7cca25998e139b16c7e77493')}
    frames={}
    for exp,(name,digest) in frozen.items():
        path=g.source(ROOT/'data_v1'/name);require(sha(path)==digest,'Frozen dataset changed')
        frames[exp]=pd.read_parquet(path)
    require(len(frames['tn1'])==23883 and len(frames['tn2'])==13657,'Frozen cohort sizes')
    require(set(frames['tn2'].appId)<=set(frames['tn1'].appId),'Paired subset lineage')
    flow(g);flow(g,True)
    benchmark(g,frames,False);benchmark(g,frames,True)
    taxonomy(g,tax,summary);paired_figures(g,part,ret);probes(g,probe,summary)
    dispersion(g,index,receipt6,contrast);heatmaps(g,index,frames)
    g.finish()
    print('GALLERY '+str(g.out/'index.html'),flush=True)


if __name__=='__main__':
    main()
