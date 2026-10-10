"""Self-contained exploratory map and machine-readable benchmark report."""
import html,json,os
from pathlib import Path
import numpy as np
import pandas as pd

TEMPLATE='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Yeast morphology · __TASK__</title>
<style>
:root{font-family:system-ui,sans-serif;color:#dce6ef;background:#101923}body{max-width:1160px;margin:auto;padding:32px}h1{font-size:34px;letter-spacing:-1px;margin:8px 0}h2{font-size:19px}p{line-height:1.55;color:#aebdcb}.tag{color:#64d2be;font-size:12px;text-transform:uppercase;letter-spacing:2px}.panel{background:#172431;border:1px solid #2b3b4b;border-radius:12px;padding:20px;margin:22px 0}select,input,button{background:#223649;color:#e6eef5;border:1px solid #476077;border-radius:5px;padding:8px;margin:4px}label{margin-right:16px;font-size:14px}canvas{width:100%;height:460px;display:block}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:10px;text-align:right;border-bottom:1px solid #304050}td:first-child,th:first-child{text-align:left}th{color:#9eafbd}.cards{display:flex;gap:24px;flex-wrap:wrap}.cards strong{font-size:26px;display:block}.small{font-size:12px}#detail{min-height:38px;color:#d2e2ed}a{color:#79d7c7}.scroll{overflow:auto}footer{font-size:12px;color:#90a4b8}</style></head>
<body><div class="tag">Genotype → morphology / exploratory benchmark</div><h1>Yeast morphology map</h1><p>__TASK__ · __FEATURES__ · observed culture profiles and held-out predictions.<br>Empty space is unobserved—not evidence of impossibility. Generated profiles do not establish viability.</p>
<div class="cards">__CARDS__</div><section class="panel"><h2>Explore the withheld organisms</h2><label>Predictor <select id="model"></select></label><label>Gene <input id="gene" placeholder="e.g. YAL002W"></label><label><input type="checkbox" id="train" checked>Training observations</label><label><input type="checkbox" id="truth" checked>Held-out observations</label><label><input type="checkbox" id="pred" checked>Predicted means</label><button id="reset">Reset view</button><p class="small">Gray: training · turquoise: held-out measurements · orange: predicted means. Dashed lines mark the +1 SD combination threshold. Click a point for its gene and prediction interval. Scroll to zoom; drag to pan.</p><canvas id="map"></canvas><div id="detail">Select an organism on the map. All axes are deviations from the matching study's reference cultures.</div></section>
<section class="panel"><h2>Held-out performance</h2><p class="small">Higher skill is better; lower CRPS and energy score are better. Coverage should approach 0.90, but is not assumed calibrated. Samples describe aggregate-profile predictions.</p><div class="scroll">__TABLE__</div></section><section class="panel"><h2>Interpretation</h2><p>__INTERPRETATION__</p><p class="small">Exact duplicate sequences are grouped. Other homologs may cross splits. Background transfer is confounded with study. PCA and all fitted transforms use training mutants only. One exploratory seed is not a definitive biological result.</p></section><footer>Source and split fingerprints, model checkpoints, numerical profiles, and paired bootstrap contrasts accompany this report. No external scripts, analytics, or network connections.</footer>
<script>const D=__DATA__;const canvas=document.getElementById('map'),ctx=canvas.getContext('2d'),sel=document.getElementById('model'),q=document.getElementById('gene');
D.models.forEach(m=>{let o=document.createElement('option');o.value=m;o.textContent=m;sel.appendChild(o)});sel.value=D.models.includes('diffusion')?'diffusion':'ridge';
let bounds,points=[],drag=null,moved=false;function reset(){const a=D.observed.map(r=>r.area).filter(Number.isFinite).sort((a,b)=>a-b),b=D.observed.map(r=>r.elongation).filter(Number.isFinite).sort((a,b)=>a-b);bounds=[a[Math.floor(a.length*.005)]-1,a[Math.floor(a.length*.995)]+1,b[Math.floor(b.length*.005)]-1,b[Math.floor(b.length*.995)]+1];draw()}function draw(){let w=canvas.clientWidth,h=460,dpr=window.devicePixelRatio||1;canvas.width=w*dpr;canvas.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);const [x0,x1,y0,y1]=bounds;let X=x=>62+(x-x0)/(x1-x0)*(w-85),Y=y=>h-48-(y-y0)/(y1-y0)*(h-75);ctx.strokeStyle='#304050';ctx.fillStyle='#9fb0bf';ctx.font='11px system-ui';for(let i=0;i<=5;i++){let x=x0+(x1-x0)*i/5,y=y0+(y1-y0)*i/5;ctx.beginPath();ctx.moveTo(X(x),25);ctx.lineTo(X(x),h-48);ctx.stroke();ctx.fillText(x.toFixed(1),X(x)-8,h-28);ctx.beginPath();ctx.moveTo(62,Y(y));ctx.lineTo(w-23,Y(y));ctx.stroke();ctx.fillText(y.toFixed(1),25,Y(y)+4)}ctx.fillText('Unbudded cell area / reference SD',w/2-90,h-6);ctx.save();ctx.translate(12,h/2+90);ctx.rotate(-Math.PI/2);ctx.fillText('Elongation / reference SD',0,0);ctx.restore();ctx.setLineDash([5,5]);ctx.strokeStyle='#7691a4';ctx.beginPath();ctx.moveTo(X(1),25);ctx.lineTo(X(1),h-48);ctx.moveTo(62,Y(1));ctx.lineTo(w-23,Y(1));ctx.stroke();ctx.setLineDash([]);points=[];let query=q.value.trim().toUpperCase();
function dot(r,x,y,color,type){if(!Number.isFinite(x)||!Number.isFinite(y)||x<x0||x>x1||y<y0||y>y1)return;let match=!query||r.gene.includes(query);ctx.globalAlpha=match?(type==='train'?.24:.8):.04;ctx.fillStyle=color;ctx.beginPath();ctx.arc(X(x),Y(y),query&&match?6:type==='train'?2:3.2,0,7);ctx.fill();if(match)points.push({x:X(x),y:Y(y),r,type});ctx.globalAlpha=1}
if(document.getElementById('train').checked)D.observed.filter(r=>r.partition==='train').forEach(r=>dot(r,r.area,r.elongation,'#95a8b9','train'));
if(document.getElementById('truth').checked)D.observed.filter(r=>r.partition==='test').forEach(r=>dot(r,r.area,r.elongation,'#64d2be','measured'));
if(document.getElementById('pred').checked)D.predicted.filter(r=>r.model===sel.value).forEach(r=>dot(r,r.predicted_area,r.predicted_elongation,'#ffb36b','predicted'));
}
canvas.addEventListener('click',e=>{if(moved)return;let b=canvas.getBoundingClientRect(),x=e.clientX-b.left,y=e.clientY-b.top;let p=points.reduce((best,p)=>Math.hypot(p.x-x,p.y-y)<Math.hypot(best.x-x,best.y-y)?p:best,{x:Infinity,y:Infinity});if(!p.r||Math.hypot(p.x-x,p.y-y)>15)return;let r=p.r,s=`${r.gene} · ${r.background} · ${p.type}`;if(p.type==='predicted')s+=` · area ${r.predicted_area.toFixed(2)} [${r.area_q05.toFixed(2)}, ${r.area_q95.toFixed(2)}] · elongation ${r.predicted_elongation.toFixed(2)} [${r.elongation_q05.toFixed(2)}, ${r.elongation_q95.toFixed(2)}] · withheld-combination probability ${(100*r.combination_probability).toFixed(1)}%`;document.getElementById('detail').textContent=s});
canvas.addEventListener('wheel',e=>{e.preventDefault();let f=e.deltaY>0?1.15:1/1.15;let [a,b,c,d]=bounds,m=(a+b)/2,n=(c+d)/2;bounds=[m+(a-m)*f,m+(b-m)*f,n+(c-n)*f,n+(d-n)*f];draw()},{passive:false});canvas.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY,...bounds];moved=false;canvas.setPointerCapture(e.pointerId)});canvas.addEventListener('pointermove',e=>{if(!drag)return;let dx=e.clientX-drag[0],dy=e.clientY-drag[1];if(Math.abs(dx)+Math.abs(dy)>3)moved=true;let[a,b,c,d]=drag.slice(2),sx=dx/(canvas.clientWidth-85)*(b-a),sy=dy/(460-75)*(d-c);bounds=[a-sx,b-sx,c+sy,d+sy];draw()});canvas.addEventListener('pointerup',()=>{drag=null});['model','train','truth','pred'].forEach(id=>document.getElementById(id).addEventListener('change',draw));q.addEventListener('input',draw);document.getElementById('reset').onclick=reset;window.addEventListener('resize',draw);reset();</script></body></html>'''

def render(out):
    out=Path(out);m=json.loads((out/'metrics.json').read_text())
    tab=pd.read_csv(out/'metrics.tsv',sep='\t',index_col=0)
    predicted=pd.read_csv(out/'per_gene.tsv',sep='\t')
    observed=pd.read_csv(out/'observed_map.tsv',sep='\t')
    best=min(m['models'],key=lambda k:m['models'][k]['crps'])
    interpretation=f"Lowest CRPS in this run: {best}. This comparison is exploratory. Examine genotype-shuffled controls and paired intervals in metrics.json before claiming genotype-specific prediction."
    report=['# Yeast genotype-to-morphology benchmark','',f"Task: **{m['config']['task']}**. Features: **{m['config']['features']}**.",'',
        'This run predicts culture-level aggregate profiles, not individual-cell images.','',
        f"Train / validation / test: {m['counts']['train']} / {m['counts']['val']} / {m['counts']['test']}.",
        f"Retained traits: {m['retained_traits']}; train PCA variance retained: {m['pca_explained_variance']:.3f}; oracle test reconstruction MSE: {m['pca_oracle_test_mse']:.3f}.",'',
        '| Model | Skill vs mean | CRPS ↓ | Energy ↓ | 90% coverage | Joint-region probability |',
        '|---|---:|---:|---:|---:|---:|']
    for name,r in m['models'].items():
        report.append(f"| {name} | {r['skill_vs_mean']:.3f} | {r['crps']:.3f} | {r['energy']:.3f} | {r['coverage90']:.3f} | {r['mean_probability_high_area_high_elongation']:.3f} |")
    report+=['',interpretation,'','## Paired comparisons','']
    for contrast,metrics in m['paired_contrasts'].items():
        r=metrics['crps'];report.append(f"- {contrast}, CRPS difference {r['difference']:.4f}, 95% group-bootstrap interval [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}].")
    report+=['','## Limits','']+['- '+s for s in m['limitations']]
    report+=['','Artifacts: `map.html`, `metrics.json`, `per_gene.tsv`, all prediction arrays, training histories and model checkpoints.','']
    (out/'report.md').write_text('\n'.join(report))
    selected=['skill_vs_mean','crps','energy','coverage90','mean_probability_high_area_high_elongation']
    table='<table><thead><tr><th>Model</th>'+''.join(f'<th>{html.escape(c)}</th>' for c in selected)+'</tr></thead><tbody>'
    for name,row in tab.iterrows():table+='<tr><td>'+html.escape(name)+'</td>'+''.join(f'<td>{row[c]:.3f}</td>' for c in selected)+'</tr>'
    table+='</tbody></table>'
    data={'models':list(m['models']),'observed':json.loads(observed.to_json(orient='records')),'predicted':json.loads(predicted.to_json(orient='records'))}
    cards=''.join(f'<div><strong>{n:,}</strong><span class="small">{html.escape(label)}</span></div>' for label,n in [('Training mutants',m['counts']['train']),('Held-out mutants',m['counts']['test']),('Morphology traits',m['retained_traits'])])
    content=TEMPLATE.replace('__TASK__',html.escape(m['config']['task'])).replace('__FEATURES__',html.escape(m['config']['features'])).replace('__CARDS__',cards).replace('__TABLE__',table).replace('__INTERPRETATION__',html.escape(interpretation)).replace('__DATA__',json.dumps(data,allow_nan=False).replace('</','<\\/'))
    (out/'map.html').write_text(content)
    # Publication-style static companion, with no invented organism images.
    os.environ.setdefault('MPLCONFIGDIR','/tmp/yeast-matplotlib')
    os.environ.setdefault('XDG_CACHE_HOME','/tmp/yeast-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,model in zip(axes,['ridge','diffusion' if 'diffusion' in m['models'] else 'gaussian']):
        train=observed[observed.partition=='train'];test=observed[observed.partition=='test'];p=predicted[predicted.model==model]
        ax.scatter(train.area,train.elongation,s=3,c='#a4b3bc',alpha=.25,label='Training observations')
        ax.scatter(test.area,test.elongation,s=7,c='#087e8b',alpha=.55,label='Held-out observations')
        ax.scatter(p.predicted_area,p.predicted_elongation,s=7,c='#e8943a',alpha=.5,label='Predicted means')
        ax.axvline(1,c='gray',ls='--',lw=.8);ax.axhline(1,c='gray',ls='--',lw=.8)
        ax.set(xlabel='Cell area (reference SD)',ylabel='Elongation (reference SD)',title=model,
            xlim=(-6,16),ylim=(-6,20));ax.legend(fontsize=7,loc='upper right')
    fig.suptitle(f"Yeast morphology · {m['config']['task']} · {m['config']['features']}")
    fig.tight_layout();fig.savefig(out/'map.png',dpi=170);plt.close(fig)
