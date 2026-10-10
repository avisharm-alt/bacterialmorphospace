"""Rank unmeasured deletion × background candidates; never label them viable."""
import html,json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import load,AXES,BACKGROUND_GENES,save_json,sha256
from .experiment import predict


def build(root,run_dir,out,embeddings=None,model='diffusion',n_samples=32,device='auto',max_genes=0):
    x,y,traits,meta=load(root,embeddings)
    # These genes have a measured single deletion, but no measured altered-background profile.
    genes=sorted(set(meta.loc[meta.background=='single','gene'])-set(meta.loc[meta.background=='triple','gene'])-BACKGROUND_GENES)
    if max_genes:genes=genes[:max_genes]
    if not genes:raise ValueError('No eligible unmeasured combinations')
    samples,bundle=predict(root,run_dir,genes,'triple',model,out,n_samples,device=device,embeddings=embeddings,compact=True)
    axes=[bundle['traits'].index(a) for a in AXES];points=samples[:,:,axes]
    measured=y[meta.background=='triple'][:,[traits.index(a) for a in AXES]]
    # Physical feature bins, fixed at one reference SD. No density tuning on candidate results.
    finite=measured[np.isfinite(measured).all(1)]
    occupied=set(map(tuple,np.floor(finite).astype(int)))
    bins=np.floor(points).astype(int)
    controls=json.loads((Path(root)/'prepared/controls.json').read_text())
    ci=[controls['traits'].index(a) for a in AXES];ref=controls['backgrounds']['triple']
    raw_axes=points*np.array(ref['sd'])[ci]+np.array(ref['mean'])[ci]
    physical=(raw_axes[:,:,0]>0)&(raw_axes[:,:,1]>=1)
    novelty=np.array([np.mean([tuple(cell) not in occupied and valid for cell,valid in zip(row,ok)]) for row,ok in zip(bins,physical)])
    q05,q95=np.quantile(points,[.05,.95],axis=1)
    rows=pd.DataFrame({'gene':genes,'background':'triple','predicted_area':points[:,:,0].mean(1),
        'predicted_elongation':points[:,:,1].mean(1),'area_q05':q05[:,0],'area_q95':q95[:,0],
        'elongation_q05':q05[:,1],'elongation_q95':q95[:,1],
        'probability_unoccupied_bin':novelty,'invalid_axis_sample_fraction':1-physical.mean(1),'mean_interval_width':(q95-q05).mean(1),
        'status':'unmeasured genotype combination; viability untested'})
    rows=rows.sort_values(['probability_unoccupied_bin','gene'],ascending=[False,True])
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows.to_csv(out/'candidates.tsv',sep='\t',index=False)
    np.savez_compressed(out/'candidate_profiles.npz',genes=np.array(genes),traits=np.array(bundle['traits']),
        mean=samples.mean(1),q05=np.quantile(samples,.05,axis=1),q95=np.quantile(samples,.95,axis=1),axis_samples=points)
    info={'candidate_count':len(genes),'model':model,'training_run':str(Path(run_dir).resolve()),
        'training_config_sha256':sha256(Path(run_dir)/'config.json'),'selection':'single deletion measured; triple-background combination unmeasured',
        'novelty_definition':'fraction of all predictive samples with positive area and axis ratio >=1 in 1-WT-SD area/elongation bins unoccupied by measured triple-background cultures',
        'warnings':['Sampling bias and batch effects can create unoccupied regions.',
            'High novelty probability is not confidence, viability, or validated extrapolation.',
            'No measured target exists here: candidates require experimental validation.']}
    save_json(out/'manifest.json',info)
    table=rows.to_html(index=False,float_format=lambda x:f'{x:.3f}',escape=True,classes='data')
    # All candidates remain available in the sortable/searchable table and CSV; no hand-picked examples.
    doc='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Yeast exploration candidates</title><style>body{font:14px system-ui;background:#111c28;color:#d7e4ed;margin:32px}h1{font-size:30px}p{max-width:1000px;line-height:1.6;color:#a8bac8}input{padding:10px;background:#203247;color:white;border:1px solid #657a8a}table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border-bottom:1px solid #304354;text-align:right}td:first-child,th:first-child{text-align:left}th{cursor:pointer;position:sticky;top:0;background:#203247}.wrap{overflow:auto;max-height:72vh}</style><h1>Unmeasured yeast genotype combinations</h1><p>__N__ candidates · __MODEL__ · each is a measured single deletion transferred computationally into the pdr1Δ pdr3Δ snq2Δ background. These combinations have no measured morphology in the source tables. Viability is untested.</p><p>Novelty is predictive mass in unoccupied 1-reference-SD bins of area and elongation, excluding samples with nonpositive area or axis ratio below one. It is not a confidence score. Check model validation before using rankings. Click a column heading to sort; search to inspect a gene.</p><input id="search" placeholder="Search gene or status"><div class="wrap">__TABLE__</div><script>const t=document.querySelector('table'),body=t.tBodies[0];document.getElementById('search').oninput=e=>{let q=e.target.value.toLowerCase();[...body.rows].forEach(r=>r.style.display=r.textContent.toLowerCase().includes(q)?'':'none')};[...t.tHead.rows[0].cells].forEach((h,i)=>h.onclick=()=>{let d=h.dataset.direction==='1'?-1:1;h.dataset.direction=d;[...body.rows].sort((a,b)=>{let x=a.cells[i].textContent,y=b.cells[i].textContent;return d*(Number.isFinite(+x)&&Number.isFinite(+y)?+x-+y:x.localeCompare(y))}).forEach(r=>body.appendChild(r))});</script></html>'''
    plot_data={'observed':finite.tolist(),'candidates':json.loads(rows.to_json(orient='records'))}
    plot="""<section><p>Gray: measured altered-background cultures. Orange: predicted means for unmeasured combinations. Search a gene to highlight its 90% intervals. The map shows the central 99% of measured coordinates; the table retains every candidate.</p><canvas id="candidate-map" style="width:100%;height:390px;max-width:1100px"></canvas><p id="map-detail">Click a candidate to inspect it.</p></section>"""
    script="""<script>
const P=__PLOT_DATA__,c=document.getElementById('candidate-map'),cx=c.getContext('2d');let drawn=[];
const quant=(a,q)=>a.slice().sort((x,y)=>x-y)[Math.floor((a.length-1)*q)];
let xs=P.observed.map(p=>p[0]),ys=P.observed.map(p=>p[1]);
const bounds=[quant(xs,.005)-1,quant(xs,.995)+1,quant(ys,.005)-1,quant(ys,.995)+1];
function paint(){let w=c.clientWidth,h=390,d=window.devicePixelRatio||1;c.width=w*d;c.height=h*d;cx.setTransform(d,0,0,d,0,0);cx.clearRect(0,0,w,h);
const q=document.getElementById('search').value.toUpperCase();let view=bounds.slice();if(q){P.candidates.filter(r=>r.gene.includes(q)).forEach(r=>{view=[Math.min(view[0],r.area_q05-1),Math.max(view[1],r.area_q95+1),Math.min(view[2],r.elongation_q05-1),Math.max(view[3],r.elongation_q95+1)]})}const [a,b,e,f]=view,X=x=>60+(x-a)/(b-a)*(w-85),Y=y=>h-45-(y-e)/(f-e)*(h-70);
cx.font='11px system-ui';cx.fillStyle='#a8bac8';cx.strokeStyle='#304354';
for(let i=0;i<=5;i++){let x=a+(b-a)*i/5,y=e+(f-e)*i/5;cx.beginPath();cx.moveTo(X(x),25);cx.lineTo(X(x),h-45);cx.moveTo(60,Y(y));cx.lineTo(w-25,Y(y));cx.stroke();cx.fillText(x.toFixed(1),X(x)-8,h-26);cx.fillText(y.toFixed(1),20,Y(y)+4)}
cx.fillText('Cell area (reference SD)',w/2-60,h-5);cx.save();cx.translate(10,h/2+55);cx.rotate(-Math.PI/2);cx.fillText('Elongation (reference SD)',0,0);cx.restore();
cx.save();cx.beginPath();cx.rect(60,25,w-85,h-70);cx.clip();
drawn=[];
function dot(x,y,color,alpha,r){cx.globalAlpha=alpha;cx.fillStyle=color;cx.beginPath();cx.arc(X(x),Y(y),r,0,7);cx.fill()}
P.observed.forEach(p=>dot(p[0],p[1],'#a8bac8',.22,2));
P.candidates.forEach(r=>{let match=!q||r.gene.includes(q);dot(r.predicted_area,r.predicted_elongation,'#ffb36b',match?.65:.025,match&&q?6:2.5);if(match){drawn.push({x:X(r.predicted_area),y:Y(r.predicted_elongation),r});if(q){cx.globalAlpha=.8;cx.strokeStyle='#ffb36b';cx.beginPath();cx.moveTo(X(r.area_q05),Y(r.predicted_elongation));cx.lineTo(X(r.area_q95),Y(r.predicted_elongation));cx.moveTo(X(r.predicted_area),Y(r.elongation_q05));cx.lineTo(X(r.predicted_area),Y(r.elongation_q95));cx.stroke()}}});cx.restore();cx.globalAlpha=1}
c.onclick=ev=>{let rect=c.getBoundingClientRect(),x=ev.clientX-rect.left,y=ev.clientY-rect.top,p=drawn.reduce((a,b)=>Math.hypot(b.x-x,b.y-y)<Math.hypot(a.x-x,a.y-y)?b:a,{x:Infinity,y:Infinity});if(!p.r||Math.hypot(p.x-x,p.y-y)>15)return;document.getElementById('search').value=p.r.gene;document.getElementById('search').dispatchEvent(new Event('input'));document.getElementById('map-detail').textContent=p.r.gene+' · predicted mass in unoccupied bins: '+(p.r.probability_unoccupied_bin*100).toFixed(1)+'% · viability untested'};
document.getElementById('search').addEventListener('input',paint);window.addEventListener('resize',paint);paint();</script>"""
    doc=doc.replace('<div class="wrap">',plot+'<div class="wrap">').replace('</body>','')
    doc=doc.replace('</html>',script.replace('__PLOT_DATA__',json.dumps(plot_data,allow_nan=False))+'</html>')
    (out/'atlas.html').write_text(doc.replace('__N__',str(len(genes))).replace('__MODEL__',html.escape(model)).replace('__TABLE__',table))
    return info
