"""Reproducible training/evaluation orchestration and inference."""
from __future__ import annotations
import importlib.metadata, json, pickle, platform, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data import AXES, load, proteins, sequence_features, save_json, sha256
from .models import Representation, fit_ridge, ridge_samples, fit_neural, neural_samples, device_for
from .metrics import per_row_scores, paired_bootstrap

def run(root,split_dir,out,task='composition',epochs=120,latent_dim=32,n_samples=32,seed=20261010,
        device='auto',embeddings=None,max_seconds=300,models=('gaussian','diffusion')):
    if n_samples<4: raise ValueError('At least 4 predictive samples required')
    torch.set_num_threads(4)
    start=time.monotonic(); root=Path(root); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    x,y,traits,meta=load(root,embeddings)
    splitpath=Path(split_dir)/f'{task}.tsv'
    split=pd.read_csv(splitpath,sep='\t')
    manifest=json.loads((Path(split_dir)/'manifest.json').read_text())
    if manifest['dataset_sha256']!=sha256(root/'prepared/dataset.npz'): raise ValueError('Dataset differs from frozen split')
    if manifest[task]['sha256']!=sha256(splitpath): raise ValueError('Split modified since freezing')
    if not split.id.equals(meta.id): raise ValueError('Split rows do not match dataset')
    if not split.gene.equals(meta.gene) or not split.background.equals(meta.background): raise ValueError('Split metadata mismatch')
    ix={p:np.flatnonzero(split.partition==p) for p in ['train','val','test']}
    for p in ix:
        if len(ix[p])<10: raise ValueError(f'Too few {p} rows')
    config={'task':task,'epochs':epochs,'latent_dim':latent_dim,'samples':n_samples,'seed':seed,
        'device':device_for(device),'models':list(models),'max_seconds_per_model':max_seconds,
        'dataset_sha256':manifest['dataset_sha256'],'split_sha256':sha256(splitpath),
        'features':'esm2' if embeddings else 'protein_composition',
        'embedding_sha256':sha256(embeddings) if embeddings else None}
    # Fingerprint every module so a stale checkpoint never silently survives code changes.
    config['code_hashes']={p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    configpath=out/'config.json'
    if configpath.exists() and json.loads(configpath.read_text())!=config:
        raise ValueError('Output directory contains a different run; choose a new --out')
    save_json(configpath,config)
    print(f'Loaded {len(meta)} profiles; fitting training representation',flush=True)
    rep=Representation().fit(x[ix['train']],y[ix['train']],latent_dim)
    xx=rep.x(x);zz=rep.y(y)
    latent={p:zz[ix[p]] for p in ix}; inputs={p:xx[ix[p]] for p in ix}
    kept=[t for t,k in zip(traits,rep.keep) if k]
    if not set(AXES)<=set(kept): raise ValueError('Required map axes missing from training targets')
    axes=[kept.index(a) for a in AXES]
    train,valid,test=ix['train'],ix['val'],ix['test']
    ytest=y[test][:,rep.keep]
    scaled=rep.standardized(y[test])
    oracle=rep.decode(latent['test'])
    oracle_mse=float(np.nanmean(((oracle-ytest)/rep.yscale.scale_)**2))
    ridge,residual,ridgestats=fit_ridge(inputs['train'],latent['train'],inputs['val'],latent['val'])
    rng=np.random.default_rng(seed)
    # Shuffle labels within background to retain the coarse background distribution.
    shuffled=latent['train'].copy()
    trainbg=meta.background.iloc[train].to_numpy()
    for bg in np.unique(trainbg):
        ii=np.flatnonzero(trainbg==bg); shuffled[ii]=shuffled[rng.permutation(ii)]
    nullridge,nullres,nullstats=fit_ridge(inputs['train'],shuffled,inputs['val'],latent['val'])
    with (out/'bundle.pkl').open('wb') as f:
        pickle.dump({'representation':rep,'ridge':ridge,'residual':residual,'traits':kept,
            'feature_kind':config['features'],'config':config,'train_ids':meta.id.iloc[train].tolist(),
            'train_backgrounds':sorted(meta.background.iloc[train].unique().tolist()),
            'train_sequences':meta.sequence_group.iloc[train].tolist()},f)
    stats={'ridge':ridgestats,'shuffled_ridge':nullstats}
    nets={}
    for kind in models:
        if kind not in ('gaussian','diffusion'): raise ValueError(kind)
        path=out/f'{kind}.pt'
        if path.exists(): checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        else:
            checkpoint=fit_neural(kind,inputs['train'],latent['train'],inputs['val'],latent['val'],
                epochs,seed,device, max_seconds=max_seconds)
            torch.save(checkpoint,path)
        nets[kind]=checkpoint
        stats[kind]={k:v for k,v in checkpoint.items() if k not in ['state','history']}
        save_json(out/f'{kind}_history.json',checkpoint['history'])
    save_json(out/'training.json',stats)
    wrong=inputs['test'].copy()
    for bg in meta.background.iloc[test].unique():
        ii=np.flatnonzero(meta.background.iloc[test].to_numpy()==bg)
        # Cyclic shuffle has no fixed point for groups > 1.
        order=rng.permutation(ii); wrong[order]=inputs['test'][np.roll(order,1)]
    predictions={
        'mean':np.broadcast_to(np.nanmean(y[train][:,rep.keep],axis=0),(len(test),n_samples,len(kept))).copy(),
        'empirical':rep.decode(latent['train'][rng.integers(len(train),size=(len(test),n_samples))]),
        'raw_empirical':np.where(np.isfinite(y[train][:,rep.keep]),y[train][:,rep.keep],rep.median)[rng.integers(len(train),size=(len(test),n_samples))],
        'ridge':rep.decode(ridge_samples(ridge,residual,inputs['test'],n_samples,seed)),
        'shuffled_ridge':rep.decode(ridge_samples(nullridge,nullres,inputs['test'],n_samples,seed))}
    for kind,net in nets.items():
        predictions[kind]=rep.decode(neural_samples(net,inputs['test'],n_samples,seed,device))
        predictions[f'{kind}_wrong_genotype']=rep.decode(neural_samples(net,wrong,n_samples,seed,device))
    results={}; scores={}; per_gene=[]
    for name,samples in predictions.items():
        s=(samples-rep.yscale.mean_)/rep.yscale.scale_
        rows=per_row_scores(scaled,s); scores[name]=rows
        r={k:float(v.mean()) for k,v in rows.items()}
        # This is an explicitly phenotype-selected test region, so report both prior and conditional mass.
        axis_samples=samples[:,:,axes]
        joint=(axis_samples>1).all(axis=2)
        targetjoint=(ytest[:,axes]>1).all(axis=1)
        r['mean_probability_high_area_high_elongation']=float(joint.mean())
        r['joint_brier']=float(np.mean((joint.mean(1)-targetjoint)**2))
        r['axis_mse_wt_units']=float(np.nanmean((axis_samples.mean(1)-ytest[:,axes])**2))
        r['test_count']=len(test)
        results[name]=r
        for i,idx in enumerate(test):
            per_gene.append({'id':meta.id.iloc[idx],'gene':meta.gene.iloc[idx],'background':meta.background.iloc[idx],
                'model':name,'observed_area':float(ytest[i,axes[0]]),'observed_elongation':float(ytest[i,axes[1]]),
                'predicted_area':float(axis_samples[i,:,0].mean()),'predicted_elongation':float(axis_samples[i,:,1].mean()),
                'area_q05':float(np.quantile(axis_samples[i,:,0],.05)),'area_q95':float(np.quantile(axis_samples[i,:,0],.95)),
                'elongation_q05':float(np.quantile(axis_samples[i,:,1],.05)),'elongation_q95':float(np.quantile(axis_samples[i,:,1],.95)),
                'combination_probability':float(joint[i].mean()),**{k:float(v[i]) for k,v in rows.items()}})
        np.savez_compressed(out/f'{name}_predictions.npz',ids=meta.id.iloc[test].to_numpy(dtype=str),
            traits=np.array(kept),samples=samples.astype('float32'),observed=ytest)
    for r in results.values(): r['skill_vs_mean']=1-r['mse']/results['mean']['mse']
    contrasts={}
    for name in ['ridge',*models]:
        base='empirical' if name=='ridge' else 'ridge'
        contrasts[f'{name}_minus_{base}']={metric:paired_bootstrap(scores[name][metric],scores[base][metric],
            split.sequence_group.iloc[test],seed) for metric in ['mse','crps','energy']}
        if name in models:
            contrasts[f'{name}_minus_wrong_genotype']={metric:paired_bootstrap(scores[name][metric],scores[f'{name}_wrong_genotype'][metric],
                split.sequence_group.iloc[test],seed) for metric in ['mse','crps','energy']}
    summary={'config':config,'counts':{p:len(v) for p,v in ix.items()},'retained_traits':len(kept),
        'pca_explained_variance':float(rep.pca.explained_variance_ratio_.sum()),
        'pca_oracle_test_mse':oracle_mse,'models':results,'paired_contrasts':contrasts,
        'seconds':time.monotonic()-start,'versions':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','torch']},
        'limitations':['Culture-level profiles, not images or single-cell distributions.',
            'Exact sequence groups only unless supplied homology groups; homolog transfer is possible.',
            'Genetic background is confounded with study.',
            'Predictive sample spread is not established epistemic uncertainty.',
            'Single-run exploratory results; no viability or causal generalization claim.']}
    save_json(out/'metrics.json',summary)
    pd.DataFrame(per_gene).to_csv(out/'per_gene.tsv',sep='\t',index=False)
    pd.DataFrame(results).T.to_csv(out/'metrics.tsv',sep='\t')
    mapdata=meta.copy();mapdata['partition']=split.partition
    mapdata['area']=y[:,traits.index(AXES[0])];mapdata['elongation']=y[:,traits.index(AXES[1])]
    mapdata.loc[mapdata.partition.isin(['train','val','test'])].to_csv(out/'observed_map.tsv',sep='\t',index=False)
    from .report import render
    render(out)
    return summary

def predict(root,run_dir,genes,background='single',model='diffusion',out='reports/yeast/predictions',
            n_samples=64,seed=20261010,device='auto',embeddings=None,compact=False):
    """Load only trusted locally generated model bundles; pickle is not an interchange format."""
    if background not in ['single','triple']: raise ValueError('Only the measured single/triple background designs are supported')
    torch.set_num_threads(4)
    run_dir=Path(run_dir);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (run_dir/'bundle.pkl').open('rb') as f: bundle=pickle.load(f)
    if background not in bundle.get('train_backgrounds',['single']): raise ValueError('Requested background was absent from model training')
    seqs=proteins(Path(root)/'raw/orf_trans_all.fasta.gz'); genes=[g.strip().upper() for g in genes]
    if any(g not in seqs for g in genes): raise ValueError('Every query must be a single systematic yeast ORF identifier in the reference FASTA')
    if background=='triple' and any(g in {'YGL013C','YBL005W','YDR011W'} for g in genes): raise ValueError('Query deletion is already part of the background')
    if bundle['feature_kind']=='esm2':
        if not embeddings: raise ValueError('This model requires its ESM embedding file')
        if sha256(embeddings)!=bundle['config']['embedding_sha256']: raise ValueError('Embedding file differs from training')
        e=np.load(embeddings,allow_pickle=False);lookup={g:i for i,g in enumerate(e['genes'])}
        if any(g not in lookup for g in genes): raise ValueError('Query gene lacks ESM embedding')
        vectors=e['embeddings']
        x=vectors[[lookup[g] for g in genes]]
    else: x=np.stack([sequence_features(seqs[g]) for g in genes])
    x=np.c_[x,np.full(len(genes),float(background=='triple'))]
    rep=bundle['representation'];xx=rep.x(x)
    if model=='ridge': z=ridge_samples(bundle['ridge'],bundle['residual'],xx,n_samples,seed)
    elif model in ['gaussian','diffusion']:
        ckpt=torch.load(run_dir/f'{model}.pt',map_location='cpu',weights_only=False)
        z=neural_samples(ckpt,xx,n_samples,seed,device)
    else: raise ValueError('Unsupported model')
    samples=rep.decode(z)
    if compact: return samples,bundle
    np.savez_compressed(out/'samples.npz',genes=np.array(genes),traits=np.array(bundle['traits']),samples=samples)
    mean=samples.mean(1);lo,hi=np.quantile(samples,[.05,.95],axis=1)
    records=[]
    for i,g in enumerate(genes):
        for j,t in enumerate(bundle['traits']):
            records.append({'gene':g,'background':background,'trait':t,'mean_wt_sd':float(mean[i,j]),
                'q05_wt_sd':float(lo[i,j]),'q95_wt_sd':float(hi[i,j]),
                'training_genotype':f'{background}:{g}' in bundle['train_ids'],'status':'model prediction; viability untested'})
    pd.DataFrame(records).to_csv(out/'profiles.tsv',sep='\t',index=False)
    save_json(out/'manifest.json',{'run':str(run_dir.resolve()),'model':model,'genes':genes,'background':background,
        'samples':n_samples,'seed':seed,'training_config':bundle['config'],
        'warning':'Only defined deletion/background inputs. Predictions do not establish viability or biological possibility.'})
    return str(out/'profiles.tsv')
