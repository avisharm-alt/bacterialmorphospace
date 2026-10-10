"""Frozen splits; no models or fitted phenotype transforms are used here."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from .data import AXES, BACKGROUND_GENES, load, save_json, sha256

def group_partition(groups, seed=20261010):
    keys=sorted(set(groups),key=lambda g:hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest())
    n=len(keys); a=int(.7*n); b=int(.85*n)
    return {g:('train' if i<a else 'val' if i<b else 'test') for i,g in enumerate(keys)}

def make_split(meta, y, traits, task, seed=20261010):
    m=meta.copy(); m['partition']='unused'; groups=m.sequence_group
    if task=='gene':
        eligible=m.background=='single'; mapping=group_partition(groups[eligible],seed)
        m.loc[eligible,'partition']=groups[eligible].map(mapping)
    elif task=='composition':
        single=m.background=='single'
        axes=y[:,[traits.index(a) for a in AXES]]
        complete=np.isfinite(axes).all(axis=1)
        test=single & complete & (axes>1).all(axis=1)
        blocked=set(groups[test]); eligible=single & ~groups.isin(blocked)
        mapping=group_partition(groups[eligible],seed)
        # Reuse group hash ordering; 15% validation, all other eligible groups train.
        m.loc[eligible,'partition']=groups[eligible].map(lambda g:'val' if mapping[g]=='val' else 'train')
        m.loc[test,'partition']='test'
        m.loc[single & groups.isin(blocked) & ~test,'partition']='buffer'
    elif task=='background':
        eligible=(m.background=='triple') & ~m.gene.isin(BACKGROUND_GENES)
        mapping=group_partition(groups[eligible],seed)
        m.loc[(m.background=='single') & ~m.gene.isin(BACKGROUND_GENES),'partition']='train'
        m.loc[eligible,'partition']=groups[eligible].map(mapping)
    else: raise ValueError(f'Unknown task {task}')
    for part in ['train','val','test']:
        if (m.partition==part).sum()<10: raise ValueError(f'Insufficient {part} rows')
    if task!='background':
        gs=[set(m.loc[m.partition==p,'sequence_group']) for p in ['train','val','test']]
        assert not(gs[0]&gs[1] or gs[0]&gs[2] or gs[1]&gs[2])
    if task=='composition':
        assert not (y[m.partition.isin(['train','val'])][:,[traits.index(a) for a in AXES]]>1).all(axis=1).any()
        trainaxes=y[m.partition=='train'][:,[traits.index(a) for a in AXES]]
        for j in [0,1]:
            if ((trainaxes[:,j]>1)&(trainaxes[:,1-j]<=1)).sum()<10:
                raise ValueError('Constituent morphology is not adequately represented in train')
    return m

def freeze(root, out, seed=20261010, homology_groups=None):
    _,y,traits,meta=load(root)
    if homology_groups:
        groups=pd.read_csv(homology_groups,sep='\t').set_index('gene')['group']
        if not set(meta.gene)<=set(groups.index): raise ValueError('Homology groups must cover every gene')
        external=meta.gene.map(groups).astype(str)
        if pd.DataFrame({'exact':meta.sequence_group,'external':external}).groupby('exact').external.nunique().max()>1:
            raise ValueError('Homology grouping splits identical protein sequences; merge those groups first')
        meta['sequence_group']=external
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    summary={}
    for task in ['gene','composition','background']:
        split=make_split(meta,y,traits,task,seed)
        dest=out/f'{task}.tsv'
        payload=split.to_csv(sep='\t',index=False)
        if dest.exists() and dest.read_text()!=payload: raise ValueError(f'Frozen split would change: {dest}; use a new directory')
        dest.write_text(payload)
        summary[task]={'counts':{k:int(v) for k,v in split.partition.value_counts().items()},'sha256':sha256(dest)}
    summary['seed']=seed
    summary['dataset_sha256']=sha256(Path(root)/'prepared/dataset.npz')
    summary['homology_groups_sha256']=sha256(homology_groups) if homology_groups else None
    save_json(out/'manifest.json',summary)
    return summary
