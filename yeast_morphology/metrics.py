"""Proper sample scores with observed-entry masks; no imputed-target scoring."""
import numpy as np

def per_row_scores(y,samples):
    if samples.ndim!=3 or samples.shape[0]!=len(y) or samples.shape[2]!=y.shape[1]: raise ValueError('Shape mismatch')
    if not np.isfinite(samples).all(): raise ValueError('Non-finite predictions')
    valid=np.isfinite(y); counts=valid.sum(1)
    if (counts==0).any(): raise ValueError('Row has no observed targets')
    yy=np.where(valid,y,0)
    mean=samples.mean(1)
    mse=np.sum(np.where(valid,(mean-yy)**2,0),axis=1)/counts
    # Exact empirical CRPS in O(S log S), including diagonal sample pairs.
    sorted_samples=np.sort(samples,axis=1); n=samples.shape[1]
    weights=(2*np.arange(1,n+1)-n-1)[None,:,None]
    spread=(sorted_samples*weights).sum(1)/(n*n)
    crps=np.abs(samples-yy[:,None,:]).mean(1)-spread
    crps=np.where(valid,crps,0).sum(1)/counts
    lo,hi=np.quantile(samples,[.05,.95],axis=1)
    coverage=np.where(valid,(yy>=lo)&(yy<=hi),False).sum(1)/counts
    width=np.where(valid,hi-lo,0).sum(1)/counts
    # RMS Euclidean distances make energy scores comparable across missing-entry counts.
    distance=np.sqrt(np.where(valid[:,None,:],(samples-yy[:,None,:])**2,0).sum(2)/counts[:,None]).mean(1)
    pair=np.zeros(len(y))
    for i in range(n):
        pair+=np.sqrt(np.where(valid[:,None,:],(samples[:,i,None,:]-samples)**2,0).sum(2)/counts[:,None]).sum(1)
    energy=distance-.5*pair/(n*n)
    return {'mse':mse,'crps':crps,'energy':energy,'coverage90':coverage,'width90':width}

def paired_bootstrap(a,b,groups,seed=20261010,n_boot=1000):
    a=np.asarray(a);b=np.asarray(b);groups=np.asarray(groups)
    unique=np.unique(groups); rng=np.random.default_rng(seed)
    sums=np.array([(a[groups==g]-b[groups==g]).sum() for g in unique])
    counts=np.array([(groups==g).sum() for g in unique])
    ix=rng.integers(len(unique),size=(n_boot,len(unique)))
    boot=sums[ix].sum(1)/counts[ix].sum(1)
    return {'difference':float((a-b).mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
            'groups':len(unique),'resamples':n_boot,'interpretation':'paired cluster bootstrap; negative favors first model for loss scores'}
