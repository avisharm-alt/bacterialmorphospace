"""Optional frozen ESM-2 sequence features. No phenotype labels enter this module."""
import time,json
from pathlib import Path
import numpy as np
from .data import proteins,save_json,sha256

def embed(root,out,device='auto',max_seconds=900):
    import torch,esm
    from .models import device_for
    from .data import load
    device=device_for(device);torch.set_num_threads(4)
    _,_,_,meta=load(root);genes=sorted(set(meta.gene))
    seqs=proteins(Path(root)/'raw/orf_trans_all.fasta.gz')
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    cache=out.with_suffix('.cache');cache.mkdir(exist_ok=True)
    fingerprint={'model':'esm2_t12_35M_UR50D','sequence_sha256':sha256(Path(root)/'raw/orf_trans_all.fasta.gz'),'chunk_size':1000,'noncanonical':'X'}
    cachemeta=cache/'manifest.json'
    if cachemeta.exists() and json.loads(cachemeta.read_text())!=fingerprint: raise ValueError('Embedding cache belongs to another input/model')
    save_json(cachemeta,fingerprint)
    model,alphabet=esm.pretrained.esm2_t12_35M_UR50D();model=model.eval().to(device)
    convert=alphabet.get_batch_converter();start=time.monotonic()
    for i,g in enumerate(genes):
        path=cache/f'{g}.npy'
        if path.exists():continue
        # All residues are represented; long proteins use non-overlapping 1,000-aa windows.
        seq=''.join(a if a in 'ACDEFGHIKLMNPQRSTVWYX' else 'X' for a in seqs[g]);total=np.zeros(480,dtype='float64');length=0
        with torch.no_grad():
            for pos in range(0,len(seq),1000):
                part=seq[pos:pos+1000]
                _,_,tokens=convert([(g,part)])
                hidden=model(tokens.to(device),repr_layers=[12],return_contacts=False)['representations'][12][0,1:len(part)+1]
                total+=hidden.sum(0).cpu().numpy();length+=len(part)
        np.save(path,(total/max(1,length)).astype('float32'))
        if (i+1)%200==0: print(f'ESM-2: {i+1}/{len(genes)} proteins',flush=True)
        if time.monotonic()-start>max_seconds:
            raise TimeoutError(f'Embedding time cap reached; {cache} is resumable. No incomplete feature matrix published.')
    np.savez_compressed(out,genes=np.array(genes),embeddings=np.stack([np.load(cache/f'{g}.npy',allow_pickle=False) for g in genes]))
    manifest={'model':'esm2_t12_35M_UR50D','layer':12,'pooling':'residue-weighted mean over all nonoverlapping <=1000-aa chunks',
        'n_genes':len(genes),'features':480,'nonstandard_residue_policy':'Internal stops and noncanonical residues mapped to X; terminal stops removed by FASTA loader',
        'nonstandard_residues':sum(sum(a not in 'ACDEFGHIKLMNPQRSTVWYX' for a in seqs[g]) for g in genes),'sequence_sha256':sha256(Path(root)/'raw/orf_trans_all.fasta.gz'),
        'sha256':sha256(out),'seconds':time.monotonic()-start,'pretraining_note':'Yeast exposure is not excluded. No phenotype annotations used.'}
    save_json(out.with_suffix('.json'),manifest);return manifest
