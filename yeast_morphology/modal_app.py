"""Bounded Modal jobs. One T4, no retries or sweeps; outputs retained in a dedicated volume.

modal run -m yeast_morphology.modal_app --task composition --out reports/yeast/runs/modal-composition
modal run -m yeast_morphology.modal_app --mode embed --out data/yeast/esm2.npz
"""
from pathlib import Path
import modal

HERE=Path(__file__).resolve().parents[1]
app=modal.App('yeast-genotype-morphology')
volume=modal.Volume.from_name('yeast-morphology',create_if_missing=True)
image=(modal.Image.debian_slim(python_version='3.12')
       .pip_install('numpy>=1.26,<3','pandas>=2.2,<4','scipy>=1.12,<2','scikit-learn>=1.4,<2',
                    'torch>=2.3,<3','matplotlib>=3.8,<4','fair-esm==2.0.0')
       .env({'MPLCONFIGDIR':'/tmp/mpl','TORCH_HOME':'/persist/torch'})
       .add_local_dir(HERE/'yeast_morphology','/root/yeast_morphology',ignore=['__pycache__'])
       .add_local_dir(HERE/'data/yeast/prepared','/inputs/prepared')
       .add_local_dir(HERE/'data/yeast/splits','/inputs/splits')
       .add_local_file(HERE/'data/yeast/raw/orf_trans_all.fasta.gz','/inputs/raw/orf_trans_all.fasta.gz'))

@app.function(image=image,gpu='T4',cpu=4,memory=8192,timeout=1200,retries=0,max_containers=1,
              scaledown_window=2,volumes={'/persist':volume})
def job(mode:str,task:str,epochs:int,run_name:str,use_esm:bool):
    import shutil,json,tarfile,io
    from yeast_morphology.data import sha256
    if mode=='embed':
        from yeast_morphology.embeddings import embed
        try: manifest=embed('/inputs','/persist/esm2.npz','cuda',max_seconds=1000)
        finally: volume.commit()
        return {'manifest':manifest,'embedding':Path('/persist/esm2.npz').read_bytes()}
    from yeast_morphology.experiment import run
    out=Path('/persist/runs')/run_name
    embeddings='/persist/esm2.npz' if use_esm else None
    if use_esm and not Path(embeddings).exists(): raise ValueError('Run --mode embed first')
    try: metrics=run('/inputs','/inputs/splits',out,task=task,epochs=epochs,device='cuda',embeddings=embeddings,max_seconds=300)
    finally: volume.commit()
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as tf:
        for path in out.iterdir():
            if path.is_file():tf.add(path,arcname=path.name)
    return {'archive':stream.getvalue(),'seconds':metrics['seconds'],'models':metrics['models']}

@app.local_entrypoint()
def main(mode:str='train',task:str='composition',epochs:int=120,out:str='reports/yeast/runs/modal-composition',use_esm:bool=False):
    import io,json,tarfile
    if mode not in ['train','embed']:raise ValueError('mode must be train or embed')
    if task not in ['gene','composition','background']:raise ValueError('Unknown task')
    if not 1<=epochs<=300:raise ValueError('Epochs must be 1..300; no unbounded jobs')
    dest=Path(out)
    if mode=='train' and dest.exists() and any(dest.iterdir()):raise ValueError('Use a new output directory')
    result=job.remote(mode,task,epochs,dest.name,use_esm)
    if mode=='embed':
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(result['embedding'])
        dest.with_suffix('.json').write_text(json.dumps(result['manifest'],indent=2))
        print(json.dumps(result['manifest'],indent=2))
    else:
        dest.mkdir(parents=True,exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(result['archive']),mode='r:gz') as tf:tf.extractall(dest,filter='data')
        print(json.dumps({'out':str(dest),'seconds':result['seconds'],'models':result['models']},indent=2))
