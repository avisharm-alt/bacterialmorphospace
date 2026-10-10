"""CLI: python -m yeast_morphology.cli --help."""
import argparse,json
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description='Yeast genotype-conditioned morphology benchmark')
    p.add_argument('--data-dir',default='data/yeast')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('fetch');sub.add_parser('prepare')
    s=sub.add_parser('split');s.add_argument('--out',default='data/yeast/splits');s.add_argument('--homology-groups');s.add_argument('--seed',type=int,default=20261010)
    s=sub.add_parser('train');s.add_argument('--task',choices=['gene','composition','background'],default='composition')
    s.add_argument('--split-dir',default='data/yeast/splits');s.add_argument('--out',required=True)
    s.add_argument('--epochs',type=int,default=120);s.add_argument('--latent-dim',type=int,default=32)
    s.add_argument('--samples',type=int,default=32);s.add_argument('--seed',type=int,default=20261010)
    s.add_argument('--device',choices=['auto','cpu','cuda','mps'],default='auto');s.add_argument('--embeddings')
    s.add_argument('--max-seconds',type=int,default=300);s.add_argument('--models',default='gaussian,diffusion')
    s=sub.add_parser('predict');s.add_argument('--run',required=True);s.add_argument('--genes',required=True)
    s.add_argument('--background',choices=['single','triple'],default='single');s.add_argument('--model',choices=['ridge','gaussian','diffusion'],default='diffusion')
    s.add_argument('--out',default='reports/yeast/predictions');s.add_argument('--samples',type=int,default=64)
    s.add_argument('--device',default='auto');s.add_argument('--embeddings')
    s=sub.add_parser('report');s.add_argument('--run',required=True)
    s=sub.add_parser('embed');s.add_argument('--out',default='data/yeast/esm2.npz');s.add_argument('--device',default='auto');s.add_argument('--max-seconds',type=int,default=900)
    s=sub.add_parser('atlas');s.add_argument('--run',required=True);s.add_argument('--out',default='reports/yeast/atlas');s.add_argument('--embeddings');s.add_argument('--model',choices=['ridge','gaussian','diffusion'],default='diffusion');s.add_argument('--samples',type=int,default=32);s.add_argument('--device',default='auto');s.add_argument('--max-genes',type=int,default=0)
    a=p.parse_args(argv)
    from .data import fetch,prepare
    if a.command=='fetch': result=fetch(a.data_dir)
    elif a.command=='prepare': result=prepare(a.data_dir)
    elif a.command=='split':
        from .splits import freeze
        result=freeze(a.data_dir,a.out,a.seed,a.homology_groups)
    elif a.command=='train':
        from .experiment import run
        result=run(a.data_dir,a.split_dir,a.out,a.task,a.epochs,a.latent_dim,a.samples,a.seed,a.device,a.embeddings,a.max_seconds,tuple(filter(None,a.models.split(','))))
        result={'out':a.out,'models':result['models'],'seconds':result['seconds']}
    elif a.command=='predict':
        from .experiment import predict
        result=predict(a.data_dir,a.run,a.genes.split(','),a.background,a.model,a.out,a.samples,device=a.device,embeddings=a.embeddings)
    elif a.command=='atlas':
        from .atlas import build
        result=build(a.data_dir,a.run,a.out,a.embeddings,a.model,a.samples,a.device,a.max_genes)
    elif a.command=='embed':
        from .embeddings import embed
        result=embed(a.data_dir,a.out,a.device,a.max_seconds)
    else:
        from .report import render
        render(a.run);result=str(Path(a.run)/'map.html')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
