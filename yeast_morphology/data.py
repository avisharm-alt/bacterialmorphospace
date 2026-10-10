"""Verified public data ingestion and phenotype-free genetic representations."""
from __future__ import annotations
import gzip, hashlib, json, re, urllib.request
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

SOURCES = json.loads(Path(__file__).with_name('sources.json').read_text())
AA = 'ACDEFGHIKLMNPQRSTVWY'
AXES = ['C11-1_A', 'C115_A']
BACKGROUND_GENES = {'YGL013C', 'YBL005W', 'YDR011W'}  # PDR1, PDR3, SNQ2

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()

def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def fetch(root):
    raw = Path(root) / 'raw'; raw.mkdir(parents=True, exist_ok=True)
    records = []
    for name, spec in SOURCES.items():
        dest = raw / name
        if not dest.exists():
            req = urllib.request.Request(spec['url'], headers={'User-Agent': 'YeastMorphologyResearch/0.1'})
            tmp = dest.with_suffix(dest.suffix + '.part')
            try:
                with urllib.request.urlopen(req, timeout=120) as response, tmp.open('wb') as out:
                    for chunk in iter(lambda: response.read(1 << 20), b''): out.write(chunk)
                if sha256(tmp) != spec['sha256']: raise ValueError(f'Source changed: {name}; review before updating source lock')
                tmp.replace(dest)
            finally:
                if tmp.exists(): tmp.unlink()
        actual = sha256(dest)
        if actual != spec['sha256']: raise ValueError(f'Checksum mismatch: {dest}')
        records.append(dict(file=name, **spec, bytes=dest.stat().st_size))
    save_json(raw / 'manifest.json', {'verified_utc': datetime.now(timezone.utc).isoformat(), 'sources': records,
        'transport_note': 'Legacy SCMD HTTP is published by authors; TLS verification is never disabled. Hashes pin first retrieval, not an author-signed checksum.'})
    return records

def read_table(path):
    df = pd.read_csv(path, sep='\t', index_col=0)
    df.index = df.index.astype(str).str.upper().str.strip()
    if df.columns.duplicated().any(): raise ValueError('Duplicate trait columns')
    df = df.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    # CalMorph descriptors are nonnegative; negative legacy missing-value sentinels are not phenotypes.
    return df.mask(df < 0)

def proteins(path):
    result = {}; gene = None; chunks = []
    opener = gzip.open if str(path).endswith('.gz') else open
    def flush():
        if gene:
            seq = ''.join(chunks).rstrip('*').upper()
            if gene in result and result[gene] != seq: raise ValueError(f'Conflicting FASTA entries for {gene}')
            result[gene] = seq
    with opener(path, 'rt') as f:
        for line in f:
            if line.startswith('>'):
                flush(); chunks = []
                match = re.search(r'\bY[A-P][LR]\d{3}[CW](?:-[A-Z])?\b', line.upper())
                gene = match.group() if match else None
            else: chunks.append(line.strip())
        flush()
    return result

def sequence_features(seq):
    """421 fixed features; no corpus fitting, gene names, or functional labels."""
    ids = {a:i for i,a in enumerate(AA)}
    mono = np.zeros(20); di = np.zeros(400)
    for a in seq:
        if a in ids: mono[ids[a]] += 1
    for a,b in zip(seq, seq[1:]):
        if a in ids and b in ids: di[20*ids[a]+ids[b]] += 1
    return np.r_[np.log1p(len(seq)), mono/max(1, mono.sum()), di/max(1, di.sum())].astype('float32')

def prepare(root):
    root = Path(root); fetch(root)
    seqs = proteins(root/'raw/orf_trans_all.fasta.gz')
    singles = read_table(root/'raw/mt4718data.tsv')
    quads = read_table(root/'raw/quad1982data.tsv')
    traits = list(singles.columns)
    if set(quads.columns) != set(traits): raise ValueError('Cross-study trait mismatch')
    frames = []; meta = []; stats = {}; controls = {}
    for background, df, wtname in [('single', singles, 'wt122'), ('triple', quads, 'wt749')]:
        wt = read_table(root/f'raw/{wtname}data.tsv')[traits]
        mean = wt.mean(); sd = wt.std(ddof=1)
        floor = np.maximum(np.abs(mean.to_numpy()) * 1e-6, 1e-8)
        sd = pd.Series(np.maximum(sd, floor), index=traits)
        controls[background] = {'mean':mean.tolist(), 'sd':sd.tolist(), 'n':len(wt)}
        missing_genes = sorted(set(df.index)-set(seqs))
        df = df.loc[df.index.isin(seqs), traits].groupby(level=0, sort=True).median()
        # Only measured values; imputation is deferred until AFTER splitting.
        effect = (df - mean) / sd
        stats[background] = {'source_rows':len(singles if background=='single' else quads),
            'matched_genes':len(df), 'missing_sequence_genes':missing_genes,
            'missing_target_entries':int(effect.isna().sum().sum()), 'control_cultures':len(wt)}
        for gene in df.index:
            meta.append({'id':f'{background}:{gene}', 'gene':gene, 'background':background,
                'sequence_group':hashlib.sha256(seqs[gene].encode()).hexdigest(), 'protein_length':len(seqs[gene])})
        frames.append(effect.to_numpy(dtype='float32'))
    metadata = pd.DataFrame(meta)
    features = np.stack([sequence_features(seqs[g]) for g in metadata.gene])
    features = np.c_[features, (metadata.background=='triple').to_numpy().astype('float32')]
    y = np.concatenate(frames)
    out = root/'prepared'; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'dataset.npz', x=features, y=y, traits=np.array(traits), ids=metadata.id.to_numpy(dtype=str))
    metadata.to_csv(out/'metadata.tsv', sep='\t', index=False)
    save_json(out/'controls.json', {'traits':traits, 'backgrounds':controls})
    audit = {'datasets':stats, 'n_traits':len(traits), 'n_features':features.shape[1],
        'n_rows':len(metadata), 'exact_sequence_groups':int(metadata.sequence_group.nunique()),
        'duplicate_sequence_extra_genes':int(metadata.drop_duplicates('gene').shape[0]-metadata.sequence_group.nunique()),
        'sources':{k:v['sha256'] for k,v in SOURCES.items()}, 'dataset_sha256':sha256(out/'dataset.npz'),
        'target_unit':'WT-standardized culture-level aggregate morphology, not single-cell measurements'}
    save_json(out/'audit.json', audit)
    return audit

def load(root, embeddings=None):
    root=Path(root); d=np.load(root/'prepared/dataset.npz', allow_pickle=False)
    metadata=pd.read_csv(root/'prepared/metadata.tsv', sep='\t')
    if not np.array_equal(d['ids'],metadata.id): raise ValueError('Metadata and arrays are misaligned')
    x=d['x'].copy()
    if embeddings:
        e=np.load(embeddings, allow_pickle=False)
        lookup={g:i for i,g in enumerate(e['genes'])}
        absent=sorted(set(metadata.gene)-set(lookup))
        if absent: raise ValueError(f'Embedding file is incomplete: {len(absent)} genes missing')
        vectors=e['embeddings']  # Decompress once, rather than once per genotype.
        x=np.c_[vectors[[lookup[g] for g in metadata.gene]],x[:,-1]]
    return x,d['y'].copy(),d['traits'].tolist(),metadata
